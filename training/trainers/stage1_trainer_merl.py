import torch
import torch.nn.functional as NF
import pytorch_lightning as pl
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from renderer import ForwardRenderer
import torchvision
import json
import cv2
import numpy as np
from tqdm import tqdm
import math
from models.emitter import RealAreaEmitter, ConstantEmitter, MultiAreaEmitter
from models.brdf import GreyPatchBRDF
import os
from utils.pose_refiner import GlobalHandEyeRefiner
from models.neural_brdf_refactored import MERLBRDF

# bitsandbytes compat: unwrap newer __bnb_optimizer_quant_state__ format so
# checkpoints saved with bnb >= 0.49 can be loaded by older bnb (e.g. 0.41.3).
try:
    import bitsandbytes as _bnb
    class Adam8bitCompat(_bnb.optim.Adam8bit):
        def load_state_dict(self, state_dict):
            for st in state_dict.get('state', {}).values():
                if isinstance(st, dict) and '__bnb_optimizer_quant_state__' in st:
                    st.update(st.pop('__bnb_optimizer_quant_state__'))
            return super().load_state_dict(state_dict)
except ImportError:
    Adam8bitCompat = None


class Stage1Trainer_MERL(pl.LightningModule):
    def __init__(self, cfg, material, gt_material, roughness, metallic):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.material = material
        self.freeze_decoder = cfg.model.freeze_decoder
        self.gt_material = gt_material
        self.gt_folder = cfg.gt_folder
        self.camera_factor = cfg.renderer.camera.linear_factor
        
        #self.latent_dim = cfg.material.latent_dim
        # Create a mapping from roughness-metallic pairs to train latent indices
        self.radiance_rgb_pairs = {}
        self.average_radiance = []
        self.emitter_calibration = cfg.model.emitter_calibration    
        self.latent_reg_weight = cfg.model.latent_reg_weight if hasattr(cfg.model, 'latent_reg_weight') else 1e-4
        self.visualize_uv = cfg.data.visualize_uv
        self.inference_lr = cfg.model.optimizer.inference_lr
        self.inference_steps = cfg.model.optimizer.inference_steps
        self.is_graypatch = material.__class__.__name__ == 'GreyPatchBRDF'
        self.grazing_ratio = getattr(cfg.model, 'grazing_ratio', 0.0)
        # Grazing mode: 'zero_exact' (legacy), 'near_zero_brdf', 'contribution_decay'.
        self.grazing_mode = getattr(cfg.model, 'grazing_mode', 'zero_exact')
        # near_zero_brdf concat-mode: c ~ U(cos_min, cos_max).
        self.grazing_cos_min = float(getattr(cfg.model, 'grazing_cos_min', 0.005))
        self.grazing_cos_max = float(getattr(cfg.model, 'grazing_cos_max', 0.03))
        # contribution_decay regularizer: separate knobs from the concat modes.
        self.grazing_decay_cos_min = float(getattr(cfg.model, 'grazing_decay_cos_min', 0.005))
        self.grazing_decay_cos_max = float(getattr(cfg.model, 'grazing_decay_cos_max', 0.2))
        self.grazing_decay_weight = float(getattr(cfg.model, 'grazing_decay_weight', 1.0))
        _decay_eps = getattr(cfg.model, 'grazing_decay_eps', None)
        if _decay_eps is None:
            try:
                _decay_eps = float(cfg.model.loss.recon_loss.log_space.logrel_eps)
            except Exception:
                _decay_eps = 1e-4
        self.grazing_decay_eps = float(_decay_eps)
        print("after latent reg weight")
        self.renderer = ForwardRenderer(cfg, self.material)
        self.gt_renderer = ForwardRenderer(cfg, self.gt_material)
        if cfg.data.handeye_refiner:
            self.handeye_refiner = GlobalHandEyeRefiner(sigma_t_mm=0.5, sigma_r_deg=0.1, json_path=cfg.data.metadata_path)
        else:
            self.handeye_refiner = None
        # self.emitter = PresetPointEmitter(
        #     read_from_metadata=True,
        #     metadata_path=os.path.join(self.cfg.metadata_path, 'emitter_metadata.json'),
        #     positions=None, 
        #     intensities=None
        # )
        self.img_hw = (cfg.renderer.camera.intrinsics.height, cfg.renderer.camera.intrinsics.width)

    # def gamma(self, x):
    #     mask = x <= 0.0031308
    #     ret = torch.empty_like(x)
    #     ret[mask] = 12.92 * x[mask]
    #     ret[~mask] = 1.055 * x[~mask].pow(1/2.4) - 0.055
    #     return ret
    def gamma(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert a tensor of linear-light RGB values to sRGB.
        Matches Blender's built-in OCIO conversion (Standard view-transform).

        Parameters
        ----------
        x : torch.Tensor
            *Linear* RGB values in **[0 … ∞)**. Negative values are clamped to 0.

        Returns
        -------
        torch.Tensor
            sRGB-encoded values in the display range **[0 … 1]**.
        """
        # --- constants taken from the official sRGB transfer function ---
        _A   = 0.055           # 1.055 - 1
        _K0  = 0.0031308       # linear-to-sRGB break-point
        _PHI = 1.0 / 2.4       # 0.416̅  = 1/γ

        x_lin = x.clamp(min=0.0)               # Blender never shows negative light
        low   = 12.92 * x_lin                  # linear segment
        high  = 1.055 * torch.pow(x_lin, _PHI) - _A

        return torch.where(x_lin <= _K0, low, high).clamp(0.0, 1.0)

    def tone_mapping(self, x):
        """
        Apply tone mapping to convert HDR image to LDR.
        
        Args:
            x (torch.Tensor): HDR image with values in range [0, 1]
            
        Returns:
            torch.Tensor: LDR image with tone mapping applied
        """
        # Simple Reinhard tone mapping: x / (1 + x)
        return x / (1 + x)
    
    def configure_optimizers(self):
        lr = self.hparams.model.optimizer.lr
        decoder_lr = getattr(self.hparams.model.optimizer, 'decoder_lr', lr)
        wd = self.hparams.model.optimizer.weight_decay
        opt_name = getattr(self.hparams.model.optimizer, 'name', 'Adam')

        embedding_params = list(self.material.point_latent_bank.parameters())
        embedding_set = set(embedding_params)
        decoder_params = [p for p in self.parameters() if p not in embedding_set]

        if self.freeze_decoder:
            for p in decoder_params:
                p.requires_grad = False
            print("Decoder frozen — optimising latent bank only.")

        dense_params = decoder_params if not self.freeze_decoder else []

        if opt_name == 'SGD':
            opt_groups = [{'params': embedding_params, 'lr': lr}]
            if len(dense_params) > 0:
                opt_groups.append({'params': dense_params, 'lr': decoder_lr})
            optimizer = torch.optim.SGD(opt_groups, momentum=0.9, weight_decay=wd)
            print(f"Using SGD (embedding lr={lr}, decoder lr={decoder_lr})")
            return optimizer

        elif opt_name == 'Adam':
            opt_groups = []
            if len(dense_params) > 0:
                opt_groups.append({'params': dense_params, 'lr': decoder_lr})
            opt_groups.append({'params': embedding_params, 'lr': lr})
            optimizer = torch.optim.Adam(opt_groups, betas=(0.9, 0.999), weight_decay=wd)
            print(f"Using Adam (embedding lr={lr}, decoder lr={decoder_lr})")
            return optimizer

        elif opt_name == 'Adam8bit':
            if Adam8bitCompat is None:
                raise RuntimeError("bitsandbytes not installed; cannot use Adam8bit")
            opt_groups = [{'params': embedding_params, 'lr': lr}]
            if len(dense_params) > 0:
                opt_groups.append({'params': dense_params, 'lr': decoder_lr})
            optimizer = Adam8bitCompat(opt_groups, betas=(0.9, 0.999), weight_decay=wd)
            print(f"Using Adam8bit (embedding lr={lr}, decoder lr={decoder_lr})")
            return optimizer

        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def load_pbr_texture(self, pbr_texture_path):
        pbr_folder = os.environ.get('ROBOCLOTH_PBR_TEXTURES', '/absolute/path/to/pbr_textures/fabric_pattern_07_4k/textures')  # PBR texture pack (not in the HF release)
        import cv2
        from PIL import Image
        import torchvision.transforms as T

        def read_img(fname):
            path = os.path.join(pbr_folder, fname)
            if path.endswith(".exr"):
                exr = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # H × W × C, float32
                # Convert BGR to RGB for OpenCV
                exr = exr[..., ::-1]

                tensor = torch.from_numpy(exr.copy())         # torch.float32
                # Check if it's only two channels and unsqueeze if needed
                if len(tensor.shape) == 2:
                    tensor = tensor.unsqueeze(2)  # Add channel dimension if missing
                return tensor.permute(2, 0, 1)                # [C,H,W]
            else:
                img = Image.open(path).convert('RGB')
                tensor = T.ToTensor()(img).float()
                return tensor if tensor.max() <= 1.0 else tensor / 255.0

        # Collect PBR texture data
        try:
            # Read all texture maps based on the image
            col_1 = read_img("fabric_pattern_07_col_1_4k.jpg")  # [3,H,W] - Color map
            ao = read_img("fabric_pattern_07_ao_4k.jpg")        # [3,H,W] - Ambient occlusion
            arm = read_img("fabric_pattern_07_arm_4k.jpg")      # [3,H,W] - (A, Roughness, Metalness)
            rough = read_img("fabric_pattern_07_rough_4k.exr")  # [1,H,W] - Roughness map (EXR)
            nor_dx = read_img("fabric_pattern_07_nor_dx_4k.exr") # [1,H,W] - Normal map X
            nor_gl = read_img("fabric_pattern_07_nor_gl_4k.exr") # [1,H,W] - Normal map GL
            
            
            # Combine all available channels into a texture
            # [Color (3) + AO (3) + ARM (3) + Roughness (1) + Normal DX (1) + Normal GL (1)] = 12 channels
            tex = torch.cat([
                col_1,                # RGB color (3 channels) 0:3
                ao,                   # Ambient occlusion (3 channels) 3:6
                arm,                  # ARM texture (3 channels) 6:9
                rough,                # Roughness map (1 channel) 9:10
                nor_dx,               # Normal map X (3 channel) 10:13
                nor_gl                # Normal map GL (3 channel) 13:16
            ], dim=0)  # [16,H,W]
            
            return tex.permute(1, 2, 0).contiguous().to(self.device)  # [H,W,6] => [U,V,6]
        except Exception as e:
            print(f"Error loading PBR textures: {e}")
            # Return a default texture if loading fails
            H, W = 1024, 1024
            return torch.ones(H, W, 6)
    
    def save_pbr_texture(self, output_dir, batch_idx, b):
        if self.hparams.material.type == "AnisotropicLatentTexturedModel" or self.hparams.material.type == "MipmapAniLatentTexturedModel": # Save normal and tangent map
            # Extract normal and tangent from latent texture (last 6 dimensions)
            # latent_texture = self.material.latent_texture.data[0]  # [latent_dim, H, W]
            
            # # Last 6 dimensions: normal (3) and tangent (3)
            # normal_map = latent_texture[-6:-3, :, :].permute(1, 2, 0)  # [H, W, 3]
            # tangent_map = latent_texture[-3:, :, :].permute(1, 2, 0)   # [H, W, 3]
            
            # torchvision.utils.save_image(
            #     normal_map.permute(2, 0, 1),
            #     os.path.join(output_dir, f'normal_map_{batch_idx}_{b}.png')
            # )
            # torchvision.utils.save_image(
            #     tangent_map.permute(2, 0, 1),
            #     os.path.join(output_dir, f'tangent_map_{batch_idx}_{b}.png')
            # )
            return
        else:
            # Check if material has prefilter option
            if self.hparams.material.prefliter:
                # Save the finest level PBR texture when using prefilter
                pbr_texture_data = self.material.pbr_texture.data[0]
                
                # Save albedo map (channels 0-3)
                pbr_albedo_map = pbr_texture_data[:, :, 0:3]
                torchvision.utils.save_image(
                    pbr_albedo_map.permute(2, 0, 1),
                    os.path.join(output_dir, f'pbr_albedo_map_{batch_idx}_{b}.png')
                )
                
                # Save normal map (channels 10-13)
                pbr_normal_map = pbr_texture_data[:, :, 10:13]
                torchvision.utils.save_image(
                    pbr_normal_map.permute(2, 0, 1),
                    os.path.join(output_dir, f'pbr_normal_map_{batch_idx}_{b}.png')
                )
                
                # save height map
                pbr_height_map = pbr_texture_data[:, :, 13:14]
                torchvision.utils.save_image(
                    pbr_height_map.permute(2, 0, 1),
                    os.path.join(output_dir, f'pbr_height_map_{batch_idx}_{b}.png')
                )
                
                # Save roughness map (channel 7)
                pbr_roughness_map = pbr_texture_data[:, :, 7:8]
                torchvision.utils.save_image(
                    pbr_roughness_map.permute(2, 0, 1),
                    os.path.join(output_dir, f'pbr_roughness_map_{batch_idx}_{b}.png')
                )
                
                # Save metallic map (channel 8)
                pbr_metallic_map = pbr_texture_data[:, :, 8:9]
                torchvision.utils.save_image(
                    pbr_metallic_map.permute(2, 0, 1),
                    os.path.join(output_dir, f'pbr_metallic_map_{batch_idx}_{b}.png')
                )
            else:
                # Save individual mipmap textures when not using prefilter
                if hasattr(self.material, 'mipmap_textures'):
                    for level, mipmap_texture in enumerate(self.material.mipmap_textures):
                        texture_data = mipmap_texture.data[0]
                        
                        # Save albedo mipmap
                        mipmap_albedo = texture_data[:, :, 0:3]
                        torchvision.utils.save_image(
                            mipmap_albedo.permute(2, 0, 1),
                            os.path.join(output_dir, f'mipmap_albedo_level_{level}_{batch_idx}_{b}.png')
                        )
                        
                        # Save normal mipmap
                        mipmap_normal = texture_data[:, :, 10:13]
                        torchvision.utils.save_image(
                            mipmap_normal.permute(2, 0, 1),
                            os.path.join(output_dir, f'mipmap_normal_level_{level}_{batch_idx}_{b}.png')
                        )
                        
                        # Save height mipmap
                        mipmap_height = texture_data[:, :, 13:14]
                        torchvision.utils.save_image(
                            mipmap_height.permute(2, 0, 1),
                            os.path.join(output_dir, f'mipmap_height_map_level_{level}_{batch_idx}_{b}.png')
                        )
                        
                        # Save roughness mipmap
                        mipmap_roughness = texture_data[:, :, 7:8]
                        torchvision.utils.save_image(
                            mipmap_roughness.permute(2, 0, 1),
                            os.path.join(output_dir, f'mipmap_roughness_level_{level}_{batch_idx}_{b}.png')
                        )
                        
                        # Save metallic mipmap
                        mipmap_metallic = texture_data[:, :, 8:9]
                        torchvision.utils.save_image(
                            mipmap_metallic.permute(2, 0, 1),
                            os.path.join(output_dir, f'mipmap_metallic_level_{level}_{batch_idx}_{b}.png')
                        )

    def loss_function(self, rgbs, rgbs_gt, vis):
        # Calculate per-pixel loss
        if self.hparams.model.loss.recon_loss.name == "l1":
            per_pix = torch.abs(rgbs[vis] - rgbs_gt.squeeze(0)[vis]).mean(dim=-1)
        elif self.hparams.model.loss.recon_loss.name == "l2":  # "l2"
            per_pix = torch.pow(rgbs[vis] - rgbs_gt.squeeze(0)[vis], 2).mean(dim=-1)
        elif self.hparams.model.loss.recon_loss.name == "normalized_l1":  # "normalized_l1"
            per_pix = torch.abs(rgbs[vis] - rgbs_gt.squeeze(0)[vis]).mean(dim=-1) / 65535.0
        else:  # "logrel" from paper
            rho_ref = getattr(
                self.hparams.model.loss.recon_loss.log_space, "logrel_ref", 0.5
            )  # reference reflectance/radiance
            eps = getattr(
                self.hparams.model.loss.recon_loss.log_space, "logrel_eps", 1e-3
            )  # fixed small constant

            ref = torch.as_tensor(rho_ref, dtype=rgbs.dtype, device=rgbs.device)

            def log_mapping(x):
                return torch.log((x + eps) / (ref + eps) + 1.0)

            per_pix_log = (log_mapping(rgbs) - log_mapping(rgbs_gt.squeeze(0))).abs().mean(dim=-1)  # [N]
            per_pix = per_pix_log

        
        # Calculate importance weights
        # Calculate reconstruction loss
        recon_loss = per_pix.mean()
        
        # Add regularizer
        loss = recon_loss

        return loss

    def _zero_angle_aux_predictions(self, material_id):
        """Build a grazing-incidence augmentation in one of three modes.

        Modes (selected by ``cfg.model.grazing_mode``):
          * ``'zero_exact'`` / ``'near_zero_brdf'``  (pseudo-observations).
              wi.z = 0  (or c ~ U(cos_min, cos_max) for near_zero_brdf);
              target = 0 raw BRDF. ``pred`` is concatenated into the real
              batch so the existing reconstruction loss handles it (effective
              weight is controlled by ``grazing_ratio``).
          * ``'contribution_decay'``  (BRDF-shape regularizer — NOT an
              observation). Sample c_high ~ U(grazing_decay_cos_min,
              grazing_decay_cos_max), alpha ~ U(0,1), c_low = alpha * c_high.
              Decode q_high = brdf(wi_high, wo) * c_high and
              q_low = brdf(wi_low, wo) * c_low with the same material id, the
              same wo and the same azimuth. The loss is a one-sided log-ratio
              hinge that only penalizes q_low > alpha * q_high:
                  excess     = log(clamp(q_low,0)  + eps)
                             - log(clamp(target,0) + eps)
                  per_sample = relu(excess).mean(dim=-1)
                  decay_loss = grazing_decay_weight * per_sample.sum() / B
              with B the real batch size so the regularizer grows with
              ``grazing_ratio`` instead of being averaged away.

        Returns
        -------
        None when disabled.
        For 'zero_exact' / 'near_zero_brdf':
            ``{'kind': 'concat', 'pred': brdf [N,3], 'target': zeros [N,3]}``.
        For 'contribution_decay':
            ``{'kind': 'loss', 'loss': scalar tensor, 'diagnostics': {...}}``.

        Material ids are drawn uniformly from the **entire material set**;
        wo is uniform on the upper hemisphere.
        """
        n_total = material_id.shape[0]
        n_grazing = int(self.grazing_ratio * n_total)
        if n_grazing <= 0:
            return None

        device = material_id.device
        mids = torch.randint(0, self.material.num_materials, (n_grazing,), device=device)

        # Shared sampling: wo on upper hemisphere; azimuth phi_i for wi.
        z_o = torch.rand(n_grazing, device=device)
        phi_o = torch.rand(n_grazing, device=device) * (2 * math.pi)
        sin_t_o = torch.sqrt((1 - z_o ** 2).clamp(min=0))
        wo_local = torch.stack([sin_t_o * torch.cos(phi_o),
                                sin_t_o * torch.sin(phi_o), z_o], dim=-1)
        phi_i = torch.rand(n_grazing, device=device) * (2 * math.pi)
        cos_phi_i, sin_phi_i = torch.cos(phi_i), torch.sin(phi_i)

        mode = self.grazing_mode

        if mode in ('zero_exact', 'near_zero_brdf'):
            if mode == 'zero_exact':
                c = torch.zeros(n_grazing, device=device)
            else:
                c = (torch.rand(n_grazing, device=device)
                     * (self.grazing_cos_max - self.grazing_cos_min)
                     + self.grazing_cos_min)
            r = torch.sqrt((1 - c ** 2).clamp(min=0))
            wi_local = torch.stack([r * cos_phi_i, r * sin_phi_i, c], dim=-1)
            brdf = self.material.eval_brdf(wi_local, wo_local, mids)
            return {'kind': 'concat', 'pred': brdf, 'target': torch.zeros_like(brdf)}

        if mode == 'contribution_decay':
            c_high = (torch.rand(n_grazing, device=device)
                      * (self.grazing_decay_cos_max - self.grazing_decay_cos_min)
                      + self.grazing_decay_cos_min)
            alpha = torch.rand(n_grazing, device=device)
            c_low = alpha * c_high
            r_high = torch.sqrt((1 - c_high ** 2).clamp(min=0))
            r_low = torch.sqrt((1 - c_low ** 2).clamp(min=0))
            wi_high = torch.stack([r_high * cos_phi_i, r_high * sin_phi_i, c_high], dim=-1)
            wi_low = torch.stack([r_low * cos_phi_i, r_low * sin_phi_i, c_low], dim=-1)

            brdf_high = self.material.eval_brdf(wi_high, wo_local, mids)
            brdf_low = self.material.eval_brdf(wi_low, wo_local, mids)

            q_high = brdf_high * c_high.unsqueeze(-1)
            q_low = brdf_low * c_low.unsqueeze(-1)
            target = alpha.unsqueeze(-1) * q_high.detach()

            eps = self.grazing_decay_eps
            q_low_pos = q_low.clamp(min=0)
            target_pos = target.clamp(min=0)
            excess = torch.log(q_low_pos + eps) - torch.log(target_pos + eps)
            per_sample = NF.relu(excess).mean(dim=-1)        # [n_grazing]
            decay_loss = self.grazing_decay_weight * per_sample.sum() / n_total

            diagnostics = {
                'q_low_mean':  q_low.detach().mean(),
                'q_low_max':   q_low.detach().max(),
                'q_high_mean': q_high.detach().mean(),
                'q_high_max':  q_high.detach().max(),
                'excess_mean': excess.detach().mean(),
                'excess_max':  excess.detach().max(),
            }
            return {'kind': 'loss', 'loss': decay_loss, 'diagnostics': diagnostics}

        raise ValueError(f"Unknown grazing_mode: {mode!r}")

    def training_step(self, batch, batch_idx):

        """
        with importance sampling
        """
        # ------------------------------------------------------------------
        # 1. Un-pack inputs
        # ------------------------------------------------------------------
        rgbs_gt, wi,wo, material_id = batch['rgb'].squeeze(0), batch['wi'].squeeze(0), batch['wo'].squeeze(0), batch['material_id'].squeeze(0)
        #vis = torch.ones_like(rgbs_gt, dtype=torch.bool)
        #return self.loss_function(rgbs_gt, rgbs_gt, vis)
        #rgbs_gt, wi,wo, material_id = batch['rgb'].squeeze(0), batch['h_vec'].squeeze(0), batch['d_vec'].squeeze(0), batch['material_id'].squeeze(0)
        #print(f"material_id: {material_id}")
        #print(f"material_id: {material_id.shape}")
        #print(f"rgbs_gt: {rgbs_gt.shape}, wi: {wi.shape}, wo: {wo.shape}, material_id: {material_id.shape}")

        # forward renders
        rgbs = self.material.eval_brdf(wi, wo, material_id)

        # --- Grazing augmentation: 'concat' modes add synthetic rays to the
        # batch so they share the main loss; 'loss' mode (contribution_decay)
        # returns its own scalar that is added to the total loss separately. ---
        grazing_aug = self._zero_angle_aux_predictions(material_id)
        grazing_loss_log = torch.tensor(0.0, device=rgbs.device)
        grazing_extra_loss = torch.tensor(0.0, device=rgbs.device)
        grazing_diag = {}
        if grazing_aug is not None:
            if grazing_aug['kind'] == 'concat':
                g_pred = grazing_aug['pred']
                g_gt = grazing_aug['target']
                g_vis = torch.ones_like(g_pred, dtype=torch.bool)
                grazing_loss_log = self.loss_function(g_pred, g_gt, g_vis)  # monitoring only
                rgbs = torch.cat([rgbs, g_pred], dim=0)
                rgbs_gt = torch.cat([rgbs_gt, g_gt], dim=0)
            else:  # 'loss'
                grazing_extra_loss = grazing_aug['loss']
                grazing_loss_log = grazing_extra_loss.detach()
                grazing_diag = grazing_aug['diagnostics']

        vis = torch.ones_like(rgbs, dtype=torch.bool)
        loss = self.loss_function(rgbs, rgbs_gt, vis)
        total_loss = loss + grazing_extra_loss

        psnr_loss  = torch.nn.functional.mse_loss(rgbs[vis], rgbs_gt.squeeze(0)[vis], reduction='mean')
        if torch.isnan(psnr_loss).any():
            print("psnr_loss is nan")
        max_val = rgbs_gt.squeeze(0)[vis].max().clamp_min(1e-8)
        psnr       = 10.0 * torch.log10((max_val ** 2) / psnr_loss.clamp_min(1e-10))

        # ------------------------------------------------------------------
        # 7.  Logging  (now includes diagnostics)
        # ------------------------------------------------------------------
        log_dict = {
            'train/recon_loss':   loss,
            'train/total_loss':   total_loss,
            'train/grazing_loss': grazing_loss_log,
            'train/psnr':         psnr,
        }
        for k, v in grazing_diag.items():
            log_dict[f'train/grazing_{k}'] = v
        self.log_dict(log_dict, prog_bar=True, batch_size=rgbs.shape[0])

        return total_loss

    # def on_after_backward(self):
    #     """Called after loss.backward() and before optimizers step."""
    #     if self.global_step % 100 == 0:  # Print every 100 steps
    #         for name, param in self.named_parameters():
    #             if param.grad is not None:
    #                 grad_norm = param.grad.norm().item()
    #                 print(f"{name}: grad_norm={grad_norm:.6f}")
                
    def validation_step(self, batch, batch_idx):
        """
        with importance sampling
        """
        # ------------------------------------------------------------------
        # 1. Un-pack inputs
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # 1. Un-pack inputs
        # ------------------------------------------------------------------
        rgbs_gt, wi,wo, material_id = batch['rgb'].squeeze(0), batch['wi'].squeeze(0), batch['wo'].squeeze(0), batch['material_id'].squeeze(0)
        #rgbs_gt, wi,wo, material_id = batch['rgb'].squeeze(0), batch['h_vec'].squeeze(0), batch['d_vec'].squeeze(0), batch['material_id'].squeeze(0)
        # forward renders
        rgbs=self.material.eval_brdf(wi, wo, material_id)
        vis = torch.ones_like(rgbs, dtype=torch.bool)
        loss = self.loss_function(rgbs, rgbs_gt, vis)

        psnr_loss  = torch.nn.functional.mse_loss(rgbs[vis], rgbs_gt.squeeze(0)[vis], reduction='mean')
        if torch.isnan(psnr_loss).any():
            print("psnr_loss is nan")
        max_val = rgbs_gt.squeeze(0)[vis].max().clamp_min(1e-8)
        psnr       = 10.0 * torch.log10((max_val ** 2) / psnr_loss.clamp_min(1e-10))

        # ------------------------------------------------------------------
        # 7.  Logging  (now includes diagnostics)
        # ------------------------------------------------------------------
        self.log_dict({
            'val/loss':         loss,  # Add this line
            'val/recon_loss':   loss,
            'val/total_loss':   loss,
            'val/psnr':         psnr,
        }, prog_bar=True, batch_size=rgbs.shape[0])

        if batch_idx == 0:
            output_dir = os.path.join(self.cfg.exp_output_root_path, 'images')
            os.makedirs(output_dir, exist_ok=True)
            self.visualize_brdf_lobe(output_dir=output_dir, num_materials=10, resolution=64)

        return loss

    def visualize_brdf_lobe(self, output_dir, num_materials=10, resolution=64):
        """Polar BRDF lobe plots for a sample of materials.

        Two visualizations:
          (1) Fix wi, vary wo — single figure per material with theta_i overlays.
          (2) Fix wo, vary wi — grid of (wi_phi × wo_phi) figures per material so
              the zero-grazing constraint can be inspected at all azimuths
              (not just wi_phi=0 / wo_phi=0).
        """
        if not hasattr(self.material, 'eval_brdf'):
            return

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        device = next(self.material.parameters()).device
        torch.manual_seed(42)
        material_indices = torch.randint(
            0, self.material.num_materials, (num_materials,), device=device)

        local_normal = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        brdf_lobe_dir = os.path.join(output_dir, 'brdf_lobes')
        os.makedirs(brdf_lobe_dir, exist_ok=True)

        # ---- Visualization 1: Fix wi, vary wo --------------------------------
        theta_i_values = [15.0, 30.0, 45.0, 60.0, 75.0]
        for mat_idx in range(num_materials):
            mid = material_indices[mat_idx:mat_idx + 1]
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

            for theta_i_deg in theta_i_values:
                theta_i = np.radians(theta_i_deg)
                wi = torch.tensor([[np.sin(theta_i), 0.0, np.cos(theta_i)]],
                                  device=device, dtype=torch.float32)

                theta_o_range = np.linspace(-np.pi / 2, np.pi / 2, resolution * 2)
                wo_batch = torch.zeros(len(theta_o_range), 3, device=device)
                for j, theta_o in enumerate(theta_o_range):
                    if theta_o >= 0:
                        wo_batch[j, 0] = -np.sin(theta_o)
                        wo_batch[j, 2] =  np.cos(theta_o)
                    else:
                        wo_batch[j, 0] =  np.sin(-theta_o)
                        wo_batch[j, 2] =  np.cos(-theta_o)

                wi_batch = wi.expand(len(theta_o_range), -1)
                mid_batch = mid.expand(len(theta_o_range))
                with torch.no_grad():
                    brdf = self.material.eval_brdf(wi_batch, wo_batch, mid_batch)

                ax.plot(theta_o_range, brdf.mean(dim=-1).cpu().numpy(),
                        label=f'θ_i={theta_i_deg}°')
                ax.axvline(x=np.radians(theta_i_deg),
                           color=ax.lines[-1].get_color(), linestyle='--', alpha=0.5)

            ax.set_theta_zero_location('N')
            ax.set_theta_direction(1)
            ax.set_thetamin(-90)
            ax.set_thetamax(90)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.set_title(f'BRDF Polar Plot (vary wo) - Mat {int(mid.item())}\n'
                         f'(Fixed wi, vary wo; 0°=normal, dashed=specular direction)')
            plt.tight_layout()
            plt.savefig(os.path.join(brdf_lobe_dir,
                f'polar_brdf_vary_wo_mat_{int(mid.item()):04d}.png'), dpi=150)
            plt.close()

        # ---- Visualization 2: Fix wo, vary wi  (BRDF × cos_theta_i) ----------
        # Original (wi_phi=0, wo_phi=0) plus 4 azimuth variations per material.
        theta_o_values = [15.0, 30.0, 45.0, 60.0]
        phi_configs = [
            (0.0,   0.0),
            (90.0,  0.0),
            (180.0, 0.0),
            (0.0,   90.0),
            (90.0,  90.0),
        ]

        for mat_idx in range(num_materials):
            mid = material_indices[mat_idx:mat_idx + 1]
            for wi_phi_deg, wo_phi_deg in phi_configs:
                    wi_phi = np.radians(wi_phi_deg)
                    wo_phi = np.radians(wo_phi_deg)
                    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

                    for theta_o_deg in theta_o_values:
                        theta_o = np.radians(theta_o_deg)
                        wo = torch.tensor([[
                            np.sin(theta_o) * np.cos(wo_phi),
                            np.sin(theta_o) * np.sin(wo_phi),
                            np.cos(theta_o),
                        ]], device=device, dtype=torch.float32)

                        theta_i_range = np.linspace(-np.pi / 2 + 0.01,
                                                    np.pi / 2 - 0.01, resolution * 2)
                        ti = torch.tensor(theta_i_range, device=device, dtype=torch.float32)
                        wi_batch = torch.stack([
                            -torch.sin(ti) * float(np.cos(wi_phi)),
                            -torch.sin(ti) * float(np.sin(wi_phi)),
                            torch.cos(ti),
                        ], dim=-1)

                        wo_batch = wo.expand(len(theta_i_range), -1)
                        mid_batch = mid.expand(len(theta_i_range))
                        with torch.no_grad():
                            brdf = self.material.eval_brdf(wi_batch, wo_batch, mid_batch)

                        cos_theta_i = wi_batch[:, 2].clamp(min=0).cpu().numpy()
                        ax.plot(theta_i_range, brdf.mean(dim=-1).cpu().numpy() * cos_theta_i,
                                label=f'θ_o={theta_o_deg}°')
                        ax.axvline(x=np.radians(theta_o_deg),
                                   color=ax.lines[-1].get_color(), linestyle='--', alpha=0.5)

                    ax.set_theta_zero_location('N')
                    ax.set_theta_direction(1)
                    ax.set_thetamin(-90)
                    ax.set_thetamax(90)
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                    ax.set_title(
                        f'BRDF × cos(θ_i) - Mat {int(mid.item())}\n'
                        f'wi_φ={wi_phi_deg:.0f}°, wo_φ={wo_phi_deg:.0f}° '
                        f'(±90° = grazing)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(
                        brdf_lobe_dir,
                        f'polar_brdf_vary_wi_mat_{int(mid.item()):04d}'
                        f'_wiphi{int(wi_phi_deg):03d}_wophi{int(wo_phi_deg):03d}.png'), dpi=150)
                    plt.close()

        n_vary_wi = num_materials * len(phi_configs)
        print(f"[BRDF Lobe Visualization] Saved {num_materials + n_vary_wi} figures to {brdf_lobe_dir}")

    def on_train_batch_start(self, batch, batch_idx):
        pass
        #step = self.global_step
        #self.trainer.train_dataloader.dataset.datasets.set_step(step)
        
    # def validation_step(self, batch, batch_idx):
    #     """ Unified validation step for both normal and radiometric calibration """
    #     rays, rgbs_gt, emitter_ids, material_ids = batch['rays'], batch['rgbs'], batch['emitter_ids'], batch['material_ids']
    #     prior = 0.0
    #     if self.handeye_refiner:
    #         rays, prior = self.handeye_refiner.apply_handeye_delta_to_rays(rays, batch['camera_ids'])

    #     # forward renders
    #     rgbs, vis, ray_params, extra_output = self.renderer.render(self.emitter, rays, emitter_ids, material_ids, self.cfg.renderer.spp.train, None, None, validation=True)
    #     # Handle radiometric calibration
    #     if self.is_graypatch:
    #         pixel_all_ok, cosine_emitter_angle = extra_output
    #         self._radiometric_calibration_step(batch_idx, rgbs, rgbs_gt, vis, pixel_all_ok, cosine_emitter_angle)
    #     else:
    #         uv_offset = extra_output
    #     rgbs = rgbs * self.camera_factor
    #     psnr_loss = torch.nn.functional.mse_loss(rgbs[vis], rgbs_gt.squeeze(0)[vis], reduction='mean')
    #     max_val = torch.max(torch.stack([rgbs.max(), rgbs_gt.squeeze(0).max()])).clamp_min(1e-8)
    #     psnr = 10.0 * torch.log10((max_val ** 2) / psnr_loss.clamp_min(1e-10))
        
    #     loss = self.loss_function(rgbs, rgbs_gt, vis)
    #     emitter_radiance = self.emitter.light_radiance.detach().cpu().numpy()
        
    #     # Handle batch of images
    #     batch_size = 1
    #     batched_rgbs = rgbs.reshape(batch_size, *self.img_hw, -1)
    #     batched_rgbs_gt = rgbs_gt.reshape(batch_size, *self.img_hw, -1)
        
    #     # Reshape uv_offset for visualization if not graypatch
    #     if not self.is_graypatch and self.visualize_uv:
    #         batched_uv_offset = uv_offset.reshape(batch_size, *self.img_hw, 2)
        
    #     for b in range(batch_size):
    #         # Reshape individual sample in batch
    #         sample_rgbs = batched_rgbs[b]
    #         sample_rgbs_gt = batched_rgbs_gt[b]
            
    #         # Create output directory for each sample
    #         output_dir = os.path.join(
    #             self.cfg.exp_output_root_path,
    #             f'images'
    #         )
    #         os.makedirs(output_dir, exist_ok=True)

    #         # Convert float32 (0-65535) to uint16 (0-65535)
    #         sample_rgbs_gt_16bit = np.clip(sample_rgbs_gt.cpu().numpy(), 0, 65535).astype(np.uint16)
    #         sample_rgbs_16bit = np.clip(sample_rgbs.cpu().numpy(), 0, 65535).astype(np.uint16)

    #         # Save as 16-bit PNG (OpenCV expects BGR)
    #         cv2.imwrite(
    #             os.path.join(output_dir, f'gt_view_{batch_idx}_{b}.png'),
    #             cv2.cvtColor(sample_rgbs_gt_16bit, cv2.COLOR_RGB2BGR)
    #         )
    #         cv2.imwrite(
    #             os.path.join(output_dir, f'result_view_{batch_idx}_{b}.png'),
    #             cv2.cvtColor(sample_rgbs_16bit, cv2.COLOR_RGB2BGR)
    #         )
            
    #         # Visualize UV offsets as grayscale images
    #         if not self.is_graypatch and self.visualize_uv:
    #             sample_uv_offset = batched_uv_offset[b].cpu().numpy()  # Shape: [H, W, 2]
    #             u_offset = sample_uv_offset[:, :, 0]  # U offset
    #             v_offset = sample_uv_offset[:, :, 1]  # V offset
                
    #             # Normalize offsets to [0, 255] for visualization
    #             # Map the range of values to 0-255, with 127 representing zero offset
    #             u_min, u_max = u_offset.min(), u_offset.max()
    #             v_min, v_max = v_offset.min(), v_offset.max()
                
    #             # Normalize to [0, 255] with proper handling of positive and negative values
    #             u_range = max(abs(u_min), abs(u_max))
    #             v_range = max(abs(v_min), abs(v_max))
                
    #             if u_range > 0:
    #                 u_offset_vis = ((u_offset / u_range) * 127 + 127).astype(np.uint8)
    #             else:
    #                 u_offset_vis = np.full_like(u_offset, 127, dtype=np.uint8)
                    
    #             if v_range > 0:
    #                 v_offset_vis = ((v_offset / v_range) * 127 + 127).astype(np.uint8)
    #             else:
    #                 v_offset_vis = np.full_like(v_offset, 127, dtype=np.uint8)
                
    #             # Save grayscale offset images
    #             cv2.imwrite(
    #                 os.path.join(output_dir, f'u_offset_{batch_idx}_{b}.png'),
    #                 u_offset_vis
    #             )
    #             cv2.imwrite(
    #                 os.path.join(output_dir, f'v_offset_{batch_idx}_{b}.png'),
    #                 v_offset_vis
    #             ) 
                
    #             # Create colorful merged visualization using Red and Green channels
    #             # R=U offset, G=V offset, B=neutral (127)
    #             blue_channel = np.full_like(u_offset_vis, 127, dtype=np.uint8)
    #             uv_color = np.stack([u_offset_vis, v_offset_vis, blue_channel], axis=2)
    #             cv2.imwrite(
    #                 os.path.join(output_dir, f'uv_offset_color_{batch_idx}_{b}.png'),
    #                 cv2.cvtColor(uv_color, cv2.COLOR_RGB2BGR)
    #             )          
    #         # )
            
    #     os.makedirs(os.path.join(self.cfg.exp_output_root_path, f'pbr_map_images'), exist_ok=True)
    #     self.save_pbr_texture(os.path.join(self.cfg.exp_output_root_path, f'pbr_map_images'), batch_idx, b)
            
    #     self.log('val/loss', loss)
    #     self.log('val/emitter_radiance', emitter_radiance.mean())
    #     self.log('val/psnr', psnr)        
    #     return

    # def _radiometric_calibration_step(self, batch_idx, rgbs, rgbs_gt, vis, pixel_all_ok, cosine_emitter_angle):
    #     """Radiometric calibration: collect radiance-camera pairs with pixel coordinates."""
    #     mask = rgbs > 0
    #     rgbs_gt.squeeze(0)[~mask] = 0
    #     non_zero = (rgbs[vis].sum(dim=-1) > 0) & pixel_all_ok[vis]
    #     rad = rgbs[vis][non_zero].mean(dim=-1).detach().cpu().numpy()
    #     cam = rgbs_gt.squeeze(0)[vis][non_zero].mean(dim=-1).detach().cpu().numpy()
    #     cosine_emitter_angle = cosine_emitter_angle[vis][non_zero].detach().cpu().numpy()
        
    #     # Store pixel indices (flat indices in the image)
    #     # vis is already a mask for valid pixels, non_zero further filters within those
    #     vis_indices = torch.where(vis)[0]  # Get indices where vis is True
    #     non_zero_cpu = non_zero.cpu()
    #     pixel_indices = vis_indices[non_zero_cpu].cpu().numpy()  # Map to original flat indices
        
    #     average_radiance = cam.mean()
    #     self.average_radiance.append(average_radiance)
    #     print(f"Average radiance: {average_radiance}")
    #     self.radiance_rgb_pairs[batch_idx] = {
    #         'radiance': rad, 
    #         'camera': cam, 
    #         'cosine_emitter_angle': cosine_emitter_angle,
    #         'batch_idx': batch_idx,
    #         'pixel_indices': pixel_indices  # Store flat pixel indices
    #     }
    
    # def fit_radiance_camera_linear_model(self, all_rad, all_cam):
    #     """
    #     Fit linear model cam = a * rad (passing through origin) using RANSAC to remove outliers.
        
    #     Args:
    #         all_rad (np.ndarray): Radiance values
    #         all_cam (np.ndarray): Camera RGB values
            
    #     Returns:
    #         dict: {'a', 'r2', 'inlier_mask', 'outlier_mask', 'mean_error', 'median_error'}
    #     """
    #     from sklearn.linear_model import RANSACRegressor, LinearRegression
    #     from sklearn.metrics import r2_score
        
    #     X = all_rad.reshape(-1, 1)
    #     # Force the model to pass through the origin by setting fit_intercept=False
    #     ransac = RANSACRegressor(
    #         estimator=LinearRegression(fit_intercept=False),
    #         random_state=42, 
    #         residual_threshold=None
    #     )
    #     ransac.fit(X, all_cam)
        
    #     a = ransac.estimator_.coef_[0]
    #     inlier_mask = ransac.inlier_mask_
        
    #     # Calculate R² score on inliers
    #     y_pred_inliers = ransac.predict(X[inlier_mask])
    #     r2 = r2_score(all_cam[inlier_mask], y_pred_inliers)
        
    #     # Calculate fitting errors (residuals) for inliers
    #     residuals = np.abs(all_cam[inlier_mask] - y_pred_inliers)
    #     mean_error = np.mean(residuals)
    #     median_error = np.median(residuals)
        
    #     return {
    #         'a': a, 
    #         'r2': r2, 
    #         'inlier_mask': inlier_mask, 
    #         'outlier_mask': ~inlier_mask,
    #         'mean_error': mean_error,
    #         'median_error': median_error
    #     }


    # def debug_multi_line_fitting_recursive_ransac(self, all_rad, all_cam, all_batch_ids, epoch, max_lines=5, min_points=100):
    #     """
    #     DEBUG FUNCTION: Fit multiple lines using recursive RANSAC.
    #     Each iteration finds the best line, removes inliers, and repeats.
        
    #     Args:
    #         all_rad: Array of radiance values
    #         all_cam: Array of camera RGB values
    #         all_batch_ids: Array of batch indices for each point
    #         epoch: Current epoch number
    #         max_lines: Maximum number of lines to fit
    #         min_points: Minimum points required to fit a line
            
    #     Returns:
    #         list: List of dictionaries containing line fitting results
    #     """
    #     from sklearn.linear_model import RANSACRegressor, LinearRegression
    #     from sklearn.metrics import r2_score
        
    #     remaining_mask = np.ones(len(all_rad), dtype=bool)
    #     line_results = []
        
    #     print(f"\n{'='*70}")
    #     print(f"DEBUG: Multi-Line Fitting Analysis (Epoch {epoch})")
    #     print(f"{'='*70}")
        
    #     # Recursively fit lines
    #     for line_idx in range(max_lines):
    #         if remaining_mask.sum() < min_points:
    #             print(f"Stopping: Only {remaining_mask.sum()} points remaining (< {min_points})")
    #             break
                
    #         X = all_rad[remaining_mask].reshape(-1, 1)
    #         y = all_cam[remaining_mask]
            
    #         # Adaptive min_samples based on remaining points
    #         n_remaining = remaining_mask.sum()
    #         adaptive_min_samples = min(50, max(10, n_remaining // 20))
            
    #         try:
    #             ransac = RANSACRegressor(
    #                 estimator=LinearRegression(fit_intercept=False),
    #                 random_state=42 + line_idx,
    #                 residual_threshold=None,
    #                 min_samples=adaptive_min_samples,
    #                 max_trials=1000
    #             )
    #             ransac.fit(X, y)
                
    #             slope = ransac.estimator_.coef_[0]
    #             local_inlier_mask = ransac.inlier_mask_
                
    #             # Check if we found enough inliers
    #             n_inliers = local_inlier_mask.sum()
    #             if n_inliers < min_points:
    #                 print(f"Line {line_idx + 1}: Only {n_inliers} inliers found (< {min_points}), stopping")
    #                 break
                
    #         except ValueError as e:
    #             print(f"RANSAC failed at iteration {line_idx + 1}: {e}")
    #             print(f"Remaining points: {n_remaining}, stopping line fitting")
    #             break
            
    #         # Map local inliers back to global indices
    #         global_indices = np.where(remaining_mask)[0]
    #         global_inliers = global_indices[local_inlier_mask]
            
    #         # Calculate metrics
    #         y_pred = slope * all_rad[global_inliers]
    #         r2 = r2_score(all_cam[global_inliers], y_pred)
    #         residuals = np.abs(all_cam[global_inliers] - y_pred)
            
    #         line_results.append({
    #             'slope': slope,
    #             'r2': r2,
    #             'indices': global_inliers,
    #             'batch_ids': all_batch_ids[global_inliers],
    #             'n_points': len(global_inliers),
    #             'mean_error': np.mean(residuals),
    #             'median_error': np.median(residuals),
    #             'radiance': all_rad[global_inliers],
    #             'camera': all_cam[global_inliers]
    #         })
            
    #         # Remove these inliers from remaining points
    #         remaining_mask[global_inliers] = False
            
    #         print(f"Line {line_idx + 1}: slope={slope:.6f}, R²={r2:.6f}, points={len(global_inliers)}, "
    #               f"mean_err={np.mean(residuals):.4f}, median_err={np.median(residuals):.4f}")
        
    #     # Add remaining outliers as separate group
    #     if remaining_mask.sum() > 0:
    #         line_results.append({
    #             'slope': None,
    #             'r2': None,
    #             'indices': np.where(remaining_mask)[0],
    #             'batch_ids': all_batch_ids[remaining_mask],
    #             'n_points': remaining_mask.sum(),
    #             'mean_error': None,
    #             'median_error': None,
    #             'radiance': all_rad[remaining_mask],
    #             'camera': all_cam[remaining_mask]
    #         })
    #         print(f"Outliers: {remaining_mask.sum()} points (not fitted)")
        
    #     print(f"{'='*70}\n")
        
    #     # Visualize with color coding
    #     self._plot_multi_line_results(all_rad, all_cam, line_results, epoch)
        
    #     # Generate ground truth images with color overlays
    #     self._generate_colored_gt_images(line_results, epoch, self.radiance_rgb_pairs)
        
    #     return line_results
    
    # def _plot_multi_line_results(self, all_rad, all_cam, line_results, epoch):
    #     """
    #     DEBUG FUNCTION: Plot scatter with each line group in different color (red to green by slope).
    #     Uses different axis ranges for better visualization like the original fitting figure.
    #     """
    #     import matplotlib
    #     matplotlib.use('Agg')
    #     import matplotlib.pyplot as plt
    #     import matplotlib.cm as cm
        
    #     fig, ax = plt.subplots(figsize=(14, 10))
        
    #     # Sort lines by slope for color mapping (red=low slope, green=high slope)
    #     valid_lines = [l for l in line_results if l['slope'] is not None]
    #     valid_lines.sort(key=lambda x: x['slope'])
        
    #     # Color map from red to green
    #     n_lines = len(valid_lines)
    #     if n_lines > 0:
    #         colors = cm.RdYlGn(np.linspace(0, 1, n_lines))
    #     else:
    #         colors = []
        
    #     # Plot each line group
    #     for i, line_info in enumerate(valid_lines):
    #         rad = line_info['radiance']
    #         cam = line_info['camera']
            
    #         ax.scatter(rad, cam, alpha=0.6, s=2, color=colors[i],
    #                   label=f"Line {i+1}: slope={line_info['slope']:.6f}, R²={line_info['r2']:.4f}, n={line_info['n_points']}")
            
    #         # Plot fitted line from 0 to max radiance
    #         rad_range = np.linspace(0, rad.max(), 100)
    #         ax.plot(rad_range, line_info['slope'] * rad_range, 
    #                color=colors[i], linewidth=2.5, linestyle='--', alpha=0.8)
        
    #     # Plot outliers in gray
    #     outlier_info = [l for l in line_results if l['slope'] is None]
    #     if outlier_info:
    #         rad = outlier_info[0]['radiance']
    #         cam = outlier_info[0]['camera']
    #         ax.scatter(rad, cam, alpha=0.3, s=1, color='gray', 
    #                   label=f"Outliers: n={len(rad)}")
        
    #     # Plot ideal y=x line - use separate max for x and y axes
    #     max_rad = all_rad.max()
    #     max_cam = all_cam.max()
    #     max_val = max(max_rad, max_cam)
    #     ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, alpha=0.5, label='y=x (ideal)')
        
    #     ax.set_xlabel('Predicted Radiance', fontsize=14)
    #     ax.set_ylabel('Camera RGB', fontsize=14)
    #     ax.set_title(f'Multi-Line Fitting Analysis (Epoch {epoch})', fontsize=16, fontweight='bold')
    #     ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    #     ax.grid(True, alpha=0.3)
        
    #     # Use different axis ranges like the original plot
    #     ax.set_xlim([0, max_rad])
    #     ax.set_ylim([0, max_cam])
        
    #     save_path = os.path.join(self.cfg.exp_output_root_path, f'debug_multiline_epoch_{epoch}.png')
    #     plt.savefig(save_path, dpi=200, bbox_inches='tight')
    #     plt.close(fig)
    #     print(f"Multi-line plot saved to: {save_path}")
    
    # def _generate_colored_gt_images(self, line_results, epoch, radiance_rgb_pairs):
    #     """
    #     DEBUG FUNCTION: Generate ground truth images with colored overlays showing which line each pixel belongs to.
    #     Colors range from red (low slope) to green (high slope).
    #     Only colors pixels from non-main lines (skips the line with most points).
    #     """
    #     import matplotlib.pyplot as plt
    #     import matplotlib.cm as cm
        
    #     valid_lines = [l for l in line_results if l['slope'] is not None]
    #     valid_lines.sort(key=lambda x: x['slope'])
    #     n_lines = len(valid_lines)
        
    #     if n_lines == 0:
    #         print("No valid lines to visualize")
    #         return
        
    #     # Find the main line (line with most points) - we'll skip this in distribution
    #     main_line_idx = max(range(len(valid_lines)), key=lambda i: valid_lines[i]['n_points'])
    #     print(f"Main line (Line {main_line_idx + 1}) has {valid_lines[main_line_idx]['n_points']} points - will be shown as white/uncolored in images")
        
    #     # Create color map (brighter, more saturated colors)
    #     colors_rgb = cm.RdYlGn(np.linspace(0, 1, n_lines))[:, :3]  # RGB only, no alpha
        
    #     # For each unique image (batch_id), create colored overlay
    #     all_batch_ids = set()
    #     for line_info in line_results:
    #         all_batch_ids.update(line_info['batch_ids'])
        
    #     output_dir = os.path.join(self.cfg.exp_output_root_path, 'debug_colored_images')
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     # Build a global index to (batch_id, local_pixel_idx) mapping
    #     # This maps the global concatenated array index to the original image pixel
    #     global_idx_to_pixel = {}
    #     current_global_idx = 0
        
    #     for batch_id in sorted(radiance_rgb_pairs.keys()):
    #         pairs = radiance_rgb_pairs[batch_id]
    #         pixel_indices = pairs['pixel_indices']
    #         n_pixels = len(pixel_indices)
            
    #         for local_idx in range(n_pixels):
    #             global_idx_to_pixel[current_global_idx] = (batch_id, pixel_indices[local_idx])
    #             current_global_idx += 1
        
    #     # Create pixel-to-line mapping for each image
    #     for batch_id in sorted(all_batch_ids):
    #         # Load the ground truth image
    #         gt_path = os.path.join(self.cfg.exp_output_root_path, 'images', f'gt_view_{int(batch_id)}_0.png')
    #         if not os.path.exists(gt_path):
    #             print(f"GT image not found: {gt_path}")
    #             continue
            
    #         # Load as 16-bit
    #         gt_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    #         if gt_img is None:
    #             print(f"Failed to load image: {gt_path}")
    #             continue
            
    #         gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    #         H, W = gt_img.shape[:2]
            
    #         # Normalize to 0-1 for display
    #         if gt_img.dtype == np.uint16:
    #             gt_img_normalized = gt_img.astype(np.float32) / 65535.0
    #         else:
    #             gt_img_normalized = gt_img.astype(np.float32) / 255.0
            
    #         # Create a color overlay (start with grayscale GT image)
    #         colored_img = gt_img_normalized.copy()
            
    #         # Get the pixel indices for this batch from the original data collection
    #         if batch_id not in radiance_rgb_pairs:
    #             print(f"No data for batch {batch_id}")
    #             continue
            
    #         # Color pixels belonging to each non-main line
    #         total_colored = 0
    #         for line_idx, line_info in enumerate(valid_lines):
    #             # Skip the main line
    #             if line_idx == main_line_idx:
    #                 continue
                
    #             # Find which global indices belong to this batch and this line
    #             global_indices_for_line = line_info['indices']
                
    #             for global_idx in global_indices_for_line:
    #                 if global_idx in global_idx_to_pixel:
    #                     bid, flat_pixel_idx = global_idx_to_pixel[global_idx]
                        
    #                     if bid == batch_id:
    #                         # Convert flat index to (row, col)
    #                         row = flat_pixel_idx // W
    #                         col = flat_pixel_idx % W
                            
    #                         if 0 <= row < H and 0 <= col < W:
    #                             # Color this pixel with the line's color
    #                             colored_img[row, col] = colors_rgb[line_idx]
    #                             total_colored += 1
            
    #         # Save the colored image
    #         if total_colored > 0:
    #             # Convert back to uint8 for saving
    #             colored_img_uint8 = (colored_img * 255).astype(np.uint8)
                
    #             save_path = os.path.join(output_dir, f'colored_gt_view_{int(batch_id)}_epoch_{epoch}.png')
    #             cv2.imwrite(save_path, cv2.cvtColor(colored_img_uint8, cv2.COLOR_RGB2BGR))
    #             print(f"Saved colored GT for image {int(batch_id)}: {total_colored} pixels colored, saved to {save_path}")
    #         else:
    #             print(f"Image {int(batch_id)}: No non-main-line pixels to color")
        
    #     # Create summary statistics file
    #     summary_path = os.path.join(output_dir, f'line_summary_epoch_{epoch}.txt')
    #     with open(summary_path, 'w') as f:
    #         f.write(f"Multi-Line Fitting Summary (Epoch {epoch})\n")
    #         f.write(f"{'='*70}\n")
    #         f.write(f"Main Line: Line {main_line_idx + 1} with {valid_lines[main_line_idx]['n_points']} points (slope={valid_lines[main_line_idx]['slope']:.6f})\n")
    #         f.write(f"{'='*70}\n\n")
            
    #         for batch_id in sorted(all_batch_ids):
    #             f.write(f"\nImage {int(batch_id)}:\n")
    #             f.write(f"{'-'*50}\n")
                
    #             total_pixels = 0
    #             line_pixel_counts = []
                
    #             for line_idx, line_info in enumerate(valid_lines):
    #                 mask = line_info['batch_ids'] == batch_id
    #                 pixel_count = mask.sum()
    #                 total_pixels += pixel_count
                    
    #                 if pixel_count > 0:
    #                     color_hex = '#%02x%02x%02x' % tuple((colors_rgb[line_idx] * 255).astype(int))
    #                     is_main = " (MAIN)" if line_idx == main_line_idx else ""
    #                     line_pixel_counts.append((line_idx + 1, pixel_count, line_info['slope'], color_hex, is_main))
                
    #             # Check outliers
    #             outlier_info = [l for l in line_results if l['slope'] is None]
    #             if outlier_info:
    #                 mask = outlier_info[0]['batch_ids'] == batch_id
    #                 outlier_count = mask.sum()
    #                 total_pixels += outlier_count
    #             else:
    #                 outlier_count = 0
                
    #             if total_pixels == 0:
    #                 f.write("  No pixels analyzed\n")
    #                 continue
                
    #             # Sort by pixel count descending
    #             line_pixel_counts.sort(key=lambda x: x[1], reverse=True)
                
    #             for line_num, count, slope, color, is_main in line_pixel_counts:
    #                 percentage = 100.0 * count / total_pixels
    #                 f.write(f"  Line {line_num}{is_main} (slope={slope:.6f}, {color}): {count} pixels ({percentage:.2f}%)\n")
                
    #             if outlier_count > 0:
    #                 percentage = 100.0 * outlier_count / total_pixels
    #                 f.write(f"  Outliers: {outlier_count} pixels ({percentage:.2f}%)\n")
                
    #             f.write(f"  Total: {total_pixels} pixels\n")
        
    #     print(f"Line summary saved to: {summary_path}")
        
    #     # Create distribution plot - EXCLUDE main line to better visualize minority lines
    #     fig, ax = plt.subplots(figsize=(12, 8))
        
    #     batch_ids_sorted = sorted(all_batch_ids)
    #     x_pos = np.arange(len(batch_ids_sorted))
        
    #     # Prepare data for stacked bar chart (excluding main line)
    #     secondary_lines = [i for i in range(n_lines) if i != main_line_idx]
    #     line_distributions = np.zeros((len(secondary_lines), len(batch_ids_sorted)))
        
    #     for i, batch_id in enumerate(batch_ids_sorted):
    #         for plot_idx, line_idx in enumerate(secondary_lines):
    #             line_info = valid_lines[line_idx]
    #             mask = line_info['batch_ids'] == batch_id
    #             line_distributions[plot_idx, i] = mask.sum()
        
    #     # Create stacked bar chart
    #     bottom = np.zeros(len(batch_ids_sorted))
    #     for plot_idx, line_idx in enumerate(secondary_lines):
    #         ax.bar(x_pos, line_distributions[plot_idx], bottom=bottom, 
    #                color=colors_rgb[line_idx], alpha=0.8,
    #                label=f"Line {line_idx+1} (slope={valid_lines[line_idx]['slope']:.6f})")
    #         bottom += line_distributions[plot_idx]
        
    #     ax.set_xlabel('Image ID', fontsize=12)
    #     ax.set_ylabel('Number of Pixels', fontsize=12)
    #     ax.set_title(f'Non-Main Line Distribution Across Images (Epoch {epoch})\nMain line (Line {main_line_idx+1}, {valid_lines[main_line_idx]["n_points"]} pts) excluded', 
    #                  fontsize=14, fontweight='bold')
    #     ax.set_xticks(x_pos)
    #     ax.set_xticklabels([int(bid) for bid in batch_ids_sorted], rotation=45, ha='right')
    #     ax.legend(loc='upper right', fontsize=9)
    #     ax.grid(True, alpha=0.3, axis='y')
        
    #     dist_plot_path = os.path.join(output_dir, f'line_distribution_epoch_{epoch}.png')
    #     plt.savefig(dist_plot_path, dpi=150, bbox_inches='tight')
    #     plt.close(fig)
    #     print(f"Line distribution plot (excluding main line) saved to: {dist_plot_path}")

    # def on_validation_epoch_end(self):
    #     """Plot accumulated radiance-RGB pairs at the end of validation."""
    #     if not self.is_graypatch or not self.radiance_rgb_pairs:
    #         return
        
    #     import matplotlib
    #     matplotlib.use('Agg')
    #     import matplotlib.pyplot as plt
    #     from matplotlib import cm
        
    #     # Concatenate all data and track batch indices
    #     all_rad = []
    #     all_cam = []
    #     all_angle = []
    #     all_batch_ids = []
    #     for batch_idx, pairs in sorted(self.radiance_rgb_pairs.items()):
    #         rad, cam, cosine_emitter_angle = pairs['radiance'], pairs['camera'], pairs['cosine_emitter_angle']
    #         all_angle.append(cosine_emitter_angle)
    #         all_rad.append(rad)
    #         all_cam.append(cam)
    #         # Create array of batch_idx repeated for each point
    #         all_batch_ids.append(np.full(len(rad), batch_idx))
        
    #     all_rad = np.concatenate(all_rad)
    #     all_cam = np.concatenate(all_cam)
    #     all_angle = np.concatenate(all_angle)
    #     all_batch_ids = np.concatenate(all_batch_ids)
        
    #     # If emitter calibration mode, only plot angle vs cam/rad ratio
    #     if self.emitter_calibration:
    #         # Convert cosine to degrees
    #         angle_degrees = np.arccos(np.clip(all_angle, -1.0, 1.0)) * 180.0 / np.pi
            
    #         # Compute cam/rad ratio
    #         cam_rad_ratio = all_cam / (all_rad + 1e-8)  # Add epsilon to avoid division by zero
            
    #         # Create plot
    #         fig = plt.figure(figsize=(12, 8))
    #         plt.scatter(angle_degrees, cam_rad_ratio, alpha=0.5, s=1, color='blue')
    #         plt.xlim([0, 90])
    #         plt.ylim([0, cam_rad_ratio.max() * 1.1])  # Start from 0, extend 10% above max
    #         plt.xlabel('Emitter Angle (degrees)', fontsize=12)
    #         plt.ylabel('Camera / Radiance Ratio', fontsize=12)
    #         plt.title(f'Emitter Angle vs Cam/Rad Ratio (Epoch {self.current_epoch})', fontsize=14)
    #         plt.grid(True, alpha=0.3)
            
    #         save_path = os.path.join(self.cfg.exp_output_root_path, f'emitter_calibration_epoch_{self.current_epoch}.png')
    #         plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #         plt.close(fig)
            
    #         print(f"\n{'='*60}")
    #         print(f"Emitter Calibration Plot saved to: {save_path}")
    #         print(f"{'='*60}\n")
            
    #         # Compute relative radiance table with 1 degree resolution
    #         import json
    #         from scipy.spatial import cKDTree
            
    #         # Normalize cam_rad_ratio to max value
    #         max_ratio = cam_rad_ratio.max()
    #         relative_ratio = cam_rad_ratio / max_ratio
            
    #         # Create 1-degree bins (0-1, 1-2, ..., 89-90)
    #         resolution = 1.0
    #         angle_bins = np.arange(0, 90, resolution)
            
    #         # Compute average ratio for each bin
    #         bin_centers = []
    #         bin_ratios = []
    #         bins_with_data = []
            
    #         for i, bin_start in enumerate(angle_bins):
    #             bin_end = bin_start + resolution
    #             # Find all points within this bin
    #             mask = (angle_degrees >= bin_start) & (angle_degrees < bin_end)
                
    #             if np.any(mask):
    #                 # Average the relative ratios in this bin
    #                 avg_ratio = np.mean(relative_ratio[mask])
    #                 bin_centers.append(bin_start + resolution / 2)  # Use bin center
    #                 bin_ratios.append(avg_ratio)
    #                 bins_with_data.append(i)
    #             else:
    #                 # No data in this bin, will use nearest neighbor later
    #                 bin_centers.append(bin_start + resolution / 2)
    #                 bin_ratios.append(None)
            
    #         # Fill missing bins using nearest neighbor from bins with data
    #         if bins_with_data:
    #             # Build KD-Tree from bin centers that have data
    #             valid_centers = np.array([bin_centers[i] for i in bins_with_data]).reshape(-1, 1)
    #             valid_ratios = np.array([bin_ratios[i] for i in bins_with_data])
    #             tree = cKDTree(valid_centers)
                
    #             # Fill in missing bins
    #             for i in range(len(bin_ratios)):
    #                 if bin_ratios[i] is None:
    #                     # Find nearest bin with data
    #                     dist, idx = tree.query([[bin_centers[i]]], k=1)
    #                     bin_ratios[i] = valid_ratios[idx[0]]
            
    #         # Create the table as a dictionary
    #         calibration_table = {
    #             "resolution_degrees": resolution,
    #             "angle_range": [0.0, 90.0],
    #             "max_cam_rad_ratio": float(max_ratio),
    #             "epoch": self.current_epoch,
    #             "data": {
    #                 f"{center:.1f}": float(ratio)
    #                 for center, ratio in zip(bin_centers, bin_ratios)
    #             }
    #         }
            
    #         # Save to JSON file
    #         json_path = os.path.join(self.cfg.exp_output_root_path, f'emitter_calibration_table_epoch_{self.current_epoch}.json')
    #         with open(json_path, 'w') as f:
    #             json.dump(calibration_table, f, indent=2)
            
    #         print(f"Emitter Calibration Table saved to: {json_path}")
    #         print(f"Table contains {len(calibration_table['data'])} entries from 0° to 90° with 0.1° resolution")
    #         print(f"Max cam/rad ratio: {max_ratio:.6f}")
    #         print(f"{'='*60}\n")
            
    #         self.average_radiance.clear()
    #         self.radiance_rgb_pairs.clear()
    #         return
        
    #     # === DEBUG: Multi-line fitting analysis ===
    #     # UNCOMMENT TO ENABLE DEBUG MODE:
    #     # self.debug_multi_line_fitting_recursive_ransac(
    #     #     all_rad, all_cam, all_batch_ids, self.current_epoch,
    #     #     max_lines=5, min_points=100
    #     # )
        
    #     # Fit linear model using RANSAC
    #     fit_res = self.fit_radiance_camera_linear_model(all_rad, all_cam)
    #     a, r2 = fit_res['a'], fit_res['r2']
    #     inlier_mask, outlier_mask = fit_res['inlier_mask'], fit_res['outlier_mask']
    #     mean_error, median_error = fit_res['mean_error'], fit_res['median_error']
        
    #     # Find top 50 outliers where ground truth (cam) >> prediction (rad * a)
    #     predicted = all_rad * a
    #     residuals = all_cam - predicted  # Positive means cam > prediction
    #     outlier_indices = np.argsort(residuals)[-50:][::-1]  # Top 50 largest residuals
        
    #     # Print and save fitting results
    #     print(f"\n{'='*60}")
    #     print(f"Linear Fitting Results (Epoch {self.current_epoch}):")
    #     print(f"  Equation: cam = {a:.6f} * rad")
    #     print(f"  R² score (inliers): {r2:.6f}")
    #     print(f"  Mean absolute error: {mean_error:.6f}")
    #     print(f"  Median absolute error: {median_error:.6f}")
    #     print(f"  Inliers: {inlier_mask.sum()} / {len(inlier_mask)} ({100*inlier_mask.sum()/len(inlier_mask):.2f}%)")
    #     print(f"  Outliers: {outlier_mask.sum()}")
    #     print(f"{'='*60}\n")
        
    #     print(f"Top 50 Outliers (Ground Truth >> Prediction):")
    #     print(f"{'Rank':<6} {'Image ID':<10} {'GT Radiance':<15} {'Predicted':<15} {'Error':<15}")
    #     print(f"{'-'*65}")
    #     for rank, idx in enumerate(outlier_indices, 1):
    #         img_id = all_batch_ids[idx]
    #         gt_val = all_cam[idx]
    #         pred_val = predicted[idx]
    #         error = residuals[idx]
    #         print(f"{rank:<6} {int(img_id):<10} {gt_val:<15.4f} {pred_val:<15.4f} {error:<15.4f}")
    #     print(f"{'-'*65}\n")
        
    #     results_path = os.path.join(self.cfg.exp_output_root_path, f'fitting_results_epoch_{self.current_epoch}.txt')
    #     with open(results_path, 'w') as f:
    #         f.write(f"Linear Fitting Results (Epoch {self.current_epoch}):\n")
    #         f.write(f"Equation: cam = {a:.6f} * rad\n")
    #         f.write(f"R² score (inliers): {r2:.6f}\n")
    #         f.write(f"Mean absolute error: {mean_error:.6f}\n")
    #         f.write(f"Median absolute error: {median_error:.6f}\n")
    #         f.write(f"Inliers: {inlier_mask.sum()} / {len(inlier_mask)} ({100*inlier_mask.sum()/len(inlier_mask):.2f}%)\n")
    #         f.write(f"Outliers: {outlier_mask.sum()}\n")
    #         f.write(f"\nTop 10 Outliers (Ground Truth >> Prediction):\n")
    #         f.write(f"{'Rank':<6} {'Image ID':<10} {'GT Radiance':<15} {'Predicted':<15} {'Error':<15}\n")
    #         f.write(f"{'-'*65}\n")
    #         for rank, idx in enumerate(outlier_indices, 1):
    #             img_id = all_batch_ids[idx]
    #             gt_val = all_cam[idx]
    #             pred_val = predicted[idx]
    #             error = residuals[idx]
    #             f.write(f"{rank:<6} {int(img_id):<10} {gt_val:<15.4f} {pred_val:<15.4f} {error:<15.4f}\n")
        
    #     # Create plot
    #     fig = plt.figure(figsize=(12, 8))
        
    #     # Plot data points
    #     for idx, (batch_idx, pairs) in enumerate(sorted(self.radiance_rgb_pairs.items())):
    #         rad, cam = pairs['radiance'], pairs['camera']
    #         plt.scatter(rad, cam, alpha=0.5, s=1, color='blue')
        
    #     # Plot fitted line and reference
    #     max_rad, max_cam = all_rad.max(), all_cam.max()
    #     rad_range = np.linspace(0, max_rad, 100)
    #     plt.plot(rad_range, a * rad_range, 'g-', linewidth=2, 
    #             label=f'Fitted: cam = {a:.4f}*rad\nR² = {r2:.4f}')
    #     plt.plot([0, max(max_rad, max_cam)], [0, max(max_rad, max_cam)], 
    #             'r--', linewidth=2, label='y=x (ideal)')
        
    #     max_rad, max_cam = all_rad.max(), all_cam.max()
    #     plt.xlim([0, max_rad])
    #     plt.ylim([0, max_cam])
    #     plt.xlabel('Predicted Radiance', fontsize=12)
    #     plt.ylabel('Camera RGB', fontsize=12)
    #     plt.title(f'Radiance vs Camera RGB (Epoch {self.current_epoch})', fontsize=14)
    #     plt.legend(loc='upper left', fontsize=10)
    #     plt.grid(True, alpha=0.3)
        
    #     save_path = os.path.join(self.cfg.exp_output_root_path, f'radiance_rgb_epoch_{self.current_epoch}.png')
    #     plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #     plt.close(fig)
        
    #     # Plot average radiance w.r.t index
    #     plt.figure()
    #     plt.plot(self.average_radiance)
    #     plt.savefig(os.path.join(self.cfg.exp_output_root_path, f'average_radiance.png'))
    #     plt.close()
        
    #     self.average_radiance.clear()
    #     self.radiance_rgb_pairs.clear()