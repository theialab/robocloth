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

class Stage1Trainer(pl.LightningModule):
    """Stage 1 — learn the material prior (paper Sec. 4.2).

    Jointly optimizes the shared BRDF decoder (lr=decoder_lr) and the
    per-point latent/normal/tangent bank (lr=lr, 8-bit Adam) across all
    training materials, with the log-relative reconstruction loss and the
    grazing-angle decay regularizer. Validation renders held-out
    observations and logs val/psnr; checkpoints keep the full bank —
    stage 2 later loads the decoder weights only.
    """
    def __init__(self, cfg, material, gt_material, roughness, metallic):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.automatic_optimization = False

        self.material = material
        self.freeze_decoder = cfg.model.freeze_decoder
        self.more_visualizations = True
        self._opt_name = getattr(cfg.model.optimizer, 'name', 'Adam')
        self.reset_latent_momentum = getattr(cfg.model.optimizer, 'reset_latent_momentum_on_chunk_switch', False)
        self.gt_material = gt_material
        self.gt_folder = cfg.gt_folder
        self.camera_factor1 = cfg.renderer.camera.linear_factor1
        self.camera_factor2 = cfg.renderer.camera.linear_factor2
        # Materials >= 352 captured at exposure 8000 instead of 20000; camera response is linear in exposure.
        self.camera_factor3 = self.camera_factor2 * (8000.0 / 20000.0)

        # Per-material camera-factor lookup table. The legacy hardcoded segments
        # (id<100→f1, id<352→f2, else→f3) are wrong for IDs 501-530 (which used
        # exposure 20000, not 8000) and don't reflect the bad↔backup material
        # swap. We load camera_factor.json from the dataset folder, expand to a
        # per-id float tensor, and index with material_ids in training_step.101
        cf_json_path = os.path.join(cfg.dataset_folder, 'camera_factor.json')
        with open(cf_json_path) as _cf_f:
            _cf_obj = json.load(_cf_f)
        _factors = (self.camera_factor1, self.camera_factor2, self.camera_factor3)
        _max_id = max(seg['id_end'] for seg in _cf_obj['camera_factor_segments'])
        _table = torch.full((_max_id + 1,), float('nan'), dtype=torch.float32)
        for seg in _cf_obj['camera_factor_segments']:
            _table[seg['id_start']:seg['id_end'] + 1] = _factors[seg['factor'] - 1]
        assert not torch.isnan(_table).any(), "camera_factor.json has gaps in ID coverage"
        # persistent=False: rebuilt from camera_factor.json at __init__, never
        # saved/loaded via checkpoint state_dict — keeps old checkpoints (saved
        # before this buffer existed) loadable under strict load.
        self.register_buffer('camera_factor_by_id', _table, persistent=False)
        print(f"[camera_factor] loaded {len(_cf_obj['camera_factor_segments'])} segments "
              f"from {cf_json_path}; covers ids 0..{_max_id}")
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
        # contribution_decay regularizer: separate cos range + weight + eps so
        # the BRDF-shape constraint can be tuned independently of the
        # pseudo-observation modes above.
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
        if cfg.model.emitter_calibration:
            self.emitter = ConstantEmitter(
                cfg = cfg.renderer.emitter,
                json_path = cfg.data.metadata_path
            )
        else:
            self.emitter = MultiAreaEmitter(
                cfg = cfg.renderer.emitter,
                # json_path = cfg.data.metadata_path
            )
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
    
    def _attach_cosine_schedulers(self, opts):
        """Wrap one or more optimizers with an epoch-based CosineAnnealingLR.

        Gated by ``cfg.model.optimizer.use_cosine_decay`` (default False — opt-in).
        Decay goes from the per-group peak LR down to ``eta_min`` over
        ``cfg.model.trainer.max_epochs`` epochs.
        """
        use_decay = getattr(self.hparams.model.optimizer, 'use_cosine_decay', False)
        if not use_decay:
            return opts

        max_epochs = self.hparams.model.trainer.max_epochs
        eta_min = getattr(self.hparams.model.optimizer, 'eta_min', 1e-5)

        is_list = isinstance(opts, list)
        opt_list = opts if is_list else [opts]

        result = []
        for opt in opt_list:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max_epochs, eta_min=eta_min,
            )
            result.append({
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
            })

        print(f"  [+] CosineAnnealingLR attached: T_max={max_epochs} epochs, eta_min={eta_min}")
        return result if is_list else result[0]

    def configure_optimizers(self):
        lr = self.hparams.model.optimizer.lr
        decoder_lr = getattr(self.hparams.model.optimizer, 'decoder_lr', lr)
        wd = self.hparams.model.optimizer.weight_decay
        opt_name = getattr(self.hparams.model.optimizer, 'name', 'Adam')

        has_latent_bank = hasattr(self.material, 'point_latent_bank') and self.material.point_latent_bank is not None
        embedding_params = list(self.material.point_latent_bank.parameters()) if has_latent_bank else []
        factor_params = [self.material.factor] if getattr(self.material, 'learnable_factor', False) else []

        decoder_params = [p for p in self.parameters()
                          if p not in set(embedding_params) and p not in set(factor_params)]

        if self.freeze_decoder:
            for p in decoder_params:
                p.requires_grad = False
            print("Decoder frozen – optimising latent bank and learnable factor (if present).")

        if opt_name == 'SparseAdam':
            latent_opt = torch.optim.SparseAdam(embedding_params, lr=lr)

            dense_params = decoder_params + factor_params if not self.freeze_decoder else factor_params
            if len(dense_params) > 0:
                dense_opt = torch.optim.Adam(
                    dense_params, lr=decoder_lr, betas=(0.9, 0.999), weight_decay=wd,
                )
                print(f"Using SparseAdam (embedding lr={lr}) + Adam (dense lr={decoder_lr})")
                return self._attach_cosine_schedulers([latent_opt, dense_opt])
            else:
                print(f"Using SparseAdam (embedding only), lr={lr}")
                return self._attach_cosine_schedulers(latent_opt)

        elif opt_name == 'SparseAdam8bit':
            import bitsandbytes as bnb
            latent_opt = bnb.optim.Adam8bit(embedding_params, lr=lr, betas=(0.9, 0.999))

            dense_params = decoder_params + factor_params if not self.freeze_decoder else factor_params
            if len(dense_params) > 0:
                dense_opt = torch.optim.Adam(
                    dense_params, lr=decoder_lr, betas=(0.9, 0.999), weight_decay=wd,
                )
                print(f"Using SparseAdam8bit (embedding lr={lr}) + Adam (dense lr={decoder_lr})")
                return self._attach_cosine_schedulers([latent_opt, dense_opt])
            else:
                print(f"Using SparseAdam8bit (embedding only), lr={lr}")
                return self._attach_cosine_schedulers(latent_opt)

        elif opt_name == 'Adam':
            opt_groups = []
            if embedding_params:
                opt_groups.append({'params': embedding_params, 'lr': lr})
            dense_params = decoder_params + factor_params if not self.freeze_decoder else factor_params
            if len(dense_params) > 0:
                opt_groups.append({'params': dense_params, 'lr': decoder_lr})
            if not opt_groups:
                opt_groups = [{'params': self.parameters(), 'lr': lr}]

            opt = torch.optim.Adam(opt_groups, betas=(0.9, 0.999), weight_decay=wd)
            print(f"Using Dense Adam (embedding lr={lr}, dense lr={decoder_lr})")
            return self._attach_cosine_schedulers(opt)

        elif opt_name == 'Adam8bit':
            import bitsandbytes as bnb
            opt_groups = []
            if embedding_params:
                opt_groups.append({'params': embedding_params, 'lr': lr})
            dense_params = decoder_params + factor_params if not self.freeze_decoder else factor_params
            if len(dense_params) > 0:
                opt_groups.append({'params': dense_params, 'lr': decoder_lr})
            if not opt_groups:
                opt_groups = [{'params': self.parameters(), 'lr': lr}]

            opt = bnb.optim.Adam8bit(opt_groups, betas=(0.9, 0.999), weight_decay=wd)
            print(f"Using Adam8bit (embedding lr={lr}, dense lr={decoder_lr})")
            return self._attach_cosine_schedulers(opt)

        elif opt_name == 'SGD':
            if self.freeze_decoder:
                params_to_optimize = embedding_params + factor_params
            else:
                params_to_optimize = list(self.parameters())

            optimizer = torch.optim.SGD(
                params_to_optimize,
                lr=lr,
                momentum=0.0,
                weight_decay=wd,
            )
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=int(self.hparams.model.optimizer.warmup_steps_ratio * self.hparams.model.trainer.max_steps),
                max_epochs=self.hparams.model.trainer.max_steps,
                eta_min=0,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            }

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

    def loss_function(self, rgbs, rgbs_gt, vis, camera_factor=None):
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
            if camera_factor is not None:
                # rho_ref is calibrated for camera_factor1; scale per-sample so all
                # groups sit in the same regime of the logrel curve.
                scale = camera_factor.squeeze(0).unsqueeze(-1) / self.camera_factor1
                ref = ref * scale

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

    def _zero_angle_aux_predictions(self, point_ids, material_ids):
        """Build a grazing-incidence augmentation in one of three modes.

        Modes (selected by ``cfg.model.grazing_mode``):
          * ``'zero_exact'`` / ``'near_zero_brdf'``  (pseudo-observations).
              wi.z = 0  (or c ~ U(cos_min, cos_max) for near_zero_brdf);
              target = 0 raw BRDF. ``pred`` is scaled by per-material
              ``camera_factor`` and concatenated into the real batch so the
              existing reconstruction loss handles it (effective weight is
              controlled by ``grazing_ratio``).
          * ``'contribution_decay'``  (BRDF-shape regularizer — NOT an
              observation). Sample c_high ~ U(grazing_decay_cos_min,
              grazing_decay_cos_max), alpha ~ U(0,1), c_low = alpha * c_high.
              Decode q_high = brdf(wi_high, wo) * c_high and
              q_low = brdf(wi_low, wo) * c_low with the same latent, the same
              wo and the same azimuth. **No camera_factor here** — this is a
              scale-relative shape constraint. The loss is a one-sided
              log-ratio hinge that only penalizes q_low > alpha * q_high:
                  excess     = log(clamp(q_low,0)  + eps)
                             - log(clamp(target,0) + eps)
                  per_sample = relu(excess).mean(dim=-1)
                  decay_loss = grazing_decay_weight * per_sample.sum() / B
              with B the real batch size (not n_grazing), so the total grows
              with grazing_ratio instead of being averaged away.

        Returns
        -------
        None when disabled.
        For 'zero_exact' / 'near_zero_brdf':
            ``{'kind': 'concat', 'pred': [N,3], 'target': [N,3], 'cf': [N]}``.
        For 'contribution_decay':
            ``{'kind': 'loss', 'loss': scalar tensor, 'diagnostics': {...}}``.

        Common to all modes: latents are drawn uniformly from the **entire
        latent bank**, wo is uniform on the upper hemisphere, normal=(0,0,1),
        material_id is recovered from each global_pid via searchsorted so the
        per-material ``camera_factor`` lookup still works.
        """
        n_total = point_ids.shape[1]
        n_grazing = int(self.grazing_ratio * n_total)
        if n_grazing <= 0:
            return None

        device = point_ids.device

        # Build sorted (offset, material_id) lookup once for global_pid -> material_id.
        # material_offset_tensor has zero-padding at gap material_ids, so we
        # use only the actually-present materials from metadata.
        if not hasattr(self, '_grazing_offsets_sorted'):
            mats = sorted(self.material.metadata['materials'],
                          key=lambda m: m['point_range'][0])
            self._grazing_offsets_sorted = torch.tensor(
                [m['point_range'][0] for m in mats], dtype=torch.long, device=device)
            self._grazing_mat_ids_sorted = torch.tensor(
                [m['material_id'] for m in mats], dtype=torch.long, device=device)

        total_latents = self.material.point_latent_bank.num_embeddings
        global_pids = torch.randint(0, total_latents, (n_grazing,), device=device)
        idx = torch.searchsorted(self._grazing_offsets_sorted, global_pids, right=True) - 1
        mids = self._grazing_mat_ids_sorted[idx]
        cf = self.camera_factor_by_id[mids]
        latent = self.material.point_latent_bank(global_pids)[:, :self.material.latent_dim]

        # Shared sampling: wo on upper hemisphere; azimuth phi_i for wi.
        z_o = torch.rand(n_grazing, device=device)
        phi_o = torch.rand(n_grazing, device=device) * (2 * math.pi)
        sin_t_o = torch.sqrt((1 - z_o ** 2).clamp(min=0))
        wo_local = torch.stack([sin_t_o * torch.cos(phi_o),
                                sin_t_o * torch.sin(phi_o), z_o], dim=-1)
        normal_local = torch.zeros(n_grazing, 3, device=device)
        normal_local[..., 2] = 1.0
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
            enc_dir = self.material.decoder.encode_directions(wi_local, wo_local, normal_local)
            brdf = self.material.decoder(enc_dir, latent)
            pred = brdf * cf.unsqueeze(-1)
            return {'kind': 'concat',
                    'pred': pred,
                    'target': torch.zeros_like(pred),
                    'cf': cf}

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

            enc_high = self.material.decoder.encode_directions(wi_high, wo_local, normal_local)
            enc_low = self.material.decoder.encode_directions(wi_low, wo_local, normal_local)
            brdf_high = self.material.decoder(enc_high, latent)
            brdf_low = self.material.decoder(enc_low, latent)

            # No camera_factor: this is a BRDF-shape regularizer.
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
        rays, rgbs_gt, emitter_ids, material_ids, camera_ids, point_ids, xyz = batch['rays'], batch['rgbs'], batch['emitter_ids'], batch['material_ids'], batch['camera_ids'], batch['point_ids'], batch['xyz']
        prior = 0.0
        if self.handeye_refiner:
            rays, prior = self.handeye_refiner.apply_handeye_delta_to_rays(rays, camera_ids)
        # forward renders
        rgbs, vis, ray_params, _, smooth_loss = self.renderer.stage1_render(self.emitter, rays, xyz, emitter_ids, material_ids, point_ids, self.cfg.renderer.spp.train, None, None, validation=False)
        # Per-material camera factor (lookup table — see __init__ for source).
        camera_factor = self.camera_factor_by_id[material_ids]
        rgbs = rgbs * camera_factor.squeeze(0).unsqueeze(-1)

        # --- Grazing augmentation: 'concat' modes add synthetic rays to the
        # batch so they share the main reconstruction loss; 'loss' mode (i.e.
        # contribution_decay) returns its own scalar that is added to
        # total_loss separately. ---
        grazing_aug = self._zero_angle_aux_predictions(point_ids, material_ids)
        grazing_loss_log = torch.tensor(0.0, device=rays.device)
        grazing_extra_loss = torch.tensor(0.0, device=rays.device)
        grazing_diag = {}
        if grazing_aug is not None:
            if grazing_aug['kind'] == 'concat':
                g_pred = grazing_aug['pred']
                g_gt = grazing_aug['target']
                g_cf = grazing_aug['cf']
                g_vis = torch.ones(g_pred.shape[0], dtype=torch.bool, device=rays.device)
                grazing_loss_log = self.loss_function(           # monitoring only
                    g_pred, g_gt.unsqueeze(0), g_vis, camera_factor=g_cf.unsqueeze(0))
                rgbs = torch.cat([rgbs, g_pred], dim=0)
                rgbs_gt = torch.cat([rgbs_gt.squeeze(0), g_gt], dim=0).unsqueeze(0)
                vis = torch.cat([vis, g_vis], dim=0)
                camera_factor = torch.cat([camera_factor.squeeze(0), g_cf], dim=0).unsqueeze(0)
            else:  # 'loss'
                grazing_extra_loss = grazing_aug['loss']
                grazing_loss_log = grazing_extra_loss.detach()
                grazing_diag = grazing_aug['diagnostics']

        loss = self.loss_function(rgbs, rgbs_gt, vis, camera_factor=camera_factor)

        # Add L2 gradient smoothness regularization
        smooth_weight = getattr(self.cfg.model.loss.reg_loss, 'weight', 0.0)
        if smooth_weight > 0:
            total_loss = loss + smooth_weight * smooth_loss
        else:
            total_loss = loss
        total_loss = total_loss + grazing_extra_loss

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
            'train/smooth_loss':  smooth_loss,
            'train/grazing_loss': grazing_loss_log,
            'train/psnr':         psnr,
        }
        for k, v in grazing_diag.items():
            log_dict[f'train/grazing_{k}'] = v
        self.log_dict(log_dict, prog_bar=True, batch_size=rays.shape[0])

        # Manual optimisation (required for SparseAdam with multiple optimiser groups)
        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]
        for opt in opts:
            opt.zero_grad()
        self.manual_backward(total_loss)
        for opt in opts:
            opt.step()

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
        rays, rgbs_gt, emitter_ids, material_ids, camera_ids, point_ids, xyz = batch['rays'], batch['rgbs'], batch['emitter_ids'], batch['material_ids'],  batch['camera_ids'], batch['point_ids'], batch['xyz']
        prior = 0.0
        if self.handeye_refiner:
            rays, prior = self.handeye_refiner.apply_handeye_delta_to_rays(rays, camera_ids)
        # forward renders
        rgbs, vis, ray_params, _, smooth_loss = self.renderer.stage1_render(self.emitter, rays, xyz, emitter_ids, material_ids, point_ids, self.cfg.renderer.spp.train, None, None, validation=False)
        # Per-material camera factor (lookup table — see __init__ for source).
        camera_factor = self.camera_factor_by_id[material_ids]
        rgbs = rgbs * camera_factor.squeeze(0).unsqueeze(-1)
        loss = self.loss_function(rgbs, rgbs_gt, vis, camera_factor=camera_factor)

        # Add L2 gradient smoothness regularization
        smooth_weight = getattr(self.cfg.model.loss.reg_loss, 'weight', 0.0)
        if smooth_weight > 0:
            total_loss = loss + smooth_weight * smooth_loss
        else:
            total_loss = loss

        psnr_loss  = torch.nn.functional.mse_loss(rgbs[vis], rgbs_gt.squeeze(0)[vis], reduction='mean')
        max_val = rgbs_gt.squeeze(0)[vis].max().clamp_min(1e-8)
        psnr       = 10.0 * torch.log10((max_val ** 2) / psnr_loss.clamp_min(1e-10))

        # ------------------------------------------------------------------
        # 7.  Logging  (now includes diagnostics)
        # ------------------------------------------------------------------
        self.log_dict({
            'val/loss':         loss,
            'val/recon_loss':   loss,
            'val/total_loss':   total_loss,
            'val/smooth_loss':  smooth_loss,
            'val/psnr':         psnr,
        }, prog_bar=True, batch_size=rays.shape[0])

        if self.more_visualizations:
            if batch_idx == 0:
                self.visualize_brdf_lobe(output_dir=os.path.join(self.cfg.exp_output_root_path, 'images'), num_latents=10, resolution=64)

        return loss
    
    def on_train_batch_start(self, batch, batch_idx):
        step = self.global_step
        self.trainer.train_dataloader.dataset.datasets.set_step(step)

    def visualize_brdf_lobe(self, output_dir, num_latents=5, resolution=64):
        """
        Visualize BRDF lobes by sampling random latents from point_latent_bank.
        Produces two polar plots per latent:
          1) Fix wi, vary wo  ->  BRDF(wo)
          2) Fix wo, vary wi  ->  BRDF(wi) * cos(theta_i)
        """
        if not hasattr(self.material, 'decoder') or not hasattr(self.material, 'point_latent_bank'):
            return

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        device = next(self.material.parameters()).device
        latent_dim = self.material.latent_dim
        total_points = self.material.point_latent_bank.num_embeddings

        # Randomly sample point indices
        torch.manual_seed(42)
        point_indices = torch.randint(0, total_points, (num_latents,), device=device)

        with torch.no_grad():
            all_latents = self.material.point_latent_bank(point_indices)  # [num_latents, total_latent_dim]
        brdf_latents = all_latents[:, :latent_dim]  # [num_latents, latent_dim]

        local_normal = torch.tensor([[0.0, 0.0, 1.0]], device=device)

        brdf_lobe_dir = os.path.join(output_dir, 'brdf_lobes')
        os.makedirs(brdf_lobe_dir, exist_ok=True)

        # =====================================================================
        # Visualization 1: Polar plot – Fix wi, vary wo
        # =====================================================================
        theta_i_values = [15.0, 30.0, 45.0, 60.0, 75.0]

        for latent_idx in range(num_latents):
            latent = brdf_latents[latent_idx:latent_idx+1]

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

            for theta_i_deg in theta_i_values:
                theta_i = np.radians(theta_i_deg)

                wi = torch.tensor([[
                    np.sin(theta_i),
                    0.0,
                    np.cos(theta_i)
                ]], device=device, dtype=torch.float32)

                theta_o_range = np.linspace(-np.pi/2, np.pi/2, resolution * 2)

                wo_batch = torch.zeros(len(theta_o_range), 3, device=device)
                for j, theta_o in enumerate(theta_o_range):
                    if theta_o >= 0:
                        wo_batch[j, 0] = -np.sin(theta_o)
                        wo_batch[j, 2] = np.cos(theta_o)
                    else:
                        wo_batch[j, 0] = np.sin(-theta_o)
                        wo_batch[j, 2] = np.cos(-theta_o)

                wi_batch = wi.expand(len(theta_o_range), -1)
                normal_batch = local_normal.expand(len(theta_o_range), -1)
                latent_batch = latent.expand(len(theta_o_range), -1)

                with torch.no_grad():
                    if hasattr(self.material.decoder, 'encode_directions'):
                        enc_dir = self.material.decoder.encode_directions(wi_batch, wo_batch, normal_batch)
                        brdf = self.material.decoder(enc_dir, latent_batch)
                    else:
                        # PBRDecoder takes (wi, wo, latent) directly in local frame
                        brdf, _ = self.material.decoder(wi_batch, wo_batch, latent_batch)

                brdf_polar = brdf.mean(dim=-1).cpu().numpy()

                polar_angles = theta_o_range
                ax.plot(polar_angles, brdf_polar, label=f'θ_i={theta_i_deg}°')

                specular_angle = np.radians(theta_i_deg)
                ax.axvline(x=specular_angle, color=ax.lines[-1].get_color(), linestyle='--', alpha=0.5)

            ax.set_theta_zero_location('N')
            ax.set_theta_direction(1)
            ax.set_thetamin(-90)
            ax.set_thetamax(90)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.set_title(f'BRDF Polar Plot (vary wo) - Point {point_indices[latent_idx].item()}\n'
                         f'(Fixed wi, vary wo; 0°=normal, dashed=specular direction)')
            plt.tight_layout()
            plt.savefig(os.path.join(brdf_lobe_dir, f'polar_brdf_vary_wo_pt_{point_indices[latent_idx].item()}.png'), dpi=150)
            plt.close()

        # =====================================================================
        # Visualization 2: Polar plot – Fix wo, vary wi  (BRDF * cos theta_i)
        # Original (wi_phi=0, wo_phi=0) plus 4 azimuth variations per latent.
        # =====================================================================
        theta_o_values = [15.0, 30.0, 45.0, 60.0]
        phi_configs = [
            (0.0,   0.0),
            (90.0,  0.0),
            (180.0, 0.0),
            (0.0,   90.0),
            (90.0,  90.0),
        ]

        for latent_idx in range(num_latents):
            latent = brdf_latents[latent_idx:latent_idx+1]
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

                        theta_i_range = np.linspace(-np.pi/2 + 0.01, np.pi/2 - 0.01, resolution * 2)
                        ti = torch.tensor(theta_i_range, device=device, dtype=torch.float32)
                        wi_batch = torch.stack([
                            -torch.sin(ti) * float(np.cos(wi_phi)),
                            -torch.sin(ti) * float(np.sin(wi_phi)),
                            torch.cos(ti),
                        ], dim=-1)

                        wo_batch = wo.expand(len(theta_i_range), -1)
                        normal_batch = local_normal.expand(len(theta_i_range), -1)
                        latent_batch = latent.expand(len(theta_i_range), -1)

                        with torch.no_grad():
                            if hasattr(self.material.decoder, 'encode_directions'):
                                enc_dir = self.material.decoder.encode_directions(wi_batch, wo_batch, normal_batch)
                                brdf = self.material.decoder(enc_dir, latent_batch)
                            else:
                                # PBRDecoder takes (wi, wo, latent) directly in local frame
                                brdf, _ = self.material.decoder(wi_batch, wo_batch, latent_batch)

                        cos_theta_i = wi_batch[:, 2].clamp(min=0).cpu().numpy()
                        brdf_polar = brdf.mean(dim=-1).cpu().numpy() * cos_theta_i

                        ax.plot(theta_i_range, brdf_polar, label=f'θ_o={theta_o_deg}°')
                        ax.axvline(x=np.radians(theta_o_deg),
                                   color=ax.lines[-1].get_color(), linestyle='--', alpha=0.5)

                    ax.set_theta_zero_location('N')
                    ax.set_theta_direction(1)
                    ax.set_thetamin(-90)
                    ax.set_thetamax(90)
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                    ax.set_title(
                        f'BRDF × cos(θ_i) - Point {point_indices[latent_idx].item()}\n'
                        f'wi_φ={wi_phi_deg:.0f}°, wo_φ={wo_phi_deg:.0f}° '
                        f'(±90° = grazing)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(
                        brdf_lobe_dir,
                        f'polar_brdf_vary_wi_pt_{point_indices[latent_idx].item()}'
                        f'_wiphi{int(wi_phi_deg):03d}_wophi{int(wo_phi_deg):03d}.png'), dpi=150)
                    plt.close()

        n_vary_wi = num_latents * len(phi_configs)
        print(f"[BRDF Lobe Visualization] Saved {num_latents + n_vary_wi} figures to {brdf_lobe_dir}")
