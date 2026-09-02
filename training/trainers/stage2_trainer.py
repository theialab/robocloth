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
from models.emitter import RealAreaEmitter, ConstantEmitter, MultiAreaEmitter, RotateAreaEmitter
from models.brdf import GreyPatchBRDF
import os
import glob
from utils.pose_refiner import GlobalHandEyeRefiner


def _keep_only_latest_result(output_dir, base):
    """Delete this view's result images from previous validation steps.

    The result filename carries the PSNR (and optionally dE/cd) tag, e.g.
    ``<base>_psnr25.30.png``/``.exr``, so a plain overwrite never happens —
    every step would otherwise leave its own file behind. Removing the prior
    ``<base>_psnr*`` matches (png and exr) keeps only the most recent step. GT
    and error images use a fixed, PSNR-free name and overwrite on their own, so
    they are left untouched.
    """
    for path in glob.glob(os.path.join(output_dir, base + '_psnr*')):
        try:
            os.remove(path)
        except OSError:
            pass


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


class Stage2Trainer(pl.LightningModule):
    """Stage 2 — dense single-material reconstruction (paper Sec. 4.3).

    Freezes the stage-1 decoder and optimizes the dense latent texture T_z,
    the parallax-aware query Q (neural_geometry) and the per-channel scale
    beta (learnable_factor) against captured HDR views. validation_step
    renders every held-out view at spp.val, computes per-view PSNR
    (peak = per-view GT max over visible pixels) and saves
    gt_view_*/result_view_*_psnr*.png — the paper's val/psnr metric is the
    epoch mean of these. Evaluation (model.test=True) re-runs exactly this
    on a trained checkpoint.
    """
    def __init__(self, cfg, material, gt_material, roughness, metallic):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.more_visualization = False
        self.use_white_balance = False
        self.use_tone_mapping = False  # When True, apply tone mapping + gamma and save as 8-bit PNG
        self.visualize_lobe = False
        self.compute_color_shift = False
        self.material = material
        self.freeze_decoder = cfg.model.freeze_decoder
        self.gt_material = gt_material
        self.gt_folder = cfg.gt_folder
        # Per-material camera-factor lookup (single material per stage2 run).
        # Only active when data.camera_factor_json is set explicitly: the paper
        # runs (and therefore the released checkpoints) used the single global
        # linear_factor below, so evaluation must keep that default.
        _ds_basename = os.path.basename(os.path.normpath(cfg.dataset_folder))
        try:
            _mid = int(_ds_basename)
            _cf_json_path = getattr(cfg.data, 'camera_factor_json', None)
            if not _cf_json_path:
                raise FileNotFoundError('data.camera_factor_json not set')
            with open(_cf_json_path) as _cf_f:
                _cf_obj = json.load(_cf_f)
            _f1 = getattr(cfg.renderer.camera, 'linear_factor1', None)
            _f2 = getattr(cfg.renderer.camera, 'linear_factor2', None)
            if _f1 is None or _f2 is None:
                raise AttributeError('renderer.camera.linear_factor1/linear_factor2 not in config')
            _f3 = _f2 * (8000.0 / 20000.0)
            _factors = (_f1, _f2, _f3)
            _factor_idx = None
            for _seg in _cf_obj['camera_factor_segments']:
                if _seg['id_start'] <= _mid <= _seg['id_end']:
                    _factor_idx = _seg['factor']
                    break
            if _factor_idx is None:
                raise KeyError(f"material_id {_mid} not covered by camera_factor.json")
            self.camera_factor = _factors[_factor_idx - 1]
            print(f"[camera_factor] stage2 material_id={_mid} → factor{_factor_idx}={self.camera_factor}")
        except (ValueError, FileNotFoundError, AttributeError, KeyError) as _e:
            # Non-numeric basename (UBO BTF / Bonn) or missing keys — legacy single factor
            self.camera_factor = cfg.renderer.camera.linear_factor
            print(f"[camera_factor] stage2 fallback to cfg.renderer.camera.linear_factor={self.camera_factor} ({_e})")
        print("Initializing stage2 trainer")
        
        #self.latent_dim = cfg.material.latent_dim
        # Create a mapping from roughness-metallic pairs to train latent indices
        self.radiance_rgb_pairs = {}
        self.average_radiance = []
        self.val_delta_e_list = []
        self.val_rgb_rel_err_list = []
        self.emitter_calibration = cfg.model.emitter_calibration    
        self.latent_reg_weight = cfg.model.latent_reg_weight if hasattr(cfg.model, 'latent_reg_weight') else 1e-4
        self.visualize_uv = cfg.data.visualize_uv
        self.inference_lr = cfg.model.optimizer.inference_lr
        self.inference_steps = cfg.model.optimizer.inference_steps
        self.is_graypatch = material.__class__.__name__ == 'GreyPatchBRDF'
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
            if cfg.renderer.emitter.type == 'rotatearea':
                self.emitter = RotateAreaEmitter(
                    cfg = cfg.renderer.emitter,
                )
            else:
                self.emitter = RealAreaEmitter(
                cfg = cfg.renderer.emitter,
                json_path = cfg.data.metadata_path
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

    def white_balance(self, x: torch.Tensor) -> torch.Tensor:
        """Chromatic adaptation from 4000K illuminant space to D65 (neutral white).

        Divides each channel by the 4000K-to-D65 illuminant ratio so that
        a neutral surface appears white on a standard display.
        Operates on the original HDR value range (e.g. 0-65535).
        """
        wb = torch.tensor([1.518, 1.000, 0.556], device=x.device, dtype=x.dtype)
        return x / wb

    def configure_optimizers(self):  
        # Exclude material.decoder parameters from optimization
        if self.freeze_decoder:
            print("Decoder frozen! Only optimizing latent bank.")
            decoder_params = set(self.material.decoder.parameters()) if hasattr(self.material, 'decoder') else set()
            params_to_optimize = [p for p in self.parameters() if p not in decoder_params]
        else:
            params_to_optimize = self.parameters()
        
        if self.hparams.model.optimizer.name == "SGD":
            optimizer = torch.optim.SGD(
                params_to_optimize,
                lr=self.hparams.model.optimizer.lr,
                momentum=0.9,
                weight_decay=1e-4,
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

        elif self.hparams.model.optimizer.name == 'Adam':
            optimizer = torch.optim.Adam(
                params_to_optimize,
                lr=self.hparams.model.optimizer.lr,
                betas=(0.9, 0.999),
                weight_decay=self.hparams.model.optimizer.weight_decay,
            )
            return optimizer

        elif self.hparams.model.optimizer.name == 'Adam8bit':
            if Adam8bitCompat is None:
                raise RuntimeError("Adam8bit requested but bitsandbytes is not installed.")
            optimizer = Adam8bitCompat(
                params_to_optimize,
                lr=self.hparams.model.optimizer.lr,
                betas=(0.9, 0.999),
                weight_decay=self.hparams.model.optimizer.weight_decay,
            )
            print(f"Using Adam8bit (lr={self.hparams.model.optimizer.lr})")
            return optimizer

        else:
            logging.error('Optimizer type not supported')

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
            return
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

    def visualize_brdf_lobe(self, output_dir, num_latents=10, resolution=64):
        """
        Visualize BRDF lobes by sampling latents from texture and evaluating BRDF
        for different wi/wo direction combinations.
        
        Args:
            output_dir: Directory to save visualization images
            num_latents: Number of random latent samples to visualize
            resolution: Resolution of the 2D BRDF slice (resolution x resolution)
        """
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        # Only works for AnisotropicLatentTexturedModel
        if self.hparams.material.type != "AnisotropicLatentTexturedModel":
            return
        
        device = next(self.material.parameters()).device
        
        # Get the latent texture (without blur for visualization)
        latent_texture = self.material.latent_texture.params  # [1, latent_dim, H, W]
        latent_dim = self.material.latent_dim
        
        # Randomly sample UV coordinates
        torch.manual_seed(42)  # For reproducibility
        random_uvs = torch.rand(num_latents, 2, device=device)  # [num_latents, 2]
        
        # Sample latents from texture using grid_sample
        grid_coords = random_uvs * 2.0 - 1.0  # Convert to [-1, 1]
        grid_coords = grid_coords.unsqueeze(0).unsqueeze(0)  # [1, 1, num_latents, 2]
        
        sampled_latents = torch.nn.functional.grid_sample(
            latent_texture,  # [1, D, H, W]
            grid_coords,     # [1, 1, num_latents, 2]
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # [1, D, 1, num_latents]
        sampled_latents = sampled_latents.squeeze(0).squeeze(1).transpose(0, 1)  # [num_latents, D]
        
        # Extract only BRDF latent (exclude frame and geometry latent)
        brdf_latents = sampled_latents[:, :latent_dim]  # [num_latents, latent_dim]
        
        # Local space normal is always (0, 0, 1)
        local_normal = torch.tensor([[0.0, 0.0, 1.0]], device=device)  # [1, 3]
        
        # Create directory for BRDF lobe visualizations
        brdf_lobe_dir = os.path.join(output_dir, 'brdf_lobes')
        os.makedirs(brdf_lobe_dir, exist_ok=True)
        
        # =====================================================================
        # Visualization 1: 3D BRDF Lobe - Fix wo, vary wi
        # wo is fixed at different elevation angles, wi varies over hemisphere
        # BRDF value determines the radius in each direction -> creates 3D lobe
        # =====================================================================
        from mpl_toolkits.mplot3d import Axes3D
        
        wo_elevations = [0.0, 30.0, 60.0]  # degrees from normal
        
        for latent_idx in range(num_latents):
            latent = brdf_latents[latent_idx:latent_idx+1]  # [1, latent_dim]
            
            fig = plt.figure(figsize=(5*len(wo_elevations), 5))
            
            for ax_idx, wo_elev in enumerate(wo_elevations):
                ax = fig.add_subplot(1, len(wo_elevations), ax_idx + 1, projection='3d')
                
                # Create fixed wo direction (in local space)
                wo_theta = np.radians(wo_elev)
                wo_phi = 0.0  # Fixed azimuth for wo
                wo = torch.tensor([[
                    np.sin(wo_theta) * np.cos(wo_phi),
                    np.sin(wo_theta) * np.sin(wo_phi),
                    np.cos(wo_theta)
                ]], device=device, dtype=torch.float32)  # [1, 3]
                
                # Create grid of wi directions (theta_i, phi_i)
                theta_range = np.linspace(0, np.pi/2, resolution)  # 0 to 90 degrees
                phi_range = np.linspace(0, 2*np.pi, resolution)    # 0 to 360 degrees
                theta_grid, phi_grid = np.meshgrid(theta_range, phi_range, indexing='ij')
                
                brdf_values = np.zeros((resolution, resolution))
                
                for i, theta_i in enumerate(theta_range):
                    # Batch all phi values for this theta
                    wi_batch = torch.zeros(resolution, 3, device=device)
                    for j, phi_i in enumerate(phi_range):
                        wi_batch[j, 0] = np.sin(theta_i) * np.cos(phi_i)
                        wi_batch[j, 1] = np.sin(theta_i) * np.sin(phi_i)
                        wi_batch[j, 2] = np.cos(theta_i)
                    
                    # Expand wo and latent for batch
                    wo_batch = wo.expand(resolution, -1)  # [resolution, 3]
                    normal_batch = local_normal.expand(resolution, -1)  # [resolution, 3]
                    latent_batch = latent.expand(resolution, -1)  # [resolution, latent_dim]
                    
                    # Encode directions
                    enc_dir = self.material.decoder.encode_directions(wi_batch, wo_batch, normal_batch)
                    
                    # Decode BRDF
                    with torch.no_grad():
                        brdf = self.material.decoder(enc_dir, latent_batch)  # [resolution, 3] or [resolution, 1]
                    
                    # Average over RGB channels
                    brdf_avg = brdf.mean(dim=-1).cpu().numpy()  # [resolution]
                    brdf_values[i, :] = brdf_avg
                
                # Convert spherical to Cartesian coordinates
                # radius = BRDF value (scaled for visualization)
                # This creates a 3D lobe where distance from origin = BRDF value
                r = brdf_values
                x = r * np.sin(theta_grid) * np.cos(phi_grid)
                y = r * np.sin(theta_grid) * np.sin(phi_grid)
                z = r * np.cos(theta_grid)
                
                # Normalize colors for the surface
                brdf_normalized = (brdf_values - brdf_values.min()) / (brdf_values.max() - brdf_values.min() + 1e-8)
                
                # Plot 3D surface (lobe)
                surf = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(brdf_normalized),
                                       alpha=0.8, linewidth=0, antialiased=True)
                
                # Draw the surface plane (z=0) for reference
                max_xy = max(np.abs(x).max(), np.abs(y).max()) * 1.2
                max_z = z.max() * 1.2 if z.max() > 0 else 1.0
                xx, yy = np.meshgrid(np.linspace(-max_xy, max_xy, 10),
                                     np.linspace(-max_xy, max_xy, 10))
                ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')
                
                # Draw normal direction arrow (scaled to z range)
                ax.quiver(0, 0, 0, 0, 0, max_z*0.8, color='red', arrow_length_ratio=0.1, linewidth=2)
                
                # Draw wo direction arrow (viewing direction, scaled appropriately)
                wo_np = wo[0].cpu().numpy()
                arrow_scale = max(max_xy, max_z) * 0.5
                ax.quiver(0, 0, 0, wo_np[0]*arrow_scale, 
                         wo_np[1]*arrow_scale, 
                         wo_np[2]*arrow_scale, 
                         color='blue', arrow_length_ratio=0.1, linewidth=2, label='wo')
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z (Normal)')
                ax.set_title(f'wo_θ={wo_elev}°\nBRDF: [{brdf_values.min():.3f}, {brdf_values.max():.3f}]\nz range: [{z.min():.3f}, {z.max():.3f}]')
                
                # Set axis limits based on actual data ranges (independent scaling for z)
                ax.set_xlim([-max_xy, max_xy])
                ax.set_ylim([-max_xy, max_xy])
                ax.set_zlim([0, max_z])
                ax.view_init(elev=30, azim=45)
            
            plt.suptitle(f'3D BRDF Lobe - Latent {latent_idx}\nUV: {random_uvs[latent_idx].cpu().numpy()}\nRed=Normal, Blue=wo')
            plt.tight_layout()
            plt.savefig(os.path.join(brdf_lobe_dir, f'lobe_3d_latent_{latent_idx}.png'), dpi=150)
            plt.close()
        
        # =====================================================================
        # Visualization 2: 2D Lobe in reflection plane
        # Fix theta_i at several values, show reflected lobe as polar curve
        # This shows the cross-section of the BRDF lobe in the plane of incidence
        # =====================================================================
        theta_i_values_lobe = [15.0, 30.0, 45.0, 60.0]  # degrees
        
        for latent_idx in range(num_latents):
            latent = brdf_latents[latent_idx:latent_idx+1]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Draw the surface (horizontal line at y=0)
            ax.axhline(y=0, color='brown', linewidth=3, label='Surface')
            
            # Draw the normal (vertical arrow)
            ax.annotate('', xy=(0, 1.0), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2))
            ax.text(0.05, 0.9, 'N', fontsize=12, color='green')
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(theta_i_values_lobe)))
            
            for idx, theta_i_deg in enumerate(theta_i_values_lobe):
                theta_i = np.radians(theta_i_deg)
                
                # wi direction (phi = 0, coming from the right)
                wi = torch.tensor([[
                    np.sin(theta_i),
                    0.0,
                    np.cos(theta_i)
                ]], device=device, dtype=torch.float32)
                
                # Vary theta_o from -90 to 90 degrees in the reflection plane
                num_samples = resolution * 2
                theta_o_range = np.linspace(-np.pi/2 + 0.01, np.pi/2 - 0.01, num_samples)
                
                wo_batch = torch.zeros(num_samples, 3, device=device)
                for j, theta_o in enumerate(theta_o_range):
                    if theta_o >= 0:
                        # phi_o = 180 (opposite side from wi)
                        wo_batch[j, 0] = -np.sin(theta_o)
                        wo_batch[j, 2] = np.cos(theta_o)
                    else:
                        # phi_o = 0 (same side as wi, for backscatter)
                        wo_batch[j, 0] = np.sin(-theta_o)
                        wo_batch[j, 2] = np.cos(-theta_o)
                
                wi_batch = wi.expand(num_samples, -1)
                normal_batch = local_normal.expand(num_samples, -1)
                latent_batch = latent.expand(num_samples, -1)
                
                enc_dir = self.material.decoder.encode_directions(wi_batch, wo_batch, normal_batch)
                
                with torch.no_grad():
                    brdf = self.material.decoder(enc_dir, latent_batch)
                
                brdf_values = brdf.mean(dim=-1).cpu().numpy()
                
                # Normalize BRDF for visualization (scale to reasonable size)
                brdf_scaled = brdf_values / (brdf_values.max() + 1e-8) * 0.8
                
                # Convert to Cartesian for the lobe curve
                # theta_o < 0: backscatter (same side as wi), theta_o > 0: forward scatter
                lobe_x = []
                lobe_z = []
                for j, theta_o in enumerate(theta_o_range):
                    r = brdf_scaled[j]
                    if theta_o >= 0:
                        # Outgoing direction on opposite side (negative x)
                        lobe_x.append(-r * np.sin(theta_o))
                        lobe_z.append(r * np.cos(theta_o))
                    else:
                        # Backscatter direction (same side, positive x)
                        lobe_x.append(r * np.sin(-theta_o))
                        lobe_z.append(r * np.cos(-theta_o))
                
                ax.fill(lobe_x, lobe_z, alpha=0.3, color=colors[idx])
                ax.plot(lobe_x, lobe_z, color=colors[idx], linewidth=2, 
                       label=f'θ_i={theta_i_deg}° (max={brdf_values.max():.2f})')
                
                # Draw incident light direction arrow
                wi_np = wi[0].cpu().numpy()
                arrow_len = 0.4
                ax.annotate('', xy=(0, 0), xytext=(wi_np[0]*arrow_len, wi_np[2]*arrow_len),
                           arrowprops=dict(arrowstyle='->', color=colors[idx], lw=1.5))
            
            ax.set_xlim([-1.2, 1.2])
            ax.set_ylim([-0.1, 1.2])
            ax.set_aspect('equal')
            ax.set_xlabel('X (horizontal)')
            ax.set_ylabel('Z (normal direction)')
            ax.set_title(f'BRDF Lobe Cross-Section - Latent {latent_idx}\n(Arrows = incident light wi)')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(brdf_lobe_dir, f'lobe_2d_slice_latent_{latent_idx}.png'), dpi=150)
            plt.close()
        
        # =====================================================================
        # Visualization 3: Polar plot of BRDF in reflection plane
        # Fix theta_i, show BRDF as function of theta_o (polar plot)
        # =====================================================================
        theta_i_values = [15.0, 30.0, 45.0, 60.0, 75.0]  # degrees
        
        for latent_idx in range(num_latents):
            latent = brdf_latents[latent_idx:latent_idx+1]
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
            
            for theta_i_deg in theta_i_values:
                theta_i = np.radians(theta_i_deg)
                
                # wi direction (phi = 0)
                wi = torch.tensor([[
                    np.sin(theta_i),
                    0.0,
                    np.cos(theta_i)
                ]], device=device, dtype=torch.float32)
                
                # Vary theta_o from -90 to 90 degrees (full reflection plane)
                theta_o_range = np.linspace(-np.pi/2, np.pi/2, resolution * 2)
                brdf_polar = np.zeros(len(theta_o_range))
                
                wo_batch = torch.zeros(len(theta_o_range), 3, device=device)
                for j, theta_o in enumerate(theta_o_range):
                    if theta_o >= 0:
                        # phi_o = 180 (opposite side from wi)
                        wo_batch[j, 0] = -np.sin(theta_o)
                        wo_batch[j, 2] = np.cos(theta_o)
                    else:
                        # phi_o = 0 (same side as wi, for backscatter)
                        wo_batch[j, 0] = np.sin(-theta_o)
                        wo_batch[j, 2] = np.cos(-theta_o)
                
                wi_batch = wi.expand(len(theta_o_range), -1)
                normal_batch = local_normal.expand(len(theta_o_range), -1)
                latent_batch = latent.expand(len(theta_o_range), -1)
                
                enc_dir = self.material.decoder.encode_directions(wi_batch, wo_batch, normal_batch)
                
                with torch.no_grad():
                    brdf = self.material.decoder(enc_dir, latent_batch)
                
                brdf_polar = brdf.mean(dim=-1).cpu().numpy()
                
                # Plot in polar coordinates (theta_o as angle, brdf as radius)
                # theta_o directly maps to polar angle:
                #   0° = normal direction (top of plot)
                #   +θ = opposite side from wi (right side, where specular reflection goes)
                #   -θ = same side as wi (left side, backscatter)
                polar_angles = theta_o_range  # Use theta_o directly (in radians)
                ax.plot(polar_angles, brdf_polar, label=f'θ_i={theta_i_deg}°')
                
                # Mark the expected specular reflection direction
                specular_angle = np.radians(theta_i_deg)
                ax.axvline(x=specular_angle, color=ax.lines[-1].get_color(), linestyle='--', alpha=0.5)
            
            ax.set_theta_zero_location('N')  # 0 degrees at top (normal direction)
            ax.set_theta_direction(1)  # Counter-clockwise (standard math convention)
            ax.set_thetamin(-90)
            ax.set_thetamax(90)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.set_title(f'BRDF Polar Plot (vary wo) - Latent {latent_idx}\n(Fixed wi, vary wo; 0°=normal, dashed=specular direction)')
            plt.tight_layout()
            plt.savefig(os.path.join(brdf_lobe_dir, f'polar_brdf_vary_wo_latent_{latent_idx}.png'), dpi=150)
            plt.close()
        
        # =====================================================================
        # Visualization 4: Polar plot of BRDF - Fix wo, vary wi
        # This is the flipped version: fix viewing direction, vary light direction
        # =====================================================================
        theta_o_values = [15.0, 30.0, 45.0, 60.0]  # degrees - fixed viewing angles
        
        for latent_idx in range(num_latents):
            latent = brdf_latents[latent_idx:latent_idx+1]
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
            
            for theta_o_deg in theta_o_values:
                theta_o = np.radians(theta_o_deg)
                
                # wo direction (fixed viewing direction, phi = 0)
                wo = torch.tensor([[
                    np.sin(theta_o),
                    0.0,
                    np.cos(theta_o)
                ]], device=device, dtype=torch.float32)
                
                # Vary theta_i from -90 to 90 degrees (full incidence plane)
                theta_i_range = np.linspace(-np.pi/2 + 0.01, np.pi/2 - 0.01, resolution * 2)
                
                wi_batch = torch.zeros(len(theta_i_range), 3, device=device)
                for j, theta_i in enumerate(theta_i_range):
                    if theta_i >= 0:
                        # phi_i = 180 (opposite side from wo)
                        wi_batch[j, 0] = -np.sin(theta_i)
                        wi_batch[j, 2] = np.cos(theta_i)
                    else:
                        # phi_i = 0 (same side as wo)
                        wi_batch[j, 0] = np.sin(-theta_i)
                        wi_batch[j, 2] = np.cos(-theta_i)
                
                wo_batch = wo.expand(len(theta_i_range), -1)
                normal_batch = local_normal.expand(len(theta_i_range), -1)
                latent_batch = latent.expand(len(theta_i_range), -1)
                
                enc_dir = self.material.decoder.encode_directions(wi_batch, wo_batch, normal_batch)
                
                with torch.no_grad():
                    brdf = self.material.decoder(enc_dir, latent_batch)
                
                # cos(theta_i) = wi · normal = wi.z (since normal is [0,0,1] in local space)
                cos_theta_i = wi_batch[:, 2].clamp(min=0).cpu().numpy()
                
                # BRDF * cos(theta_i) - the actual rendering contribution
                brdf_polar = brdf.mean(dim=-1).cpu().numpy() * cos_theta_i
                
                # Plot in polar coordinates (theta_i as angle, brdf*cos as radius)
                polar_angles = theta_i_range
                ax.plot(polar_angles, brdf_polar, label=f'θ_o={theta_o_deg}°')
                
                # Mark the expected specular reflection direction (mirror of wo)
                specular_angle = np.radians(theta_o_deg)
                ax.axvline(x=specular_angle, color=ax.lines[-1].get_color(), linestyle='--', alpha=0.5)
            
            ax.set_theta_zero_location('N')  # 0 degrees at top (normal direction)
            ax.set_theta_direction(1)
            ax.set_thetamin(-90)
            ax.set_thetamax(90)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.set_title(f'BRDF × cos(θ_i) Polar Plot - Latent {latent_idx}\n(Fixed wo, vary wi; 0°=normal, dashed=specular direction)')
            plt.tight_layout()
            plt.savefig(os.path.join(brdf_lobe_dir, f'polar_brdf_vary_wi_latent_{latent_idx}.png'), dpi=150)
            plt.close()
        
        print(f"[BRDF Lobe Visualization] Saved {num_latents * 4} figures to {brdf_lobe_dir}")

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
    
    def training_step(self, batch, batch_idx):
        """
        with importance sampling
        """
        # ------------------------------------------------------------------
        # 1. Un-pack inputs
        # ------------------------------------------------------------------
        rays, rgbs_gt, emitter_ids, camera_ids = batch['rays'], batch['rgbs'], batch['emitter_ids'], batch['camera_ids']
        prior = 0.0
        if self.handeye_refiner:
            rays, prior = self.handeye_refiner.apply_handeye_delta_to_rays(rays, camera_ids)
        # forward renders
        rgbs, vis, ray_params, _ = self.renderer.stage2_render(self.emitter, rays, emitter_ids, self.cfg.renderer.spp.train,
                                        None, None, validation=False)                       # f(r)
        rgbs = rgbs * self.camera_factor
        loss = self.loss_function(rgbs, rgbs_gt, vis)

        psnr_loss  = torch.nn.functional.mse_loss(rgbs[vis], rgbs_gt.squeeze(0)[vis], reduction='mean')
        max_val = rgbs_gt.squeeze(0)[vis].max().clamp_min(1e-8)
        psnr       = 10.0 * torch.log10((max_val ** 2) / psnr_loss.clamp_min(1e-10))

        # ------------------------------------------------------------------
        # 7.  Logging  (now includes diagnostics)
        # ------------------------------------------------------------------
        log_dict = {
            'train/recon_loss':   loss,
            'train/total_loss':   loss,
            'train/psnr':         psnr,
        }
        if getattr(self.material, 'learnable_factor', False):
            factor_val = self.material.factor.detach()
            if factor_val.ndim == 0:
                log_dict['train/learnable_factor'] = factor_val
            else:
                log_dict['train/learnable_factor_r'] = factor_val[0]
                log_dict['train/learnable_factor_g'] = factor_val[1]
                log_dict['train/learnable_factor_b'] = factor_val[2]
        self.log_dict(log_dict, prog_bar=True, batch_size=rays.shape[0])

        return loss

    def compute_color_shift_metrics(self, sample_rgbs_gt, sample_rgbs):
        """
        Compute Delta E (CIE76) and per-channel relative RGB error between
        predicted and ground-truth images (uint16 range 0-65535).

        Returns:
            dict with keys: mean_delta_e, mean_dL, mean_da, mean_db,
                            rel_R, rel_G, rel_B, avg_rgb_diff
        """
        gt_np = sample_rgbs_gt.cpu().numpy().astype(np.float64) / 65535.0
        res_np = sample_rgbs.cpu().numpy().astype(np.float64) / 65535.0
        mask = gt_np.sum(axis=-1) > 1e-7

        M_rgb2xyz = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        xyz_ref = np.array([0.95047, 1.0, 1.08883])
        delta_t = 6.0 / 29.0

        def lab_f(t):
            return np.where(t > delta_t**3, np.cbrt(t), t / (3 * delta_t**2) + 4.0 / 29.0)

        def rgb_to_lab(rgb):
            xyz = np.clip(rgb, 0, None) @ M_rgb2xyz.T
            xyz_n = xyz / xyz_ref
            fx, fy, fz = lab_f(xyz_n[..., 0]), lab_f(xyz_n[..., 1]), lab_f(xyz_n[..., 2])
            return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)

        gt_lab = rgb_to_lab(gt_np)
        res_lab = rgb_to_lab(res_np)
        dL = (res_lab[..., 0] - gt_lab[..., 0])[mask]
        da = (res_lab[..., 1] - gt_lab[..., 1])[mask]
        db = (res_lab[..., 2] - gt_lab[..., 2])[mask]
        delta_e_map = np.sqrt(dL**2 + da**2 + db**2)
        mean_delta_e = float(delta_e_map.mean())
        mean_dL = float(dL.mean())
        mean_da = float(da.mean())
        mean_db = float(db.mean())

        eps = 1e-6
        rel_err = (res_np - gt_np) / (gt_np + eps)
        rel_R = float(rel_err[mask, 0].mean())
        rel_G = float(rel_err[mask, 1].mean())
        rel_B = float(rel_err[mask, 2].mean())
        avg_rgb_diff = float((abs(rel_R - rel_G) + abs(rel_G - rel_B) + abs(rel_B - rel_R)) / 3.0)

        return {
            'mean_delta_e': mean_delta_e,
            'mean_dL': mean_dL,
            'mean_da': mean_da,
            'mean_db': mean_db,
            'rel_R': rel_R,
            'rel_G': rel_G,
            'rel_B': rel_B,
            'avg_rgb_diff': avg_rgb_diff,
        }

    def validation_step(self, batch, batch_idx):
        """ Unified validation step for both normal and radiometric calibration """
        rays, rgbs_gt, emitter_ids, camera_ids = batch['rays'], batch['rgbs'], batch['emitter_ids'], batch['camera_ids']
        prior = 0.0
        if self.handeye_refiner:
            rays, prior = self.handeye_refiner.apply_handeye_delta_to_rays(rays, batch['camera_ids'])

        # forward renders
        rgbs, vis, ray_params, extra_output = self.renderer.stage2_render(self.emitter, rays, emitter_ids, self.cfg.renderer.spp.val, None, None, validation=True)
        # Handle radiometric calibration
        if self.is_graypatch:
            pixel_all_ok, cosine_emitter_angle = extra_output
            self._radiometric_calibration_step(batch_idx, rgbs, rgbs_gt, vis, pixel_all_ok, cosine_emitter_angle)
        else:
            uv_offset = extra_output
        rgbs = rgbs * self.camera_factor
        psnr_loss = torch.nn.functional.mse_loss(rgbs[vis], rgbs_gt.squeeze(0)[vis], reduction='mean')
        max_val = rgbs_gt.squeeze(0)[vis].max().clamp_min(1e-8)
        psnr = 10.0 * torch.log10((max_val ** 2) / psnr_loss.clamp_min(1e-10))
        
        loss = self.loss_function(rgbs, rgbs_gt, vis)
        emitter_radiance = self.emitter.light_radiance.detach().cpu().numpy()
        
        # Handle batch of images
        batch_size = 1
        batched_rgbs = rgbs.reshape(batch_size, *self.img_hw, -1)
        batched_rgbs_gt = rgbs_gt.reshape(batch_size, *self.img_hw, -1)

        # Gate per-view file writes to the first valid_num val items so that
        # metrics still get computed on the full val set.
        valid_num   = getattr(self.cfg.data, 'valid_num', -1)
        save_visuals = (valid_num <= 0) or (batch_idx < valid_num)

        # Reshape uv_offset for visualization if not graypatch
        if not self.is_graypatch and self.visualize_uv:
            batched_uv_offset = uv_offset.reshape(batch_size, *self.img_hw, 2)

        for b in range(batch_size):
            # Reshape individual sample in batch
            sample_rgbs = batched_rgbs[b]
            sample_rgbs_gt = batched_rgbs_gt[b]

            # Create output directory for each sample
            output_dir = os.path.join(
                self.cfg.exp_output_root_path,
                f'images'
            )
            os.makedirs(output_dir, exist_ok=True)

            psnr_str = f'{psnr.item():.2f}'
            metric_suffix = f'_psnr{psnr_str}'

            # Color-shift metric accumulation runs every val item — these lists
            # feed the on_validation_epoch_end summary across the full set.
            if self.compute_color_shift:
                cm = self.compute_color_shift_metrics(sample_rgbs_gt, sample_rgbs)
                self.val_delta_e_list.append(cm['mean_delta_e'])
                self.val_rgb_rel_err_list.append([cm['rel_R'], cm['rel_G'], cm['rel_B'], cm['avg_rgb_diff']])
                de_str = f'{cm["mean_delta_e"]:.2f}'
                cdiff_str = f'{cm["avg_rgb_diff"]:.4f}'
                metric_suffix += f'_dE{de_str}_cd{cdiff_str}'

            if not save_visuals:
                continue

            # Drop this view's result file(s) from previous validation steps so
            # the PSNR-tagged result keeps only the latest step (GT/error use
            # fixed names and overwrite themselves).
            _keep_only_latest_result(output_dir, f'result_view_{batch_idx}_{b}')

            if self.more_visualization:
                # Save original images as 32-bit EXR without clipping
                sample_rgbs_gt_32bit = sample_rgbs_gt.cpu().numpy().astype(np.float32)/(65535.0*3.0) 
                sample_rgbs_32bit = sample_rgbs.cpu().numpy().astype(np.float32)/(65535.0*3.0)
                
                # Compute error image
                error_image = (sample_rgbs_32bit - sample_rgbs_gt_32bit).astype(np.float32)
                
                # Save as 32-bit EXR (OpenCV expects BGR)
                cv2.imwrite(
                    os.path.join(output_dir, f'gt_view_{batch_idx}_{b}.exr'),
                    cv2.cvtColor(sample_rgbs_gt_32bit, cv2.COLOR_RGB2BGR)
                )
                cv2.imwrite(
                    os.path.join(output_dir, f'result_view_{batch_idx}_{b}{metric_suffix}.exr'),
                    cv2.cvtColor(sample_rgbs_32bit, cv2.COLOR_RGB2BGR)
                )
                cv2.imwrite(
                    os.path.join(output_dir, f'error_view_{batch_idx}_{b}.exr'),
                    cv2.cvtColor(error_image, cv2.COLOR_RGB2BGR)
                )
            else:
                if self.use_tone_mapping:
                    # Apply tone mapping and save as 8-bit PNG
                    # Tone mapping handles arbitrary HDR range, maps [0, inf) to [0, 1)
                    sample_rgbs_gt_tonemapped = self.tone_mapping(sample_rgbs_gt/(65535.0))
                    sample_rgbs_tonemapped = self.tone_mapping(sample_rgbs/(65535.0))
                    
                    sample_rgbs_gt_8bit = (sample_rgbs_gt_tonemapped.cpu().numpy() * 255).astype(np.uint8)
                    sample_rgbs_8bit = (sample_rgbs_tonemapped.cpu().numpy() * 255).astype(np.uint8)
                    
                    # Save as 8-bit PNG (OpenCV expects BGR)
                    cv2.imwrite(
                        os.path.join(output_dir, f'gt_view_{batch_idx}_{b}.png'),
                        cv2.cvtColor(sample_rgbs_gt_8bit, cv2.COLOR_RGB2BGR)
                    )
                    cv2.imwrite(
                        os.path.join(output_dir, f'result_view_{batch_idx}_{b}{metric_suffix}.png'),
                        cv2.cvtColor(sample_rgbs_8bit, cv2.COLOR_RGB2BGR)
                    )
                else:
                    # White-balance from 4000K to D65 in HDR float, then save 16-bit PNG
                    if self.use_white_balance:
                        sample_rgbs_gt_wb = self.white_balance(sample_rgbs_gt)
                        sample_rgbs_wb = self.white_balance(sample_rgbs)
                        sample_rgbs_gt_16bit = np.clip(sample_rgbs_gt_wb.cpu().numpy(), 0, 65535).astype(np.uint16)
                        sample_rgbs_16bit = np.clip(sample_rgbs_wb.cpu().numpy(), 0, 65535).astype(np.uint16)
                    else:
                        sample_rgbs_gt_16bit = np.clip(sample_rgbs_gt.cpu().numpy(), 0, 65535).astype(np.uint16)
                        sample_rgbs_16bit = np.clip(sample_rgbs.cpu().numpy(), 0, 65535).astype(np.uint16)

                    # Save as 16-bit PNG (OpenCV expects BGR)
                    cv2.imwrite(
                        os.path.join(output_dir, f'gt_view_{batch_idx}_{b}.png'),
                        cv2.cvtColor(sample_rgbs_gt_16bit, cv2.COLOR_RGB2BGR)
                    )
                    cv2.imwrite(
                        os.path.join(output_dir, f'result_view_{batch_idx}_{b}{metric_suffix}.png'),
                        cv2.cvtColor(sample_rgbs_16bit, cv2.COLOR_RGB2BGR)
                    )
            
            # Visualize UV offsets as grayscale images
            if not self.is_graypatch and self.visualize_uv:
                sample_uv_offset = batched_uv_offset[b].cpu().numpy()  # Shape: [H, W, 2]
                u_offset = sample_uv_offset[:, :, 0]  # U offset
                v_offset = sample_uv_offset[:, :, 1]  # V offset
                
                # Normalize offsets to [0, 255] for visualization
                # Map the range of values to 0-255, with 127 representing zero offset
                u_min, u_max = u_offset.min(), u_offset.max()
                v_min, v_max = v_offset.min(), v_offset.max()
                
                # Normalize to [0, 255] with proper handling of positive and negative values
                u_range = max(abs(u_min), abs(u_max))
                v_range = max(abs(v_min), abs(v_max))
                
                if u_range > 0:
                    u_offset_vis = ((u_offset / u_range) * 127 + 127).astype(np.uint8)
                else:
                    u_offset_vis = np.full_like(u_offset, 127, dtype=np.uint8)
                    
                if v_range > 0:
                    v_offset_vis = ((v_offset / v_range) * 127 + 127).astype(np.uint8)
                else:
                    v_offset_vis = np.full_like(v_offset, 127, dtype=np.uint8)
                
                # Save grayscale offset images
                cv2.imwrite(
                    os.path.join(output_dir, f'u_offset_{batch_idx}_{b}.png'),
                    u_offset_vis
                )
                cv2.imwrite(
                    os.path.join(output_dir, f'v_offset_{batch_idx}_{b}.png'),
                    v_offset_vis
                ) 
                
                # Create colorful merged visualization using Red and Green channels
                # R=U offset, G=V offset, B=neutral (127)
                blue_channel = np.full_like(u_offset_vis, 127, dtype=np.uint8)
                uv_color = np.stack([u_offset_vis, v_offset_vis, blue_channel], axis=2)
                cv2.imwrite(
                    os.path.join(output_dir, f'uv_offset_color_{batch_idx}_{b}.png'),
                    cv2.cvtColor(uv_color, cv2.COLOR_RGB2BGR)
                )          
            # )
            
        if save_visuals:
            os.makedirs(os.path.join(self.cfg.exp_output_root_path, f'pbr_map_images'), exist_ok=True)
            self.save_pbr_texture(os.path.join(self.cfg.exp_output_root_path, f'pbr_map_images'), batch_idx, b)
        
        # Visualize BRDF lobes (only on first batch to avoid redundant visualizations)
        if self.visualize_lobe and batch_idx == 0:
            self.visualize_brdf_lobe(
                output_dir=os.path.join(self.cfg.exp_output_root_path, f'images'),
                num_latents=10,
                resolution=64
            )
        
        # Save PBR maps at first batch
        if batch_idx == 0:
            pbr_map_dir = os.path.join(self.cfg.exp_output_root_path, 'pbr_map_images')
            os.makedirs(pbr_map_dir, exist_ok=True)
            if hasattr(self.material, 'mipmap_latent_texture'):
                mlt = self.material.mipmap_latent_texture
                tex_pyramid = mlt.get_mipmap_textures(step=self.global_step, apply_blur=False)
                finest = tex_pyramid[0]  # [1, C, H, W]

                # Normal is stored at channels -6:-3 (only when predict_frame)
                if self.material.predict_frame:
                    normal_map = finest[:, -6:-3, :, :]
                    normal_map_vis = (normal_map + 1.0) / 2.0
                    torchvision.utils.save_image(
                        normal_map_vis,
                        os.path.join(pbr_map_dir, 'normal_map_batch0.png')
                    )

                # Color map: channels 0:3
                if mlt.prefilter:
                    color_map_vis = torch.sigmoid(finest[:, 0:3, :, :])
                    torchvision.utils.save_image(
                        color_map_vis,
                        os.path.join(pbr_map_dir, 'color_map_batch0.png')
                    )
                else:
                    for lvl, tex_lvl in enumerate(tex_pyramid):
                        color_map_vis = torch.sigmoid(tex_lvl[:, 0:3, :, :])
                        torchvision.utils.save_image(
                            color_map_vis,
                            os.path.join(pbr_map_dir, f'color_map_level{lvl}.png')
                        )

                # Roughness (channel 4) and metallic (channel 5) — finest level only
                roughness_map = torch.sigmoid(finest[:, 4:5, :, :])
                metallic_map = torch.sigmoid(finest[:, 5:6, :, :])
                torchvision.utils.save_image(
                    roughness_map,
                    os.path.join(pbr_map_dir, 'roughness_map_batch0.png')
                )
                torchvision.utils.save_image(
                    metallic_map,
                    os.path.join(pbr_map_dir, 'metallic_map_batch0.png')
                )
            elif hasattr(self.material, 'latent_texture'):
                # Non-mipmap path: single latent_texture.params [1, latent_dim, H, W]
                params = self.material.latent_texture.params

                if self.material.predict_frame:
                    normal_map = params[:, -6:-3, :, :]
                    normal_map_vis = (normal_map + 1.0) / 2.0
                    torchvision.utils.save_image(
                        normal_map_vis,
                        os.path.join(pbr_map_dir, 'normal_map_batch0.png')
                    )

                color_map_vis = torch.sigmoid(params[:, 0:3, :, :])
                torchvision.utils.save_image(
                    color_map_vis,
                    os.path.join(pbr_map_dir, 'color_map_batch0.png')
                )

                roughness_map = torch.sigmoid(params[:, 4:5, :, :])
                metallic_map = torch.sigmoid(params[:, 5:6, :, :])
                torchvision.utils.save_image(
                    roughness_map,
                    os.path.join(pbr_map_dir, 'roughness_map_batch0.png')
                )
                torchvision.utils.save_image(
                    metallic_map,
                    os.path.join(pbr_map_dir, 'metallic_map_batch0.png')
                )
            
        self.log('val/loss', loss)
        self.log('val/emitter_radiance', emitter_radiance.mean())
        if getattr(self.material, 'learnable_factor', False):
            factor_val = self.material.factor.detach()
            if factor_val.ndim == 0:
                self.log('val/learnable_factor', factor_val)
            else:
                self.log('val/learnable_factor_r', factor_val[0])
                self.log('val/learnable_factor_g', factor_val[1])
                self.log('val/learnable_factor_b', factor_val[2])
        self.log('val/psnr', psnr)
        return

    def test_step(self, batch, batch_idx):
        """ Unified validation step for both normal and radiometric calibration """
        rays, rgbs_gt, emitter_ids, camera_ids = batch['rays'], batch['rgbs'], batch['emitter_ids'], batch['camera_ids']

        # forward renders
        rgbs, vis, ray_params, extra_output = self.renderer.stage2_render(self.emitter, rays, emitter_ids, self.cfg.renderer.spp.val, None, None, validation=True)
        # Handle radiometric calibration
        uv_offset = extra_output
        rgbs = rgbs * self.camera_factor
        psnr_loss = torch.nn.functional.mse_loss(rgbs[vis], rgbs_gt.squeeze(0)[vis], reduction='mean')
        MAX_VAL = 65535.0
        max_val = MAX_VAL
        psnr = 10.0 * torch.log10((max_val ** 2) / psnr_loss.clamp_min(1e-10))
        
        loss = self.loss_function(rgbs, rgbs_gt, vis)
        emitter_radiance = self.emitter.light_radiance.detach().cpu().numpy()
        
        # Handle batch of images
        batch_size = 1
        batched_rgbs = rgbs.reshape(batch_size, *self.img_hw, -1)
        batched_rgbs_gt = rgbs_gt.reshape(batch_size, *self.img_hw, -1)
        
        for b in range(batch_size):
            # Reshape individual sample in batch
            sample_rgbs = batched_rgbs[b]
            sample_rgbs_gt = batched_rgbs_gt[b]
            
            # Create output directory for each sample
            output_dir = os.path.join(
                self.cfg.exp_output_root_path,
                f'test_images'
            )
            os.makedirs(output_dir, exist_ok=True)

            if self.more_visualization:
                # Save original images as 32-bit EXR without clipping
                sample_rgbs_gt_32bit = sample_rgbs_gt.cpu().numpy().astype(np.float32)/65535.0 
                sample_rgbs_32bit = sample_rgbs.cpu().numpy().astype(np.float32)/65535.0
                
                # Compute error image
                error_image = (sample_rgbs_32bit - sample_rgbs_gt_32bit).astype(np.float32)
                
                # Save as 32-bit EXR (OpenCV expects BGR)
                cv2.imwrite(
                    os.path.join(output_dir, f'gt_view_{batch_idx}_{b}.exr'),
                    cv2.cvtColor(sample_rgbs_gt_32bit, cv2.COLOR_RGB2BGR)
                )
                cv2.imwrite(
                    os.path.join(output_dir, f'result_view_{batch_idx}_{b}.exr'),
                    cv2.cvtColor(sample_rgbs_32bit, cv2.COLOR_RGB2BGR)
                )
                cv2.imwrite(
                    os.path.join(output_dir, f'error_view_{batch_idx}_{b}.exr'),
                    cv2.cvtColor(error_image, cv2.COLOR_RGB2BGR)
                )
            else:
                # Convert float32 (0-65535) to uint16 (0-65535)
                sample_rgbs_gt_16bit = np.clip(sample_rgbs_gt.cpu().numpy(), 0, 65535).astype(np.uint16)
                sample_rgbs_16bit = np.clip(sample_rgbs.cpu().numpy(), 0, 65535).astype(np.uint16)

                # Save as 16-bit PNG (OpenCV expects BGR)
                cv2.imwrite(
                    os.path.join(output_dir, f'gt_view_{batch_idx}_{b}.png'),
                    cv2.cvtColor(sample_rgbs_gt_16bit, cv2.COLOR_RGB2BGR)
                )
                cv2.imwrite(
                    os.path.join(output_dir, f'result_view_{batch_idx}_{b}.png'),
                    cv2.cvtColor(sample_rgbs_16bit, cv2.COLOR_RGB2BGR)
                )
            
        self.log('test/loss', loss)
        self.log('test/emitter_radiance', emitter_radiance.mean())
        self.log('test/psnr', psnr)        
        return
    
    def _radiometric_calibration_step(self, batch_idx, rgbs, rgbs_gt, vis, pixel_all_ok, cosine_emitter_angle):
        """Radiometric calibration: collect radiance-camera pairs with pixel coordinates."""
        mask = rgbs > 0
        rgbs_gt.squeeze(0)[~mask] = 0
        non_zero = (rgbs[vis].sum(dim=-1) > 0) & pixel_all_ok[vis]
        rad = rgbs[vis][non_zero].mean(dim=-1).detach().cpu().numpy()
        cam = rgbs_gt.squeeze(0)[vis][non_zero].mean(dim=-1).detach().cpu().numpy()
        cosine_emitter_angle = cosine_emitter_angle[vis][non_zero].detach().cpu().numpy()
        
        # Store pixel indices (flat indices in the image)
        # vis is already a mask for valid pixels, non_zero further filters within those
        vis_indices = torch.where(vis)[0]  # Get indices where vis is True
        non_zero_cpu = non_zero.cpu()
        pixel_indices = vis_indices[non_zero_cpu].cpu().numpy()  # Map to original flat indices
        
        average_radiance = cam.mean()
        self.average_radiance.append(average_radiance)
        print(f"Average radiance: {average_radiance}")
        self.radiance_rgb_pairs[batch_idx] = {
            'radiance': rad, 
            'camera': cam, 
            'cosine_emitter_angle': cosine_emitter_angle,
            'batch_idx': batch_idx,
            'pixel_indices': pixel_indices  # Store flat pixel indices
        }
    
    def fit_radiance_camera_linear_model(self, all_rad, all_cam):
        """
        Fit linear model cam = a * rad (passing through origin) using RANSAC to remove outliers.
        
        Args:
            all_rad (np.ndarray): Radiance values
            all_cam (np.ndarray): Camera RGB values
            
        Returns:
            dict: {'a', 'r2', 'inlier_mask', 'outlier_mask', 'mean_error', 'median_error'}
        """
        from sklearn.linear_model import RANSACRegressor, LinearRegression
        from sklearn.metrics import r2_score
        
        X = all_rad.reshape(-1, 1)
        # Force the model to pass through the origin by setting fit_intercept=False
        ransac = RANSACRegressor(
            estimator=LinearRegression(fit_intercept=False),
            random_state=42, 
            residual_threshold=None
        )
        ransac.fit(X, all_cam)
        
        a = ransac.estimator_.coef_[0]
        inlier_mask = ransac.inlier_mask_
        
        # Calculate R² score on inliers
        y_pred_inliers = ransac.predict(X[inlier_mask])
        r2 = r2_score(all_cam[inlier_mask], y_pred_inliers)
        
        # Calculate fitting errors (residuals) for inliers
        residuals = np.abs(all_cam[inlier_mask] - y_pred_inliers)
        mean_error = np.mean(residuals)
        median_error = np.median(residuals)
        
        return {
            'a': a, 
            'r2': r2, 
            'inlier_mask': inlier_mask, 
            'outlier_mask': ~inlier_mask,
            'mean_error': mean_error,
            'median_error': median_error
        }
    
    def on_train_batch_start(self, batch, batch_idx):
        step = self.global_step
        self.trainer.train_dataloader.dataset.datasets.set_step(step)

    def debug_multi_line_fitting_recursive_ransac(self, all_rad, all_cam, all_batch_ids, epoch, max_lines=5, min_points=100):
        """
        DEBUG FUNCTION: Fit multiple lines using recursive RANSAC.
        Each iteration finds the best line, removes inliers, and repeats.
        
        Args:
            all_rad: Array of radiance values
            all_cam: Array of camera RGB values
            all_batch_ids: Array of batch indices for each point
            epoch: Current epoch number
            max_lines: Maximum number of lines to fit
            min_points: Minimum points required to fit a line
            
        Returns:
            list: List of dictionaries containing line fitting results
        """
        from sklearn.linear_model import RANSACRegressor, LinearRegression
        from sklearn.metrics import r2_score
        
        remaining_mask = np.ones(len(all_rad), dtype=bool)
        line_results = []
        
        print(f"\n{'='*70}")
        print(f"DEBUG: Multi-Line Fitting Analysis (Epoch {epoch})")
        print(f"{'='*70}")
        
        # Recursively fit lines
        for line_idx in range(max_lines):
            if remaining_mask.sum() < min_points:
                print(f"Stopping: Only {remaining_mask.sum()} points remaining (< {min_points})")
                break
                
            X = all_rad[remaining_mask].reshape(-1, 1)
            y = all_cam[remaining_mask]
            
            # Adaptive min_samples based on remaining points
            n_remaining = remaining_mask.sum()
            adaptive_min_samples = min(50, max(10, n_remaining // 20))
            
            try:
                ransac = RANSACRegressor(
                    estimator=LinearRegression(fit_intercept=False),
                    random_state=42 + line_idx,
                    residual_threshold=None,
                    min_samples=adaptive_min_samples,
                    max_trials=1000
                )
                ransac.fit(X, y)
                
                slope = ransac.estimator_.coef_[0]
                local_inlier_mask = ransac.inlier_mask_
                
                # Check if we found enough inliers
                n_inliers = local_inlier_mask.sum()
                if n_inliers < min_points:
                    print(f"Line {line_idx + 1}: Only {n_inliers} inliers found (< {min_points}), stopping")
                    break
                
            except ValueError as e:
                print(f"RANSAC failed at iteration {line_idx + 1}: {e}")
                print(f"Remaining points: {n_remaining}, stopping line fitting")
                break
            
            # Map local inliers back to global indices
            global_indices = np.where(remaining_mask)[0]
            global_inliers = global_indices[local_inlier_mask]
            
            # Calculate metrics
            y_pred = slope * all_rad[global_inliers]
            r2 = r2_score(all_cam[global_inliers], y_pred)
            residuals = np.abs(all_cam[global_inliers] - y_pred)
            
            line_results.append({
                'slope': slope,
                'r2': r2,
                'indices': global_inliers,
                'batch_ids': all_batch_ids[global_inliers],
                'n_points': len(global_inliers),
                'mean_error': np.mean(residuals),
                'median_error': np.median(residuals),
                'radiance': all_rad[global_inliers],
                'camera': all_cam[global_inliers]
            })
            
            # Remove these inliers from remaining points
            remaining_mask[global_inliers] = False
            
            print(f"Line {line_idx + 1}: slope={slope:.6f}, R²={r2:.6f}, points={len(global_inliers)}, "
                  f"mean_err={np.mean(residuals):.4f}, median_err={np.median(residuals):.4f}")
        
        # Add remaining outliers as separate group
        if remaining_mask.sum() > 0:
            line_results.append({
                'slope': None,
                'r2': None,
                'indices': np.where(remaining_mask)[0],
                'batch_ids': all_batch_ids[remaining_mask],
                'n_points': remaining_mask.sum(),
                'mean_error': None,
                'median_error': None,
                'radiance': all_rad[remaining_mask],
                'camera': all_cam[remaining_mask]
            })
            print(f"Outliers: {remaining_mask.sum()} points (not fitted)")
        
        print(f"{'='*70}\n")
        
        # Visualize with color coding
        self._plot_multi_line_results(all_rad, all_cam, line_results, epoch)
        
        # Generate ground truth images with color overlays
        self._generate_colored_gt_images(line_results, epoch, self.radiance_rgb_pairs)
        
        return line_results
    
    def _plot_multi_line_results(self, all_rad, all_cam, line_results, epoch):
        """
        DEBUG FUNCTION: Plot scatter with each line group in different color (red to green by slope).
        Uses different axis ranges for better visualization like the original fitting figure.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Sort lines by slope for color mapping (red=low slope, green=high slope)
        valid_lines = [l for l in line_results if l['slope'] is not None]
        valid_lines.sort(key=lambda x: x['slope'])
        
        # Color map from red to green
        n_lines = len(valid_lines)
        if n_lines > 0:
            colors = cm.RdYlGn(np.linspace(0, 1, n_lines))
        else:
            colors = []
        
        # Plot each line group
        for i, line_info in enumerate(valid_lines):
            rad = line_info['radiance']
            cam = line_info['camera']
            
            ax.scatter(rad, cam, alpha=0.6, s=2, color=colors[i],
                      label=f"Line {i+1}: slope={line_info['slope']:.6f}, R²={line_info['r2']:.4f}, n={line_info['n_points']}")
            
            # Plot fitted line from 0 to max radiance
            rad_range = np.linspace(0, rad.max(), 100)
            ax.plot(rad_range, line_info['slope'] * rad_range, 
                   color=colors[i], linewidth=2.5, linestyle='--', alpha=0.8)
        
        # Plot outliers in gray
        outlier_info = [l for l in line_results if l['slope'] is None]
        if outlier_info:
            rad = outlier_info[0]['radiance']
            cam = outlier_info[0]['camera']
            ax.scatter(rad, cam, alpha=0.3, s=1, color='gray', 
                      label=f"Outliers: n={len(rad)}")
        
        # Plot ideal y=x line - use separate max for x and y axes
        max_rad = all_rad.max()
        max_cam = all_cam.max()
        max_val = max(max_rad, max_cam)
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, alpha=0.5, label='y=x (ideal)')
        
        ax.set_xlabel('Predicted Radiance', fontsize=14)
        ax.set_ylabel('Camera RGB', fontsize=14)
        ax.set_title(f'Multi-Line Fitting Analysis (Epoch {epoch})', fontsize=16, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Use different axis ranges like the original plot
        ax.set_xlim([0, max_rad])
        ax.set_ylim([0, max_cam])
        
        save_path = os.path.join(self.cfg.exp_output_root_path, f'debug_multiline_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Multi-line plot saved to: {save_path}")
    
    def _generate_colored_gt_images(self, line_results, epoch, radiance_rgb_pairs):
        """
        DEBUG FUNCTION: Generate ground truth images with colored overlays showing which line each pixel belongs to.
        Colors range from red (low slope) to green (high slope).
        Only colors pixels from non-main lines (skips the line with most points).
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        valid_lines = [l for l in line_results if l['slope'] is not None]
        valid_lines.sort(key=lambda x: x['slope'])
        n_lines = len(valid_lines)
        
        if n_lines == 0:
            print("No valid lines to visualize")
            return
        
        # Find the main line (line with most points) - we'll skip this in distribution
        main_line_idx = max(range(len(valid_lines)), key=lambda i: valid_lines[i]['n_points'])
        print(f"Main line (Line {main_line_idx + 1}) has {valid_lines[main_line_idx]['n_points']} points - will be shown as white/uncolored in images")
        
        # Create color map (brighter, more saturated colors)
        colors_rgb = cm.RdYlGn(np.linspace(0, 1, n_lines))[:, :3]  # RGB only, no alpha
        
        # For each unique image (batch_id), create colored overlay
        all_batch_ids = set()
        for line_info in line_results:
            all_batch_ids.update(line_info['batch_ids'])
        
        output_dir = os.path.join(self.cfg.exp_output_root_path, 'debug_colored_images')
        os.makedirs(output_dir, exist_ok=True)
        
        # Build a global index to (batch_id, local_pixel_idx) mapping
        # This maps the global concatenated array index to the original image pixel
        global_idx_to_pixel = {}
        current_global_idx = 0
        
        for batch_id in sorted(radiance_rgb_pairs.keys()):
            pairs = radiance_rgb_pairs[batch_id]
            pixel_indices = pairs['pixel_indices']
            n_pixels = len(pixel_indices)
            
            for local_idx in range(n_pixels):
                global_idx_to_pixel[current_global_idx] = (batch_id, pixel_indices[local_idx])
                current_global_idx += 1
        
        # Create pixel-to-line mapping for each image
        for batch_id in sorted(all_batch_ids):
            # Load the ground truth image
            gt_path = os.path.join(self.cfg.exp_output_root_path, 'images', f'gt_view_{int(batch_id)}_0.png')
            if not os.path.exists(gt_path):
                print(f"GT image not found: {gt_path}")
                continue
            
            # Load as 16-bit
            gt_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            if gt_img is None:
                print(f"Failed to load image: {gt_path}")
                continue
            
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            H, W = gt_img.shape[:2]
            
            # Normalize to 0-1 for display
            if gt_img.dtype == np.uint16:
                gt_img_normalized = gt_img.astype(np.float32) / 65535.0
            else:
                gt_img_normalized = gt_img.astype(np.float32) / 255.0
            
            # Create a color overlay (start with grayscale GT image)
            colored_img = gt_img_normalized.copy()
            
            # Get the pixel indices for this batch from the original data collection
            if batch_id not in radiance_rgb_pairs:
                print(f"No data for batch {batch_id}")
                continue
            
            # Color pixels belonging to each non-main line
            total_colored = 0
            for line_idx, line_info in enumerate(valid_lines):
                # Skip the main line
                if line_idx == main_line_idx:
                    continue
                
                # Find which global indices belong to this batch and this line
                global_indices_for_line = line_info['indices']
                
                for global_idx in global_indices_for_line:
                    if global_idx in global_idx_to_pixel:
                        bid, flat_pixel_idx = global_idx_to_pixel[global_idx]
                        
                        if bid == batch_id:
                            # Convert flat index to (row, col)
                            row = flat_pixel_idx // W
                            col = flat_pixel_idx % W
                            
                            if 0 <= row < H and 0 <= col < W:
                                # Color this pixel with the line's color
                                colored_img[row, col] = colors_rgb[line_idx]
                                total_colored += 1
            
            # Save the colored image
            if total_colored > 0:
                # Convert back to uint8 for saving
                colored_img_uint8 = (colored_img * 255).astype(np.uint8)
                
                save_path = os.path.join(output_dir, f'colored_gt_view_{int(batch_id)}_epoch_{epoch}.png')
                cv2.imwrite(save_path, cv2.cvtColor(colored_img_uint8, cv2.COLOR_RGB2BGR))
                print(f"Saved colored GT for image {int(batch_id)}: {total_colored} pixels colored, saved to {save_path}")
            else:
                print(f"Image {int(batch_id)}: No non-main-line pixels to color")
        
        # Create summary statistics file
        summary_path = os.path.join(output_dir, f'line_summary_epoch_{epoch}.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Multi-Line Fitting Summary (Epoch {epoch})\n")
            f.write(f"{'='*70}\n")
            f.write(f"Main Line: Line {main_line_idx + 1} with {valid_lines[main_line_idx]['n_points']} points (slope={valid_lines[main_line_idx]['slope']:.6f})\n")
            f.write(f"{'='*70}\n\n")
            
            for batch_id in sorted(all_batch_ids):
                f.write(f"\nImage {int(batch_id)}:\n")
                f.write(f"{'-'*50}\n")
                
                total_pixels = 0
                line_pixel_counts = []
                
                for line_idx, line_info in enumerate(valid_lines):
                    mask = line_info['batch_ids'] == batch_id
                    pixel_count = mask.sum()
                    total_pixels += pixel_count
                    
                    if pixel_count > 0:
                        color_hex = '#%02x%02x%02x' % tuple((colors_rgb[line_idx] * 255).astype(int))
                        is_main = " (MAIN)" if line_idx == main_line_idx else ""
                        line_pixel_counts.append((line_idx + 1, pixel_count, line_info['slope'], color_hex, is_main))
                
                # Check outliers
                outlier_info = [l for l in line_results if l['slope'] is None]
                if outlier_info:
                    mask = outlier_info[0]['batch_ids'] == batch_id
                    outlier_count = mask.sum()
                    total_pixels += outlier_count
                else:
                    outlier_count = 0
                
                if total_pixels == 0:
                    f.write("  No pixels analyzed\n")
                    continue
                
                # Sort by pixel count descending
                line_pixel_counts.sort(key=lambda x: x[1], reverse=True)
                
                for line_num, count, slope, color, is_main in line_pixel_counts:
                    percentage = 100.0 * count / total_pixels
                    f.write(f"  Line {line_num}{is_main} (slope={slope:.6f}, {color}): {count} pixels ({percentage:.2f}%)\n")
                
                if outlier_count > 0:
                    percentage = 100.0 * outlier_count / total_pixels
                    f.write(f"  Outliers: {outlier_count} pixels ({percentage:.2f}%)\n")
                
                f.write(f"  Total: {total_pixels} pixels\n")
        
        print(f"Line summary saved to: {summary_path}")
        
        # Create distribution plot - EXCLUDE main line to better visualize minority lines
        fig, ax = plt.subplots(figsize=(12, 8))
        
        batch_ids_sorted = sorted(all_batch_ids)
        x_pos = np.arange(len(batch_ids_sorted))
        
        # Prepare data for stacked bar chart (excluding main line)
        secondary_lines = [i for i in range(n_lines) if i != main_line_idx]
        line_distributions = np.zeros((len(secondary_lines), len(batch_ids_sorted)))
        
        for i, batch_id in enumerate(batch_ids_sorted):
            for plot_idx, line_idx in enumerate(secondary_lines):
                line_info = valid_lines[line_idx]
                mask = line_info['batch_ids'] == batch_id
                line_distributions[plot_idx, i] = mask.sum()
        
        # Create stacked bar chart
        bottom = np.zeros(len(batch_ids_sorted))
        for plot_idx, line_idx in enumerate(secondary_lines):
            ax.bar(x_pos, line_distributions[plot_idx], bottom=bottom, 
                   color=colors_rgb[line_idx], alpha=0.8,
                   label=f"Line {line_idx+1} (slope={valid_lines[line_idx]['slope']:.6f})")
            bottom += line_distributions[plot_idx]
        
        ax.set_xlabel('Image ID', fontsize=12)
        ax.set_ylabel('Number of Pixels', fontsize=12)
        ax.set_title(f'Non-Main Line Distribution Across Images (Epoch {epoch})\nMain line (Line {main_line_idx+1}, {valid_lines[main_line_idx]["n_points"]} pts) excluded', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([int(bid) for bid in batch_ids_sorted], rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        dist_plot_path = os.path.join(output_dir, f'line_distribution_epoch_{epoch}.png')
        plt.savefig(dist_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Line distribution plot (excluding main line) saved to: {dist_plot_path}")

    def on_validation_epoch_end(self):
        if self.val_delta_e_list:
            de_arr = np.array(self.val_delta_e_list)
            rel_arr = np.array(self.val_rgb_rel_err_list)  # [N, 4]: rel_R, rel_G, rel_B, avg_diff
            print(f"\n{'='*60}")
            print(f"Validation Epoch {self.current_epoch} - Color Metrics ({len(de_arr)} images)")
            print(f"  Delta E (CIE76):       mean={de_arr.mean():.4f}, std={de_arr.std():.4f}")
            print(f"  Rel err R:             mean={rel_arr[:,0].mean():.6f}, std={rel_arr[:,0].std():.6f}")
            print(f"  Rel err G:             mean={rel_arr[:,1].mean():.6f}, std={rel_arr[:,1].std():.6f}")
            print(f"  Rel err B:             mean={rel_arr[:,2].mean():.6f}, std={rel_arr[:,2].std():.6f}")
            print(f"  Avg RGB channel diff:  mean={rel_arr[:,3].mean():.6f}, std={rel_arr[:,3].std():.6f}")
            print(f"{'='*60}\n")
            self.val_delta_e_list.clear()
            self.val_rgb_rel_err_list.clear()

        """Plot accumulated radiance-RGB pairs at the end of validation."""
        if not self.is_graypatch or not self.radiance_rgb_pairs:
            return
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        # Concatenate all data and track batch indices
        all_rad = []
        all_cam = []
        all_angle = []
        all_batch_ids = []
        for batch_idx, pairs in sorted(self.radiance_rgb_pairs.items()):
            rad, cam, cosine_emitter_angle = pairs['radiance'], pairs['camera'], pairs['cosine_emitter_angle']
            all_angle.append(cosine_emitter_angle)
            all_rad.append(rad)
            all_cam.append(cam)
            # Create array of batch_idx repeated for each point
            all_batch_ids.append(np.full(len(rad), batch_idx))
        
        all_rad = np.concatenate(all_rad)
        all_cam = np.concatenate(all_cam)
        all_angle = np.concatenate(all_angle)
        all_batch_ids = np.concatenate(all_batch_ids)
        
        # If emitter calibration mode, only plot angle vs cam/rad ratio
        if self.emitter_calibration:
            # Convert cosine to degrees
            angle_degrees = np.arccos(np.clip(all_angle, -1.0, 1.0)) * 180.0 / np.pi
            
            # Compute cam/rad ratio
            cam_rad_ratio = all_cam / (all_rad + 1e-8)  # Add epsilon to avoid division by zero
            
            # Create plot
            fig = plt.figure(figsize=(12, 8))
            plt.scatter(angle_degrees, cam_rad_ratio, alpha=0.5, s=1, color='blue')
            plt.xlim([0, 90])
            plt.ylim([0, cam_rad_ratio.max() * 1.1])  # Start from 0, extend 10% above max
            plt.xlabel('Emitter Angle (degrees)', fontsize=12)
            plt.ylabel('Camera / Radiance Ratio', fontsize=12)
            plt.title(f'Emitter Angle vs Cam/Rad Ratio (Epoch {self.current_epoch})', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(self.cfg.exp_output_root_path, f'emitter_calibration_epoch_{self.current_epoch}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"\n{'='*60}")
            print(f"Emitter Calibration Plot saved to: {save_path}")
            print(f"{'='*60}\n")
            
            # Compute relative radiance table with 1 degree resolution
            import json
            from scipy.spatial import cKDTree
            
            # Normalize cam_rad_ratio to max value
            max_ratio = cam_rad_ratio.max()
            relative_ratio = cam_rad_ratio / max_ratio
            
            # Create 1-degree bins (0-1, 1-2, ..., 89-90)
            resolution = 1.0
            angle_bins = np.arange(0, 90, resolution)
            
            # Compute average ratio for each bin
            bin_centers = []
            bin_ratios = []
            bins_with_data = []
            
            for i, bin_start in enumerate(angle_bins):
                bin_end = bin_start + resolution
                # Find all points within this bin
                mask = (angle_degrees >= bin_start) & (angle_degrees < bin_end)
                
                if np.any(mask):
                    # Average the relative ratios in this bin
                    avg_ratio = np.mean(relative_ratio[mask])
                    bin_centers.append(bin_start + resolution / 2)  # Use bin center
                    bin_ratios.append(avg_ratio)
                    bins_with_data.append(i)
                else:
                    # No data in this bin, will use nearest neighbor later
                    bin_centers.append(bin_start + resolution / 2)
                    bin_ratios.append(None)
            
            # Fill missing bins using nearest neighbor from bins with data
            if bins_with_data:
                # Build KD-Tree from bin centers that have data
                valid_centers = np.array([bin_centers[i] for i in bins_with_data]).reshape(-1, 1)
                valid_ratios = np.array([bin_ratios[i] for i in bins_with_data])
                tree = cKDTree(valid_centers)
                
                # Fill in missing bins
                for i in range(len(bin_ratios)):
                    if bin_ratios[i] is None:
                        # Find nearest bin with data
                        dist, idx = tree.query([[bin_centers[i]]], k=1)
                        bin_ratios[i] = valid_ratios[idx[0]]
            
            # Create the table as a dictionary
            calibration_table = {
                "resolution_degrees": resolution,
                "angle_range": [0.0, 90.0],
                "max_cam_rad_ratio": float(max_ratio),
                "epoch": self.current_epoch,
                "data": {
                    f"{center:.1f}": float(ratio)
                    for center, ratio in zip(bin_centers, bin_ratios)
                }
            }
            
            # Save to JSON file
            json_path = os.path.join(self.cfg.exp_output_root_path, f'emitter_calibration_table_epoch_{self.current_epoch}.json')
            with open(json_path, 'w') as f:
                json.dump(calibration_table, f, indent=2)
            
            print(f"Emitter Calibration Table saved to: {json_path}")
            print(f"Table contains {len(calibration_table['data'])} entries from 0° to 90° with 0.1° resolution")
            print(f"Max cam/rad ratio: {max_ratio:.6f}")
            print(f"{'='*60}\n")
            
            self.average_radiance.clear()
            self.radiance_rgb_pairs.clear()
            return
        
        # === DEBUG: Multi-line fitting analysis ===
        # UNCOMMENT TO ENABLE DEBUG MODE:
        # self.debug_multi_line_fitting_recursive_ransac(
        #     all_rad, all_cam, all_batch_ids, self.current_epoch,
        #     max_lines=5, min_points=100
        # )
        
        # Fit linear model using RANSAC
        fit_res = self.fit_radiance_camera_linear_model(all_rad, all_cam)
        a, r2 = fit_res['a'], fit_res['r2']
        inlier_mask, outlier_mask = fit_res['inlier_mask'], fit_res['outlier_mask']
        mean_error, median_error = fit_res['mean_error'], fit_res['median_error']
        
        # Find top 50 outliers where ground truth (cam) >> prediction (rad * a)
        predicted = all_rad * a
        residuals = all_cam - predicted  # Positive means cam > prediction
        outlier_indices = np.argsort(residuals)[-50:][::-1]  # Top 50 largest residuals
        
        # Print and save fitting results
        print(f"\n{'='*60}")
        print(f"Linear Fitting Results (Epoch {self.current_epoch}):")
        print(f"  Equation: cam = {a:.6f} * rad")
        print(f"  R² score (inliers): {r2:.6f}")
        print(f"  Mean absolute error: {mean_error:.6f}")
        print(f"  Median absolute error: {median_error:.6f}")
        print(f"  Inliers: {inlier_mask.sum()} / {len(inlier_mask)} ({100*inlier_mask.sum()/len(inlier_mask):.2f}%)")
        print(f"  Outliers: {outlier_mask.sum()}")
        print(f"{'='*60}\n")
        
        print(f"Top 50 Outliers (Ground Truth >> Prediction):")
        print(f"{'Rank':<6} {'Image ID':<10} {'GT Radiance':<15} {'Predicted':<15} {'Error':<15}")
        print(f"{'-'*65}")
        for rank, idx in enumerate(outlier_indices, 1):
            img_id = all_batch_ids[idx]
            gt_val = all_cam[idx]
            pred_val = predicted[idx]
            error = residuals[idx]
            print(f"{rank:<6} {int(img_id):<10} {gt_val:<15.4f} {pred_val:<15.4f} {error:<15.4f}")
        print(f"{'-'*65}\n")
        
        results_path = os.path.join(self.cfg.exp_output_root_path, f'fitting_results_epoch_{self.current_epoch}.txt')
        with open(results_path, 'w') as f:
            f.write(f"Linear Fitting Results (Epoch {self.current_epoch}):\n")
            f.write(f"Equation: cam = {a:.6f} * rad\n")
            f.write(f"R² score (inliers): {r2:.6f}\n")
            f.write(f"Mean absolute error: {mean_error:.6f}\n")
            f.write(f"Median absolute error: {median_error:.6f}\n")
            f.write(f"Inliers: {inlier_mask.sum()} / {len(inlier_mask)} ({100*inlier_mask.sum()/len(inlier_mask):.2f}%)\n")
            f.write(f"Outliers: {outlier_mask.sum()}\n")
            f.write(f"\nTop 10 Outliers (Ground Truth >> Prediction):\n")
            f.write(f"{'Rank':<6} {'Image ID':<10} {'GT Radiance':<15} {'Predicted':<15} {'Error':<15}\n")
            f.write(f"{'-'*65}\n")
            for rank, idx in enumerate(outlier_indices, 1):
                img_id = all_batch_ids[idx]
                gt_val = all_cam[idx]
                pred_val = predicted[idx]
                error = residuals[idx]
                f.write(f"{rank:<6} {int(img_id):<10} {gt_val:<15.4f} {pred_val:<15.4f} {error:<15.4f}\n")
        
        # Create plot
        fig = plt.figure(figsize=(12, 8))
        
        # Plot data points
        for idx, (batch_idx, pairs) in enumerate(sorted(self.radiance_rgb_pairs.items())):
            rad, cam = pairs['radiance'], pairs['camera']
            plt.scatter(rad, cam, alpha=0.5, s=1, color='blue')
        
        # Plot fitted line and reference
        max_rad, max_cam = all_rad.max(), all_cam.max()
        rad_range = np.linspace(0, max_rad, 100)
        plt.plot(rad_range, a * rad_range, 'g-', linewidth=2, 
                label=f'Fitted: cam = {a:.4f}*rad\nR² = {r2:.4f}')
        plt.plot([0, max(max_rad, max_cam)], [0, max(max_rad, max_cam)], 
                'r--', linewidth=2, label='y=x (ideal)')
        
        max_rad, max_cam = all_rad.max(), all_cam.max()
        plt.xlim([0, max_rad])
        plt.ylim([0, max_cam])
        plt.xlabel('Predicted Radiance', fontsize=12)
        plt.ylabel('Camera RGB', fontsize=12)
        plt.title(f'Radiance vs Camera RGB (Epoch {self.current_epoch})', fontsize=14)
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.cfg.exp_output_root_path, f'radiance_rgb_epoch_{self.current_epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Plot average radiance w.r.t index
        plt.figure()
        plt.plot(self.average_radiance)
        plt.savefig(os.path.join(self.cfg.exp_output_root_path, f'average_radiance.png'))
        plt.close()
        
        self.average_radiance.clear()
        self.radiance_rgb_pairs.clear()