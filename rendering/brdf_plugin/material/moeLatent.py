import torch
import torch.nn as nn
import torch.nn.functional as NF
import math
from pytorch_lightning import LightningModule
import sys
from brdf_plugin.utils.ops import *
from brdf_plugin.utils.cuda_manage import print_cuda_memory_info
from brdf_plugin.utils.moe import MoE

def load_latent_texture_from_checkpoint(checkpoint_path, device='cuda'):
    """
    Load latent_texture parameter from an existing checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file (.ckpt or .pth)
        device (str): Device to load the tensor on ('cuda', 'cpu', etc.)
    
    Returns:
        torch.Tensor: The latent_texture tensor with shape [1, latent_dim, H, W]
    
    Example:
        >>> latent_texture = load_latent_texture_from_checkpoint('checkpoints/model.ckpt')
        >>> print(latent_texture.shape)
        torch.Size([1, 32, 256, 256])
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Try different common checkpoint formats
    latent_texture = None
    
    print(f"Checkpoint top-level keys: {list(checkpoint.keys())}")
    
    # Format 1: Direct state_dict
    if 'latent_texture' in checkpoint:
        latent_texture = checkpoint['latent_texture']
    
    # Format 2: PyTorch Lightning checkpoint format
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"Found state_dict with {len(state_dict)} keys")
        
        # Print all keys that contain 'latent' for debugging
        latent_keys = [key for key in state_dict.keys() if 'latent' in key.lower()]
        print(f"Keys containing 'latent': {latent_keys}")
        
        # Look for latent_texture in state_dict (may have prefixes like 'material.latent_texture')
        for key in state_dict.keys():
            if 'latent_texture' in key:
                latent_texture = state_dict[key]
                print(f"Found latent_texture at key: '{key}'")
                break
    
    # Format 3: Model stored directly
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        if isinstance(state_dict, dict):
            for key in state_dict.keys():
                if 'latent_texture' in key:
                    latent_texture = state_dict[key]
                    break
    
    # Format 4: Try all keys in checkpoint
    if latent_texture is None:
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                for sub_key in checkpoint[key].keys():
                    if 'latent_texture' in sub_key:
                        latent_texture = checkpoint[key][sub_key]
                        break
                if latent_texture is not None:
                    break
    
    if latent_texture is None:
        # Print more detailed error information
        if 'state_dict' in checkpoint:
            all_keys = list(checkpoint['state_dict'].keys())
            print(f"\nAll keys in state_dict ({len(all_keys)} total):")
            for i, k in enumerate(all_keys[:20]):  # Print first 20 keys
                print(f"  {k}")
            if len(all_keys) > 20:
                print(f"  ... and {len(all_keys) - 20} more keys")
        
        raise ValueError(
            f"Could not find 'latent_texture' in checkpoint. "
            f"Check the printed keys above to find the correct parameter name."
        )
    
    # Ensure it's on the correct device
    latent_texture = latent_texture.to(device)
    
    print(f"Loaded latent_texture with shape: {latent_texture.shape}")
    return latent_texture.cuda()

def load_pbr_texture(pbr_folder):
    import cv2
    from PIL import Image
    import torchvision.transforms as T
    import os

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

    # Detect cloth name from path
    cloth_name = "fabric_pattern_07"  # default
    if "denim" in pbr_folder.lower():
        cloth_name = "denim_fabric_03"
    
    # Collect PBR texture data
    try:
        # Read all texture maps based on the detected cloth name
        if cloth_name == "denim_fabric_03":
            diffuse = read_img(f"{cloth_name}_diff_4k.jpg")                    # [3,H,W] - Diffuse map
            ao = read_img(f"{cloth_name}_ao_4k.jpg")                         # [3,H,W] - Ambient occlusion
            arm = read_img(f"{cloth_name}_arm_4k.jpg")                       # [3,H,W] - (A, Roughness, Metalness)
            rough = read_img(f"{cloth_name}_rough_4k.exr")                   # [1,H,W] - Roughness map (EXR)
            metal = read_img(f"{cloth_name}_metal_4k.exr")                   # [1,H,W] - Metallic map (EXR)
            spec_ior = read_img(f"{cloth_name}_spec_ior_4k.exr")             # [1,H,W] - Specular IOR (EXR)
            aniso_rot = read_img(f"{cloth_name}_anisotropy_rotation_4k.jpg") # [3,H,W] - Anisotropy rotation
            aniso_str = read_img(f"{cloth_name}_anisotropy_strength_4k.jpg") # [3,H,W] - Anisotropy strength
            nor_dx = read_img(f"{cloth_name}_nor_dx_4k.exr")                 # [3,H,W] - Normal map X
            nor_gl = read_img(f"{cloth_name}_nor_gl_4k.exr")                 # [3,H,W] - Normal map GL
            
            # Combine all available channels for denim fabric
            # [Color (3) + AO (3) + ARM (3) + Roughness (1) + Metal (1) + Spec IOR (1) + Aniso Rot (3) + Aniso Str (3) + Normal DX (1) + Normal GL (1)] = 20 channels
            tex = torch.cat([
                diffuse,                # RGB color (3 channels) 0:3
                ao,                   # Ambient occlusion (3 channels) 3:6
                arm,                  # ARM texture (3 channels) 6:9
                rough,                # Roughness map (1 channel) 9:10
                metal,                # Metallic map (1 channel) 10:11
                spec_ior,             # Specular IOR (1 channel) 11:12
                aniso_rot,            # Anisotropy rotation (3 channels) 12:15
                aniso_str,            # Anisotropy strength (3 channels) 15:18
                nor_dx,               # Normal map X (3 channels) 18:21
                nor_gl                # Normal map GL (3 channels) 21:24
            ], dim=0)  # [24,H,W]
        else:
            # Default fabric pattern maps
            col_1 = read_img(f"{cloth_name}_col_1_4k.jpg")      # [3,H,W] - Color map
            ao = read_img(f"{cloth_name}_ao_4k.jpg")            # [3,H,W] - Ambient occlusion
            arm = read_img(f"{cloth_name}_arm_4k.jpg")          # [3,H,W] - (A, Roughness, Metalness)
            rough = read_img(f"{cloth_name}_rough_4k.exr")      # [1,H,W] - Roughness map (EXR)
            nor_dx = read_img(f"{cloth_name}_nor_dx_4k.exr")    # [3,H,W] - Normal map X
            nor_gl = read_img(f"{cloth_name}_nor_gl_4k.exr")    # [3,H,W] - Normal map GL
            
            # Combine all available channels for default fabric
            # [Color (3) + AO (3) + ARM (3) + Roughness (1) + Normal DX (1) + Normal GL (1)] = 12 channels
            tex = torch.cat([
                col_1,                # RGB color (3 channels) 0:3
                ao,                   # Ambient occlusion (3 channels) 3:6
                arm,                  # ARM texture (3 channels) 6:9
                rough,                # Roughness map (1 channel) 9:10
                nor_dx,               # Normal map X (3 channels) 10:13
                nor_gl                # Normal map GL (3 channels) 13:16
            ], dim=0)  # [16,H,W]
        
        return tex.permute(1, 2, 0).contiguous()  # [H,W,C] => [U,V,C]
    except Exception as e:
        print(f"Error loading PBR textures: {e}")
        # Return a default texture if loading fails
        H, W = 1024, 1024
        return torch.ones(H, W, 12)  # Default to 12 channels for compatibility

class MoeLatentModel(LightningModule):
    """ MLP-based BRDF class with 2D texture latent grids """
    def __init__(self, cfg):
        super().__init__()

        # Latent dimension from config
        self.latent_dim = cfg.latent_dim
        self.colorful_texture = cfg.colorful_texture
        self.larger_latent_dim = cfg.larger_latent_dim
        self.different_decoder = cfg.different_decoder
        self.predict_frame = cfg.predict_frame
        self.gt_frame = cfg.gt_frame
        self.anisotropic = True
        self.neural_geometry = cfg.neural_geometry.enable
        self.geometry_latent_dim = cfg.neural_geometry.latent_dim
        self.local_wi_wo = cfg.neural_geometry.local_wi_wo
        self.neural_geometry_pos_enc = cfg.neural_geometry.positional_encoding
        self.recompute_frame = cfg.neural_geometry.recompute_frame
        if self.colorful_texture and self.larger_latent_dim:
            total_latent_dim = self.latent_dim * 3
        else:
            total_latent_dim = self.latent_dim
        if self.predict_frame:
            total_latent_dim = total_latent_dim + 6
            
        if self.neural_geometry:
            total_latent_dim = total_latent_dim + self.geometry_latent_dim # 8 for neural geometry latent

        
        
        # Create 2D texture latent grids
        self.texture_resolution = getattr(cfg, 'texture_resolution', 256)
        # Initialize latent texture with special initialization for directional components
        latent_init = torch.randn(1, total_latent_dim, self.texture_resolution, self.texture_resolution) * 0.1
        
        if self.predict_frame:
            # Last 6 dimensions: normal (1,0,0) and tangent (0,1,0)
            latent_init[:, -6:-3, :, :] = torch.tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1)  # normal
            latent_init[:, -3:, :, :] = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1)    # tangent
        
        #self.latent_texture = nn.Parameter(latent_init)
        self.latent_texture = load_latent_texture_from_checkpoint(cfg.latent_texture_path)
        # gaussian blur parameters
        self.Gaussian_blur = cfg.Gaussian_blur
        self.blur_sigma0 = 8.0
        self.blur_half_life = 3333
        # Add SH positional encoding module
        self.degree = 3
        self.pos_enc = True
        if self.pos_enc:
            self.sh_encoder = lambda x: components_from_spherical_harmonics(self.degree, x)
            
        # Calculate input dimension after SH encoding
        sh_dim = num_sh_bases(self.degree)
        encoded_input_dim = sh_dim * 3  # wi, wo, normal each encoded by SH
        
        # Add latent dimension to input
        input_dim = encoded_input_dim + self.latent_dim if self.pos_enc else cfg.input_channels + self.latent_dim
        
        # Build preprocessing MLP before MoE
        self.pre_moe_hidden_dim = getattr(cfg.moe, 'pre_moe_hidden_dim', 128)
        self.pre_moe_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.pre_moe_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.pre_moe_hidden_dim, self.pre_moe_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.pre_moe_hidden_dim, self.pre_moe_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.pre_moe_hidden_dim, self.pre_moe_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.pre_moe_hidden_dim, self.latent_dim),
            nn.ReLU(),
        )
        
        # MoE configuration
        self.num_experts = getattr(cfg.moe, 'num_experts', 8)
        self.moe_k = getattr(cfg.moe, 'moe_k', 4)
        self.moe_hidden_size = getattr(cfg.moe, 'moe_hidden_size', 128)
        self.moe_loss_coef = getattr(cfg.moe, 'moe_loss_coef', 1e-2)
        self.moe_loss_coef=float(self.moe_loss_coef)

        print("self.num_experts", self.num_experts)
        print("self.moe_hidden_size", self.moe_hidden_size)
        print("self.moe_loss_coef", self.moe_loss_coef)
        # Build MoE layers (input is now from pre_moe_mlp)
        if self.different_decoder:
            # Create separate MoE for RGB channels
            self.moe_r = MoE(input_dim, self.latent_dim, cfg.output_channels, self.num_experts, self.moe_hidden_size, noisy_gating=True, k=self.moe_k)
            self.moe_g = MoE(input_dim, self.latent_dim, cfg.output_channels, self.num_experts, self.moe_hidden_size, noisy_gating=True, k=self.moe_k)
            self.moe_b = MoE(input_dim, self.latent_dim, cfg.output_channels, self.num_experts, self.moe_hidden_size, noisy_gating=True, k=self.moe_k)
        else:
            # Single MoE for all channels
            self.moe = MoE(input_dim, self.latent_dim, cfg.output_channels, self.num_experts, self.moe_hidden_size, noisy_gating=True, k=self.moe_k)
        
        # Store MoE loss for training
        self.moe_loss = 0.0

        # Build geometry decoder if neural geometry is enabled
        if self.neural_geometry:
            layers = []
            prev_dim = encoded_input_dim + self.geometry_latent_dim if cfg.neural_geometry.positional_encoding else 6 + self.geometry_latent_dim
            for hidden_dim in cfg.neural_geometry.hidden_layers:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if cfg.activation.lower() == "relu":
                    layers.append(nn.ReLU())
                prev_dim = hidden_dim
                
            layers.append(nn.Linear(prev_dim, cfg.neural_geometry.output_channels))
            layers.append(nn.LeakyReLU(0.2))
            
            self.geometry_decoder = nn.Sequential(*layers)
        # Initialize proxy BRDF for importance sampling
        #self.proxy_brdf = ProxyPBRBRDF()  # Default roughness

    def _gaussian_kernel(self, sigma: float, channels: int):
        """Return a (C×1×k×k) kernel usable by depth-wise conv2d."""
        if sigma < 0.5:                       # almost no blur → skip
            return None
        radius  = int(math.ceil(3 * sigma))
        ksize   = 2 * radius + 1
        grid    = torch.arange(-radius, radius + 1,
                               dtype=self.latent_texture.dtype,
                               device=self.latent_texture.device)
        g1d     = torch.exp(-0.5 * (grid / sigma) ** 2)
        g1d     = g1d / g1d.sum()
        g2d     = (g1d[:, None] * g1d[None, :]).expand(
                    channels, 1, ksize, ksize)
        return g2d

    def _blur_latent(self, step: int):
        """Return blurred copy of latent texture for this training step."""
        # σ(t) = σ₀ · 2^{-t/h}
        sigma = self.blur_sigma0 * (0.5 ** (step / self.blur_half_life))
        kernel = self._gaussian_kernel(sigma, self.latent_texture.shape[1])
        if kernel is None:                       # σ<0.5 → no-op
            return self.latent_texture
        pad = kernel.shape[-1] // 2
        # depth-wise ⇒ groups = channels
        return NF.conv2d(self.latent_texture, kernel,
                        padding=pad, groups=self.latent_texture.shape[1])

    def compute_uv(self, pos, width=0.4, length=0.4):
        half_w   = width  * 0.5               # 0.2
        half_l   = length * 0.5               # 0.2
        x, z     = pos[:, 0], pos[:, 2]

        u = (x + half_w) / width              # (-0.2→0,  +0.2→1)
        v = (z + half_l) / length             # (-0.2→0,  +0.2→1)

        return torch.stack([u, v], dim=-1)     # (N,2)

    def sample_latent_from_texture(self, uv, texture):
        """
        Sample latent codes from 2D texture using bilinear interpolation
        Args:
            uv: Bx2 UV coordinates
        Returns:
            latent: BxD latent codes
        """
        # Convert sphere positions to UV coordinates
        
        # Convert UV to grid coordinates for F.grid_sample
        # grid_sample expects coordinates in [-1, 1] range
        grid_coords = uv * 2.0 - 1.0  # Convert [0,1] to [-1,1]
        grid_coords = grid_coords.unsqueeze(1).unsqueeze(0)  # 1x1xBx2
        
        # Sample from latent texture using bilinear interpolation
        latent = NF.grid_sample(
            texture,  # 1xDxHxW
            grid_coords,          # 1x1xBx2
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # 1xDx1xB
        
        # Reshape to BxD
        latent = latent.squeeze(0).squeeze(-1).transpose(0, 1)  # BxD
        
        return latent

    def forward(self, enc_dir, latent=None, channel=None):
        """
        Evaluate BRDF using MLP with 2D texture latent encoding, wi,wo and normalare in local space
        Args:
            pos: Bx3 position on sphere surface
            wi: Bx3 incoming light direction 
            wo: Bx3 outgoing view direction
            normal: Bx3 normal
            latent: ignored (for compatibility)
            global_step: ignored (for compatibility)
        Returns:
            brdf: Bx1 BRDF values
        """
        
        # Pass through preprocessing MLP before MoE
        x = self.pre_moe_mlp(latent)
            
        if self.different_decoder:
            if channel == 'r':
                output, moe_loss = self.moe_r(torch.cat([enc_dir, latent], dim=-1), self.moe_loss_coef)
            elif channel == 'g':
                output, moe_loss = self.moe_g(torch.cat([enc_dir, latent], dim=-1), self.moe_loss_coef)
            else:
                output, moe_loss = self.moe_b(torch.cat([enc_dir, latent], dim=-1), self.moe_loss_coef)
            self.moe_loss = moe_loss
            return output
        else:
            output, moe_loss = self.moe(torch.cat([enc_dir, latent], dim=-1), self.moe_loss_coef)
            self.moe_loss = moe_loss
            return output

    def world_to_local(self, v, normal, tangent):
        
        # choose arbitrary tangent
        if tangent is None:
            tangent = torch.cross(normal, torch.tensor([0.0, 0.0, 1.0], device=normal.device).expand_as(normal))
        tangent_len = tangent.norm(dim=-1, keepdim=True)
        tangent = tangent / tangent_len
        bitangent = torch.cross(normal, tangent)

        v_local = torch.stack([
            (v * tangent).sum(dim=-1),
            (v * bitangent).sum(dim=-1),
            (v * normal).sum(dim=-1)
        ], dim=-1)

        return v_local
    
    def eval_brdf(self, gt_params, pos, wi, wo, normal,uv, TBN, latent=None, batch_mask=None, footprint_vis=None, dp_du=None, dp_dv=None):
        """
        Evaluate BRDF and pdf after transforming world-space vectors to local space.
        Args:
            gt_params: dictionary of ground truth parameters
            pos: Bx3 position
            wi: Bx3 light direction in world space
            wo: Bx3 viewing direction in world space
            normal: Bx3 normal in world space
            latent: optional latent code
            batch_mask: optional batch mask
        Returns:
            brdf: Bx3 BRDF values
            pdf: Bx1 probability
        """
        # Ensure normal is normalized
        NoL = (wi*normal).sum(-1,keepdim=True)
        NoV = (wo*normal).sum(-1,keepdim=True)

        
        if self.Gaussian_blur:
            tex = self._blur_latent(self.global_step)
        else:
            tex = self.latent_texture       
        latent = self.sample_latent_from_texture(uv, tex)
        if self.gt_frame:
            """ load gt frame for reference """
            tangent = None
            factor = 1.0
            params = self.pbr_texture
            H, W = params.shape[1], params.shape[2]
            crop_h = int((H - H * factor) // 2)
            crop_w = int((W - W * factor) // 2)
            new_h = int(H * factor)
            new_w = int(W * factor)
            params = params[:, crop_h:crop_h+new_h, crop_w:crop_w+new_w, :]
            #uv = compute_uv(pos, 0.8, 0.8)
            # Step 1: TBN frame
            T, B, N_geo = compute_tbn(pos, uv, 0.8, 0.8)
            # Step 2: Texture sampling
            sampled_texture = sample_texture(params, uv)
            if self.anisotropic:
                # Extract anisotropic texture maps
                diffuse = sampled_texture[:, 0:3]                    # Color channels
                ao = sampled_texture[:, 3:4]                       # Ambient occlusion
                arm = sampled_texture[:, 6:9]                      # ARM channels
                roughness_map = sampled_texture[:, 9:10]           # Roughness map
                metallic_map = sampled_texture[:, 10:11]           # Metallic map
                ior_map = sampled_texture[:, 11:12]                # Specular IOR
                aniso_rot = sampled_texture[:, 12:15]              # Anisotropy rotation
                aniso_str = sampled_texture[:, 15:18]              # Anisotropy strength
                normal_local = sampled_texture[:, 18:21]           # Normal channels (DX)
                
                # Extract material properties
                roughness = roughness_map                          # Use dedicated roughness map
                metallic = metallic_map                            # Use dedicated metallic map
                ior = ior_map                                      # Use IOR map
                
                # Compute anisotropic roughness parameters
                # aniso_str controls the strength of anisotropy (0 = isotropic, 1 = fully anisotropic)
                aniso_strength = aniso_str[:, 0:1]                 # Use first channel of anisotropy strength
                base_roughness = roughness.clamp(0.02, 1.0)       # Clamp roughness to valid range
                
                # Compute ax and ay based on anisotropy strength
                ax = base_roughness * (1.0 + aniso_strength)      # Tangent direction roughness
                ay = base_roughness * (1.0 - aniso_strength * 0.5) # Bitangent direction roughness
                # Step 3: Transform local normal to world space
                n_world = local_to_world_normal(normal_local, T, B, N_geo)
                rotation_angle = aniso_rot[:, 0:1] * 2.0 * torch.pi  # Convert [0,1] to [0, 2π]
                
                # Create rotation matrix in tangent space (rotate around normal)
                cos_theta = torch.cos(rotation_angle)
                sin_theta = torch.sin(rotation_angle)
                
                # Rotate the tangent vector in the T-B plane
                T_rot = cos_theta * T + sin_theta * B
                T_reproj = T_rot - (T_rot * n_world).sum(-1, keepdim=True) * n_world
                T_ortho  = NF.normalize(T_reproj, dim=-1)
                # The rotated tangent is already in world space since T and B are in world space
                B_ortho  = torch.cross(n_world, T_ortho, dim=-1)
                B_ortho  = NF.normalize(B_ortho, dim=-1)  
                
            else:
                arm = sampled_texture[:, 6:9]      # ARM channels
                color = sampled_texture[:, 0:3]    # Color channels
                normal_local = sampled_texture[:, 10:13]  # Normal channels
                albedo, roughness, metallic = arm[:, 0:1], arm[:, 1:2], arm[:, 2:3]
                # Step 4: Transform local normal to world
                n_world = local_to_world_normal(normal_local, T, B, N_geo)
                
            predicted_normal = n_world
            predicted_tangent = T_ortho
            
        if self.predict_frame:
            # Extract predicted normal and tangent from latent
            predicted_normal = latent[..., -6:-3]  # Last 6-3 dimensions for normal
            predicted_tangent = latent[..., -3:]   # Last 3 dimensions for tangent       
            # Normalize predicted vectors
            predicted_normal = torch.nn.functional.normalize(predicted_normal, dim=-1)
            predicted_tangent = torch.nn.functional.normalize(predicted_tangent, dim=-1)
            
            # Use Gram-Schmidt orthogonalization to make tangent perpendicular to normal
            # Keep normal unchanged and orthogonalize tangent
            predicted_tangent = predicted_tangent - torch.sum(predicted_tangent * predicted_normal, dim=-1, keepdim=True) * predicted_normal
            predicted_tangent = torch.nn.functional.normalize(predicted_tangent, dim=-1)
            
        wi_local = self.world_to_local(wi, predicted_normal, predicted_tangent)
        wo_local = self.world_to_local(wo, predicted_normal, predicted_tangent)
        local_normal = torch.zeros_like(wi_local)
        local_normal[..., 2] = 1.0  # Normal is always (0,0,1) in local space
        if self.neural_geometry:
            geometry_latent = latent[..., -6-self.geometry_latent_dim:-6]
            if self.local_wi_wo:
                if self.neural_geometry_pos_enc:
                    wi_local_enc = self.sh_encoder(wi_local)
                    wo_local_enc = self.sh_encoder(wo_local)
                    uv_offset = self.geometry_decoder(torch.cat([wi_local_enc, wo_local_enc], dim=-1))
                else:
                    uv_offset = self.geometry_decoder(torch.cat([geometry_latent, wi_local, wo_local], dim=-1))
            else:
                if self.neural_geometry_pos_enc:
                    wi_enc = self.sh_encoder(wi)
                    wo_enc = self.sh_encoder(wo)
                    uv_offset = self.geometry_decoder(torch.cat([wi_enc, wo_enc], dim=-1))
                else:
                    uv_offset = self.geometry_decoder(torch.cat([geometry_latent, wi, wo], dim=-1))
            uv = uv + uv_offset
            latent = self.sample_latent_from_texture(uv, tex)
            
            if self.recompute_frame: # recompute frame use new uv
                predicted_normal = latent[..., -6:-3]  # Last 6-3 dimensions for normal
                predicted_tangent = latent[..., -3:]   # Last 3 dimensions for tangent       
                predicted_normal = torch.nn.functional.normalize(predicted_normal, dim=-1)
                predicted_tangent = torch.nn.functional.normalize(predicted_tangent, dim=-1)
                predicted_tangent = predicted_tangent - torch.sum(predicted_tangent * predicted_normal, dim=-1, keepdim=True) * predicted_normal
                predicted_tangent = torch.nn.functional.normalize(predicted_tangent, dim=-1)   
                wi_local = self.world_to_local(wi, predicted_normal, predicted_tangent)
                wo_local = self.world_to_local(wo, predicted_normal, predicted_tangent)
                local_normal = torch.zeros_like(wi_local)
                local_normal[..., 2] = 1.0  # Normal is always (0,0,1) in local space
        # Split latent into three parts for RGB channels

        if self.pos_enc:
            wi_enc = self.sh_encoder(wi_local)
            wo_enc = self.sh_encoder(wo_local)
            normal_enc = self.sh_encoder(local_normal)
            enc_dir = torch.cat([wi_enc, wo_enc, normal_enc], dim=-1)
        else:
            enc_dir = torch.cat([wi_local, wo_local, local_normal], dim=-1)
        
        #print("latent_shape", latent.shape)
        if self.colorful_texture:
            if self.different_decoder:
                if self.larger_latent_dim:
                    latent_r = latent[..., :self.latent_dim]
                    latent_g = latent[..., self.latent_dim:2*self.latent_dim]
                    latent_b = latent[..., 2*self.latent_dim:3*self.latent_dim]
                else:
                    latent_r = latent[...,:self.latent_dim]
                    latent_g = latent[...,:self.latent_dim]
                    latent_b = latent[...,:self.latent_dim]
                    
                brdf_r = self.forward(enc_dir, latent_r, 'r')
                brdf_g = self.forward(enc_dir, latent_g, 'g')
                brdf_b = self.forward(enc_dir, latent_b, 'b')
                # Combine channels
                brdf = torch.cat([brdf_r, brdf_g, brdf_b], dim=-1)
            else:
                brdf = self.forward(enc_dir, latent[...,:self.latent_dim], None)
        else:
            brdf = self.forward(enc_dir, latent[...,:self.latent_dim], None)
            brdf = brdf.repeat(1,3)
        # # brdf = brdf * color
        pdf = NoL / math.pi
        brdf = brdf/10.0
        return brdf, pdf
    
    def sample_brdf(self, params, pos, sample1, sample2, wo, normal, latent=None, batch_mask=None):
        """
        Importance sampling BRDF using proxy (PBRBRDF) and evaluating MLP BRDF.
        
        Args:
            params: dictionary of parameters
            pos: Bx3 position
            sample1: B uniform samples [0,1] to select diffuse or specular sampling
            sample2: Bx2 uniform samples for hemisphere sampling
            wo: Bx3 viewing direction in world space
            normal: Bx3 normal in world space
            latent: optional latent code
            batch_mask: optional batch mask

        Returns:
            wi: Bx3 sampled incoming directions (world space)
            pdf: Bx1 sampling pdf values from proxy_brdf
            brdf_weight: Bx3 ratio (MLP evaluated BRDF / pdf)
        """
        # Sample direction using proxy BRDF
        wi_proxy, pdf_proxy = self.proxy_brdf.sample_brdf(pos, sample1, sample2, wo, normal, params['roughness'], batch_mask)
        
        stop_gradient_pdf_proxy = pdf_proxy.detach()
        mlp_brdf, _ = self.eval_brdf(params, pos, wi_proxy, wo, normal, latent, batch_mask)
        
        # Calculate weight (MLP BRDF / PDF)
        mlp_brdf = mlp_brdf * pdf_proxy / (stop_gradient_pdf_proxy + 1e-8)
        brdf_weight = torch.where(pdf_proxy > 0, mlp_brdf / (stop_gradient_pdf_proxy + 1e-8), torch.zeros_like(mlp_brdf))

        return wi_proxy, stop_gradient_pdf_proxy, brdf_weight 
