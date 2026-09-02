import torch
import torch.nn as nn
import torch.nn.functional as NF
import math
from pytorch_lightning import LightningModule
import sys
from brdf_plugin.utils.ops import *
from brdf_plugin.utils.cuda_manage import print_cuda_memory_info

class LatentTexturedModel(LightningModule):
    """ MLP-based BRDF class with 2D texture latent grids """
    def __init__(self, cfg):
        super().__init__()

        # Latent dimension from config
        self.latent_dim = cfg.latent_dim
        self.colorful_texture = cfg.colorful_texture
        self.larger_latent_dim = cfg.larger_latent_dim
        self.different_decoder = cfg.different_decoder
        self.use_gt_normal = cfg.use_gt_normal
        if self.colorful_texture and self.larger_latent_dim:
            total_latent_dim = self.latent_dim * 3
        else:
            total_latent_dim = self.latent_dim
        self.predict_normal = cfg.predict_normal
        if self.predict_normal:
            total_latent_dim = total_latent_dim + 3
        self.pbr_texture = None
        
        # Create 2D texture latent grids
        self.texture_resolution = getattr(cfg, 'texture_resolution', 256)
        self.latent_texture = nn.Parameter(
            torch.randn(1, total_latent_dim, self.texture_resolution, self.texture_resolution) * 0.1
        )
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
        
        # Build MLP layers
        if self.different_decoder:
            # Create separate MLPs for RGB channels
            def build_mlp():
                layers = []
                prev_dim = input_dim
                for hidden_dim in cfg.hidden_layers:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    if cfg.activation.lower() == "relu":
                        layers.append(nn.ReLU())
                    prev_dim = hidden_dim
                    
                layers.append(nn.Linear(prev_dim, cfg.output_channels))
                layers.append(nn.LeakyReLU(0.2))
                return nn.Sequential(*layers)
            
            self.mlp_r = build_mlp()
            self.mlp_g = build_mlp()
            self.mlp_b = build_mlp()
        else:
            # Single MLP for all channels
            layers = []
            prev_dim = input_dim
            for hidden_dim in cfg.hidden_layers:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if cfg.activation.lower() == "relu":
                    layers.append(nn.ReLU())
                prev_dim = hidden_dim
                
            layers.append(nn.Linear(prev_dim, cfg.output_channels))
            layers.append(nn.LeakyReLU(0.2))
            
            self.mlp = nn.Sequential(*layers)

        # Initialize proxy BRDF for importance sampling
        #self.proxy_brdf = ProxyPBRBRDF()  # Default roughness
        self.proxy_brdf = None

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
    
    # def sphere_to_uv(self, pos):
    #     """
    #     Convert 3D sphere surface positions to UV coordinates
    #     Args:
    #         pos: Bx3 positions on sphere surface
    #     Returns:
    #         uv: Bx2 UV coordinates in [0,1] range
    #     """
    #     # Normalize positions to ensure they're on unit sphere
    #     pos_norm = pos / (pos.norm(dim=-1, keepdim=True) + 1e-8)
        
    #     # Convert to spherical coordinates
    #     x, y, z = pos_norm[..., 0], pos_norm[..., 1], pos_norm[..., 2]
        
    #     # Calculate UV coordinates
    #     u = 0.5 + torch.atan2(x, z) / (2 * math.pi)
    #     v = 0.5 - torch.asin(torch.clamp(y, -1.0, 1.0)) / math.pi
        
    #     return torch.stack([u, v], dim=-1)

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
            pos: Bx3 positions on sphere surface
        Returns:
            latent: BxD latent codes
        """
        # Convert sphere positions to UV coordinates
        #uv = self.compute_uv(pos, 0.8, 0.8)  # Bx2
        
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

    def forward(self, pos, wi, wo, normal, latent=None, batch_mask=None, channel=None):
        """
        Evaluate BRDF using MLP with 2D texture latent encoding
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
        if self.pos_enc:
            wi_enc = self.sh_encoder(wi)
            wo_enc = self.sh_encoder(wo)
            normal_enc = self.sh_encoder(normal)
            x = torch.cat([wi_enc, wo_enc, normal_enc, latent], dim=-1)
        else:
            x = torch.cat([wi, wo, normal, latent], dim=-1)
            
        if self.different_decoder:
            if channel == 'r':
                return self.mlp_r(x)
            elif channel == 'g':
                return self.mlp_g(x)
            else:
                return self.mlp_b(x)
        else:
            return self.mlp(x)

    def world_to_local(self, v, normal):
        
        # choose arbitrary tangent
        up = torch.tensor([0.0, 1.0, 0.0], device=normal.device).expand_as(normal)
        tangent = torch.cross(up, normal)
        tangent_len = tangent.norm(dim=-1, keepdim=True)
        
        # if normal is collinear with [0,1,0], choose another tangent
        # collinear_mask = tangent_len.squeeze(-1) < 1e-6
        # if collinear_mask.any():
        #     tangent[collinear_mask] = torch.cross(normal[collinear_mask], torch.tensor([1., 0., 0.].expand_as(normal[collinear_mask])), device=normal.device)
        #     tangent_len = tangent.norm(dim=-1, keepdim=True)

        tangent = tangent / tangent_len

        bitangent = torch.cross(normal, tangent)

        v_local = torch.stack([
            (v * tangent).sum(dim=-1),
            (v * bitangent).sum(dim=-1),
            (v * normal).sum(dim=-1)
        ], dim=-1)

        return v_local
    
    def eval_brdf(self, gt_params, pos, wi, wo, normal,tangent,uv, latent=None, batch_mask=None):
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
        NoL=wi[:,2:3].repeat(1, 3)
        NoV=wo[:,2:3].repeat(1, 3)
        #NoL = (wi*normal).sum(-1,keepdim=True)
        #NoV = (wo*normal).sum(-1,keepdim=True)
        '''
        """ load gt color for reference """
        factor = 1.0
        params = self.pbr_texture
        H, W = params.shape[1], params.shape[2]
        crop_h = int((H - H * factor) // 2)
        crop_w = int((W - W * factor) // 2)
        new_h = int(H * factor)
        new_w = int(W * factor)
        params = params[:, crop_h:crop_h+new_h, crop_w:crop_w+new_w, :]
        uv = compute_uv(pos, 0.8, 0.8)

        # Step 2: Texture sampling
        arm, color, normal_local = sample_texture(params, uv)
        albedo, roughness, metallic = arm[:, 0:1], arm[:, 1:2], arm[:, 2:3]

        # Step 3: TBN frame
        T, B, N_geo = compute_tbn(pos, uv, 0.8, 0.8)

        # Step 4: Transform local normal to world
        n_world = local_to_world_normal(normal_local, T, B, N_geo)
        '''
        if self.training and self.Gaussian_blur:
            tex = self._blur_latent(self.global_step)
        else:
            tex = self.latent_texture       
        
        latent = self.sample_latent_from_texture(uv, tex)
        #print("uv",uv.shape)
        if self.use_gt_normal:
            normal = n_world
        if self.predict_normal:
            normal = torch.nn.functional.normalize(latent[..., -3:], dim=-1)
            #normal = vector_transform(normal)

        wi = wi[:, [0, 2, 1]]
        wi[:,2]=-wi[:,2]
        #wi[:,1]=-wi[:,1]

        wo = wo[:, [0, 2, 1]]
        wo[:,2]=-wo[:,2]
        #wo[:,1]=-wo[:,1]
        wi_local = self.world_to_local(wi, normal)
        wo_local = self.world_to_local(wo, normal)

        '''
        mask_x=wi[...,0]<0
        #print("neg_x",torch.sum(mask_x))
        mask_y=wi[...,1]<0
        #print("neg_y",torch.sum(mask_y))
        mask_z=wi[...,2]<0
        #print("neg_z",torch.sum(mask_z))

        mask_x=wi[...,0]>0
        #print("pos_x",torch.sum(mask_x))
        mask_y=wi[...,1]>0
        #print("pos_y",torch.sum(mask_y))
        mask_z=wi[...,2]>0
        #print("pos_z",torch.sum(mask_z))
        '''
        local_normal = torch.zeros_like(wi_local)
        local_normal[..., 2] = 1.0  # Normal is always (0,0,1) in local space

        # Split latent into three parts for RGB channels
        if self.colorful_texture:
            if self.larger_latent_dim:
                latent_r = latent[..., :self.latent_dim]
                latent_g = latent[..., self.latent_dim:2*self.latent_dim]
                latent_b = latent[..., 2*self.latent_dim:3*self.latent_dim]
            
                # Get BRDF value for each channel
                brdf_r = self.forward(pos, wi_local, wo_local, local_normal, latent_r, batch_mask, 'r')
                brdf_g = self.forward(pos, wi_local, wo_local, local_normal, latent_g, batch_mask, 'g')
                brdf_b = self.forward(pos, wi_local, wo_local, local_normal, latent_b, batch_mask, 'b')
                # Combine channels
                brdf = torch.cat([brdf_r, brdf_g, brdf_b], dim=-1)
            else:
                brdf = self.forward(pos, wi_local, wo_local, local_normal, latent, batch_mask)
        else:
            brdf = self.forward(pos, wi_local, wo_local, local_normal, latent, batch_mask)
            brdf = brdf.repeat(1,3)
        # # brdf = brdf * color
        pdf = NoL / math.pi

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