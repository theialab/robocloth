import torch
import torch.nn as nn
import torch.nn.functional as NF
import math
from pytorch_lightning import LightningModule
import sys
from brdf_plugin.utils.ops import *
from brdf_plugin.utils.cuda_manage import print_cuda_memory_info
import torch.nn.functional as F

class CompressedModel(LightningModule):
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
        latent_texture = torch.zeros(
                1, self.latent_dim*3+3+11, self.texture_resolution, self.texture_resolution
            )
        self.register_buffer("latent_texture", latent_texture)
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
        input_dim = 86
        
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 3)
        self.activation = nn.LeakyReLU(0.2)

    def update_latent_texture(self, new_texture: torch.Tensor):
        self.latent_texture.copy_(new_texture)

    def compute_uv(self, pos, width=0.4, length=0.4):
        half_w   = width  * 0.5               # 0.2
        half_l   = length * 0.5               # 0.2
        x, z     = pos[:, 0], pos[:, 2]

        u = (x + half_w) / width              # (-0.2→0,  +0.2→1)
        v = (z + half_l) / length             # (-0.2→0,  +0.2→1)

        return torch.stack([u, v], dim=-1)     # (N,2)

    def sample_latent_from_texture(self,pos, texture,uv):
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

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=self.fc5(x)
        x=self.activation(x)
        return x

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
        print("eval CompressedModel")
        tex = self.latent_texture       
        latent = self.sample_latent_from_texture(pos, tex,uv)
        #print("uv",uv.shape)
        normal = torch.nn.functional.normalize(latent[..., -6:-3], dim=-1)
        tangent = torch.nn.functional.normalize(latent[..., -3:], dim=-1)
        tangent = tangent - torch.sum(tangent * normal, dim=-1, keepdim=True) * normal
        tangent = torch.nn.functional.normalize(tangent, dim=-1)
            #normal = vector_transform(normal)
        #wo[:,1]=-wo[:,1]
        wi_local = self.world_to_local(wi, normal, tangent)
        wo_local = self.world_to_local(wo, normal, tangent)
        encoded_wo_list = []
        encoded_wi_list = []
        frequencies = [2**i for i in range(4)]  # [1, 2, 4, 8]
        
        for freq in frequencies:
            # Apply sine and cosine encoding for each frequency
            encoded_wo_list.append(torch.sin(freq * math.pi * wo_local))
            encoded_wo_list.append(torch.cos(freq * math.pi * wo_local))
            encoded_wi_list.append(torch.sin(freq * math.pi * wi_local))
            encoded_wi_list.append(torch.cos(freq * math.pi * wi_local))

        wo_encoded = torch.cat(encoded_wo_list, dim=1)
        wi_encoded = torch.cat(encoded_wi_list, dim=1)

        #x=torch.cat([latent[..., :3*self.latent_dim],wo_encoded,wi_encoded],dim=1)
        x=torch.cat([latent,wo_encoded,wi_encoded],dim=1)
        # Split latent into three parts for RGB channels
        brdf = self.forward(x)
        pdf = 1.0

        brdf=brdf/10.0
        return brdf, pdf

    def learn_brdf(self, wi, wo, latent):
        encoded_wo_list = []
        encoded_wi_list = []
        frequencies = [2**i for i in range(4)]  # [1, 2, 4, 8]
        
        for freq in frequencies:
            # Apply sine and cosine encoding for each frequency
            encoded_wo_list.append(torch.sin(freq * math.pi * wo))
            encoded_wo_list.append(torch.cos(freq * math.pi * wo))
            encoded_wi_list.append(torch.sin(freq * math.pi * wi))
            encoded_wi_list.append(torch.cos(freq * math.pi * wi))

        wo_encoded = torch.cat(encoded_wo_list, dim=1)
        wi_encoded = torch.cat(encoded_wi_list, dim=1)

        x=torch.cat([latent,wo_encoded,wi_encoded],dim=1)
        brdf = self.forward(x)
        return brdf
    
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