"""
Refactored Neural BRDF Classes
- Latent: Single latent code vector
- LatentTexture: 2D grid of latent codes with interpolation and blur
- NeuralGeometry: Neural UV offset prediction
- BRDFDecoder: MLP decoder for BRDF evaluation
- AnisotropicLatentTexturedModel: Combined material class with all APIs
"""

import torch
import torch.nn as nn
import torch.nn.functional as NF
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import math
# from nerfstudio.field_components import encoding
from utils.ops import components_from_spherical_harmonics, num_sh_bases, D_GGX, fresnelSchlick, G_Smith, G_Smith_aniso, D_GGX_aniso
from utils.checkpoint_io import load_checkpoint_file


# ============================================================================
# 1. LATENT CLASS - Single latent code vector
# ============================================================================
class Latent(nn.Module):
    """
    Single latent code vector.
    This is a conceptual class - in practice, latents are stored as tensors,
    but this provides a clean interface for operations on individual latents.
    """
    def __init__(self, dim: int, init_std: float = 0.1):
        super().__init__()
        self.dim = dim
        self.code = nn.Parameter(torch.randn(dim) * init_std)
    
    def get_code(self):
        """Get the latent code vector"""
        return self.code
    
    def set_code(self, code: torch.Tensor):
        """Set the latent code vector"""
        assert code.shape[-1] == self.dim, f"Code dimension mismatch: {code.shape[-1]} vs {self.dim}"
        self.code.data = code


# ============================================================================
# 2. LATENT TEXTURE CLASS - 2D grid of latent codes
# ============================================================================
class LatentTexture(nn.Module):
    """
    2D texture grid storing latent codes with interpolation and Gaussian blur.
    Each texel contains a latent code vector.
    """
    def __init__(
        self, 
        resolution: int,
        latent_dim: int,
        predict_frame: bool = False,
        init_std: float = 0.1,
        blur_config: dict = None
    ):
        """
        Args:
            resolution: Texture resolution (HxW)
            latent_dim: Dimension of each latent code
            predict_frame: Whether to include normal+tangent (6D) in latent
            init_std: Standard deviation for initialization
            blur_config: Dict with blur_sigma0 and blur_half_life
        """
        super().__init__()
        self.resolution = resolution
        self.latent_dim = latent_dim
        self.predict_frame = predict_frame
        
        # Initialize latent grid [1, latent_dim, H, W]
        latent_init = torch.randn(1, latent_dim, resolution, resolution) * init_std
        
        # Initialize frame components if needed
        if predict_frame:
            # Last 6 dimensions: normal (0,0,1) and tangent (0,1,0)
            latent_init[:, -6:-3, :, :] = torch.tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1)
            latent_init[:, -3:, :, :] = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1)
        
        self.params = nn.Parameter(latent_init)
        
        # Gaussian blur parameters
        if blur_config is not None:
            self.blur_sigma0 = blur_config.get('blur_sigma0', 8.0)
            self.blur_half_life = blur_config.get('blur_half_life', 3333)
        else:
            self.blur_sigma0 = 8.0
            self.blur_half_life = 3333
    
    def _gaussian_kernel(self, sigma: float, channels: int):
        """Generate Gaussian kernel for depth-wise convolution."""
        if sigma < 0.5:
            return None
        
        radius = int(math.ceil(3 * sigma))
        ksize = 2 * radius + 1
        grid = torch.arange(-radius, radius + 1,
                           dtype=self.params.dtype,
                           device=self.params.device)
        g1d = torch.exp(-0.5 * (grid / sigma) ** 2)
        g1d = g1d / g1d.sum()
        g2d = (g1d[:, None] * g1d[None, :]).expand(channels, 1, ksize, ksize)
        return g2d
    
    def apply_gaussian_blur(self, step: int):
        """
        Apply progressive Gaussian blur based on training step.
        σ(t) = σ₀ · 2^{-t/half_life}
        
        Args:
            step: Current training step
        
        Returns:
            Blurred latent texture [1, total_dim, H, W]
        """
        sigma = self.blur_sigma0 * (0.5 ** (step / self.blur_half_life))
        kernel = self._gaussian_kernel(sigma, self.params.shape[1])
        
        if kernel is None:
            return self.params
        
        pad = kernel.shape[-1] // 2
        return NF.conv2d(self.params, kernel, padding=pad, groups=self.params.shape[1])
    
    def query(self, uv: torch.Tensor, blur_step: int = None):
        """
        Query latent codes at UV coordinates with bilinear interpolation.
        
        Args:
            uv: [B, 2] UV coordinates in [0, 1]
            blur_step: If not None, apply Gaussian blur for this training step
        
        Returns:
            latent: [B, total_dim] sampled latent codes
        """
        # Apply blur if training step is provided
        if blur_step is not None:
            texture = self.apply_gaussian_blur(blur_step)
        else:
            texture = self.params
        
        # Convert UV to grid coordinates for F.grid_sample
        # grid_sample expects coordinates in [-1, 1]
        grid_coords = uv * 2.0 - 1.0  # [B, 2] -> [-1, 1]
        grid_coords = grid_coords.unsqueeze(1).unsqueeze(0)  # [1, 1, B, 2]
        
        # Bilinear sampling
        latent = NF.grid_sample(
            texture,  # [1, D, H, W]
            grid_coords,  # [1, 1, B, 2]
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # [1, D, 1, B]
        
        # Reshape to [B, D]
        latent = latent.squeeze(0).squeeze(-1).transpose(0, 1)
        
        return latent
    
    def get_full_texture(self):
        """Get the full latent texture grid"""
        return self.params
    
    def save(self, path: str):
        """Save latent texture to file"""
        torch.save(self.params.data, path)
    
    def load(self, path: str):
        """Load latent texture from file (fails closed: missing file or wrong shape raises)"""
        loaded = load_checkpoint_file(path, map_location=self.params.device)
        if not torch.is_tensor(loaded) or loaded.shape != self.params.shape:
            got = tuple(loaded.shape) if torch.is_tensor(loaded) else type(loaded).__name__
            raise RuntimeError(
                f"latent texture {path}: expected a tensor of shape {tuple(self.params.shape)}, got {got}")
        self.params.data = loaded


# ============================================================================
# 2b. MIPMAP LATENT TEXTURE CLASS - Multi-level latent texture pyramid
# ============================================================================


# ============================================================================
# 3. NEURAL GEOMETRY CLASS - UV offset prediction
# ============================================================================
class NeuralGeometry(nn.Module):
    """
    Neural network for predicting UV offsets based on viewing/lighting directions.
    This enables view-dependent displacement/parallax effects.
    """
    def __init__(
        self,
        cfg,
        geometry_latent_dim: int,
        use_local_wi_wo: bool = True,
        use_pos_enc: bool = True
    ):
        """
        Args:
            cfg: Configuration with hidden_layers, activation, output_channels
            geometry_latent_dim: Dimension of geometry-specific latent
            use_local_wi_wo: Use local space directions (vs world space)
            use_pos_enc: Use spherical harmonics encoding
        """
        super().__init__()
        self.geometry_latent_dim = geometry_latent_dim
        self.use_local_wi_wo = use_local_wi_wo
        self.use_pos_enc = use_pos_enc
        
        # Setup positional encoding if enabled
        if use_pos_enc:
            self.degree = 3
            self.sh_encoder = lambda x: components_from_spherical_harmonics(self.degree, x)
            sh_dim = num_sh_bases(self.degree)
            encoded_dim = sh_dim * 2  # wi and wo
        else:
            encoded_dim = 6  # wi (3) + wo (3)
        
        # Input: encoded directions (or geometry latent + directions)
        input_dim = encoded_dim + geometry_latent_dim if not use_pos_enc else encoded_dim
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        for hidden_dim in cfg.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if cfg.activation.lower() == "relu":
                layers.append(nn.ReLU())
            else:
                layers.append(nn.LeakyReLU(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, cfg.output_channels))  # 2D UV offset
        layers.append(nn.Tanh())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        wi: torch.Tensor,
        wo: torch.Tensor,
        geometry_latent: torch.Tensor = None
    ):
        """
        Predict UV offset based on directions.
        
        Args:
            wi: [B, 3] incoming light direction (local or world space)
            wo: [B, 3] outgoing view direction (local or world space)
            geometry_latent: [B, geometry_latent_dim] geometry latent code
        
        Returns:
            uv_offset: [B, 2] UV offset
        """
        if self.use_pos_enc:
            wi_enc = self.sh_encoder(wi)
            wo_enc = self.sh_encoder(wo)
            mlp_input = torch.cat([wi_enc, wo_enc], dim=-1)
        else:
            if geometry_latent is not None:
                mlp_input = torch.cat([geometry_latent, wi, wo], dim=-1)
            else:
                mlp_input = torch.cat([wi, wo], dim=-1)
        
        return self.mlp(mlp_input)


# ============================================================================
# 4. BRDF DECODER CLASS - MLP decoder
# ============================================================================
class BRDFDecoder(nn.Module):
    """
    MLP decoder that maps (encoded_directions + latent) -> BRDF value.
    With different_decoder=False: one MLP; output is cfg.output_channels (or [B,3]
    when use_color_decomp).
    With different_decoder=True: three independent MLPs (no shared trunk). Latent
    must be [B, 3 * latent_dim] — slices [:D], [D:2D], [2D:3D] feed R, G, B
    decoders respectively (each MLP sees encoded directions + its D-dim slice).

    Supports two additional conditioning modes (configurable via cfg):
      - FiLM conditioning (use_film): latent modulates hidden features via
        learned per-layer scale (gamma) and shift (beta) instead of being
        concatenated at the input.
      - Explicit color decomposition (use_color_decomp): latent is split into
        a color part (projected to base_color) and a shape part (used for
        angular response).  Output = base_color * angular_response.
    Both can be combined.  When both are False the decoder is identical to the
    original concatenation-based design.
    """
    def __init__(
        self,
        cfg,
        latent_dim: int,
        use_pos_enc: bool = True,
        different_decoder: bool = False
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_pos_enc = use_pos_enc
        self.different_decoder = different_decoder

        # Skip connection config
        self.use_skip_connection = getattr(cfg, 'use_skip_connection', False)
        self.skip_layer = getattr(cfg, 'skip_layer', None)

        # FiLM and color decomposition config
        self.use_film = getattr(cfg, 'use_film', False)
        self.use_color_decomp = getattr(cfg, 'use_color_decomp', False)
        self.color_latent_dim = getattr(cfg, 'color_latent_dim', 3)

        if self.use_color_decomp:
            assert self.color_latent_dim < latent_dim, \
                f"color_latent_dim ({self.color_latent_dim}) must be < latent_dim ({latent_dim})"
            self.shape_latent_dim = latent_dim - self.color_latent_dim
        else:
            self.shape_latent_dim = latent_dim

        # Setup positional encoding
        if use_pos_enc:
            self.degree = getattr(cfg, 'degree', 3)
            use_nerfstudio_sh = getattr(cfg, 'use_nerfstudio_sh', False)

            if use_nerfstudio_sh:
                self.sh_encoder = encoding.SHEncoding(levels=self.degree + 1)
                sh_dim = (self.degree + 1) ** 2
            else:
                self.sh_encoder = lambda x: components_from_spherical_harmonics(self.degree, x)
                sh_dim = num_sh_bases(self.degree)

            encoded_input_dim = sh_dim * 3  # wi, wo, normal
        else:
            encoded_input_dim = cfg.input_channels  # 9 (wi + wo + normal)

        self.encoded_input_dim = encoded_input_dim

        # MLP input dimension depends on conditioning mode
        if self.use_film:
            input_dim = encoded_input_dim
        else:
            input_dim = encoded_input_dim + self.shape_latent_dim

        self.input_dim = input_dim

        # Determine skip layer index (default to middle)
        num_hidden = len(cfg.hidden_layers)
        if self.skip_layer is None:
            self.skip_layer = num_hidden // 2

        # Output activation
        act = cfg.activation.lower()
        if act == "leakyrelu":
            output_activation = nn.LeakyReLU()
        elif act == "softplus":
            output_activation = nn.Softplus()
        else:
            output_activation = nn.ReLU()

        # Intermediate (hidden-layer) activation — defaults to "relu" for
        # backward compatibility with the original merl-branch weights.
        inter_act = getattr(cfg, 'intermediate_activation', 'relu').lower()
        if inter_act == "leakyrelu":
            intermediate_activation = nn.LeakyReLU()
        elif inter_act == "softplus":
            intermediate_activation = nn.Softplus()
        else:
            intermediate_activation = nn.ReLU()

        # MLP output channels: 1 when color_decomp provides the color
        mlp_output_channels = 1 if self.use_color_decomp else cfg.output_channels

        # ----- Color decomposition: latent_color -> base_color [B, 3] -----
        if self.use_color_decomp:
            self.color_proj = nn.Sequential(
                nn.Linear(self.color_latent_dim, 3),
                nn.Softplus(),
            )

        # ----- FiLM mappers: latent_shape -> (gamma, beta) per hidden layer -----
        if self.use_film:
            self.film_mappers = nn.ModuleList()
            for hidden_dim in cfg.hidden_layers:
                mapper = nn.Linear(self.shape_latent_dim, 2 * hidden_dim)
                nn.init.zeros_(mapper.weight)
                nn.init.zeros_(mapper.bias)
                mapper.bias.data[:hidden_dim] = 1.0  # gamma=1, beta=0
                self.film_mappers.append(mapper)

        # ----- Build MLP(s) -----
        def build_mlp():
            if self.use_film or self.use_skip_connection:
                layers = nn.ModuleList()
                prev_dim = input_dim
                for i, hidden_dim in enumerate(cfg.hidden_layers):
                    if self.use_skip_connection and i == self.skip_layer:
                        prev_dim = prev_dim + input_dim
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    prev_dim = hidden_dim
                layers.append(nn.Linear(prev_dim, mlp_output_channels))
                return layers
            else:
                layers = []
                prev_dim = input_dim
                for hidden_dim in cfg.hidden_layers:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(intermediate_activation)
                    prev_dim = hidden_dim
                layers.append(nn.Linear(prev_dim, mlp_output_channels))
                layers.append(output_activation)
                return nn.Sequential(*layers)

        # Store activation for manual forward passes (FiLM / skip connection)
        if self.use_film or self.use_skip_connection:
            self.activation = intermediate_activation
            self.output_activation = output_activation

        if different_decoder:
            assert not self.use_film and not self.use_color_decomp, \
                "FiLM / color_decomp not supported with different_decoder=True"
            assert not self.use_skip_connection, \
                "skip connection not supported with different_decoder=True"

            def build_channel_mlp():
                layers = []
                prev_dim = input_dim
                for hidden_dim in cfg.hidden_layers:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(intermediate_activation)
                    prev_dim = hidden_dim
                layers.append(nn.Linear(prev_dim, 1))
                layers.append(output_activation)
                return nn.Sequential(*layers)

            self.mlp_r = build_channel_mlp()
            self.mlp_g = build_channel_mlp()
            self.mlp_b = build_channel_mlp()
        else:
            self.mlp = build_mlp()
    
    def encode_directions(
        self,
        wi_local: torch.Tensor,
        wo_local: torch.Tensor,
        normal_local: torch.Tensor
    ):
        """
        Encode local-space directions with spherical harmonics.
        
        Args:
            wi_local: [B, 3] incoming light direction (local space)
            wo_local: [B, 3] outgoing view direction (local space)
            normal_local: [B, 3] normal (local space, typically [0,0,1])
        
        Returns:
            encoded: [B, encoded_dim] encoded directions
        """
        if self.use_pos_enc:
            wi_enc = self.sh_encoder(wi_local)
            wo_enc = self.sh_encoder(wo_local)
            normal_enc = self.sh_encoder(normal_local)
            return torch.cat([wi_enc, wo_enc, normal_enc], dim=-1)
        else:
            return torch.cat([wi_local, wo_local, normal_local], dim=-1)
    
    def _forward_with_skip(self, mlp_input: torch.Tensor, layers: nn.ModuleList):
        """Forward pass with skip connection for ModuleList-based MLP."""
        x = mlp_input
        num_layers = len(layers)
        
        for i, layer in enumerate(layers):
            # Inject skip connection at specified layer
            if i == self.skip_layer:
                x = torch.cat([x, mlp_input], dim=-1)
            
            x = layer(x)
            
            # Apply activation: ReLU for middle layers, output_activation for last layer
            if i < num_layers - 1:
                x = self.activation(x)
            else:
                x = self.output_activation(x)
        
        return x

    def _forward_with_film(self, mlp_input: torch.Tensor, layers: nn.ModuleList,
                           film_latent: torch.Tensor):
        """Forward pass with FiLM conditioning (and optional skip connection)."""
        x = mlp_input
        num_layers = len(layers)
        num_hidden = num_layers - 1

        for i in range(num_hidden):
            if self.use_skip_connection and i == self.skip_layer:
                x = torch.cat([x, mlp_input], dim=-1)
            x = layers[i](x)
            film_out = self.film_mappers[i](film_latent)
            gamma, beta = film_out.chunk(2, dim=-1)
            x = gamma * x + beta
            x = self.activation(x)

        x = layers[-1](x)
        x = self.output_activation(x)
        return x

    def forward(
        self,
        enc_dir: torch.Tensor,
        latent: torch.Tensor,
        channel: str = None
    ):
        """
        Decode BRDF from encoded directions and latent.

        Returns:
            brdf: [B, 3] when different_decoder=True or use_color_decomp=True,
                  else [B, output_channels]
            If different_decoder and channel is 'r'|'g'|'b', returns [B, 1] for
            that channel only (latent must still contain the full 3*latent_dim
            slice passed by the caller).
        """
        if self.use_color_decomp:
            latent_color = latent[:, :self.color_latent_dim]
            latent_shape = latent[:, self.color_latent_dim:]
            base_color = self.color_proj(latent_color)  # [B, 3]
        else:
            latent_shape = latent

        if self.use_film:
            mlp_input = enc_dir
        else:
            mlp_input = torch.cat([enc_dir, latent_shape], dim=-1)

        if self.different_decoder:
            d = self.latent_dim
            if latent.shape[-1] < 3 * d:
                raise ValueError(
                    f"different_decoder expects latent dim >= 3*latent_dim ({3*d}), "
                    f"got {latent.shape[-1]}"
                )
            zr = latent[:, :d]
            zg = latent[:, d : 2 * d]
            zb = latent[:, 2 * d : 3 * d]
            ir = torch.cat([enc_dir, zr], dim=-1)
            ig = torch.cat([enc_dir, zg], dim=-1)
            ib = torch.cat([enc_dir, zb], dim=-1)
            if channel is None:
                brdf_r = self.mlp_r(ir)
                brdf_g = self.mlp_g(ig)
                brdf_b = self.mlp_b(ib)
                return torch.cat([brdf_r, brdf_g, brdf_b], dim=-1)
            ch = channel.lower()
            if ch == 'r':
                return self.mlp_r(ir)
            if ch == 'g':
                return self.mlp_g(ig)
            if ch == 'b':
                return self.mlp_b(ib)
            raise ValueError(f"channel must be None, 'r', 'g', or 'b', got {channel!r}")

        if self.use_film:
            angular = self._forward_with_film(mlp_input, self.mlp, latent_shape)
        elif self.use_skip_connection:
            angular = self._forward_with_skip(mlp_input, self.mlp)
        else:
            angular = self.mlp(mlp_input)

        if self.use_color_decomp:
            return base_color * angular  # [B, 3] * [B, 1] -> [B, 3]

        return angular

class PBRDecoder(nn.Module):
    """
    PBR decoder that maps a single latent (material properties) + directions -> BRDF value.
    Works in canonical/local space where normal=(0,0,1), tangent=(0,1,0).
    Supports both isotropic and anisotropic BRDF models.
    
    Latent structure:
        Isotropic: [color(3), albedo(1), roughness(1), metallic(1)] = 6 channels
        Anisotropic: [diffuse(3), ao(1), roughness(1), metallic(1), ior(1), aniso_strength(1), aniso_rot(1)] = 9 channels
        Disney (anisotropic + Disney lobes):
            [baseColor(3), ao(1), roughness(1), metallic(1), ior(1), aniso_strength(1), aniso_rot(1),
             specularTint(1), sheen(1), sheenTint(1)] = 12 channels
    """
    def __init__(
        self,
        cfg,
        latent_dim: int = None,  # Not used, kept for API compatibility
        soft_constraint: bool = True
    ):
        """
        Args:
            cfg: Configuration with anisotropic flag
            latent_dim: Not used, kept for API compatibility
            soft_constraint: If True, use sigmoid for clamping; else use hard clamp
        """
        super().__init__()
        self.anisotropic = getattr(cfg, 'anisotropic', False)
        self.disney = getattr(cfg, 'disney', False)
        self.soft_constraint = soft_constraint
        
        # Canonical space basis vectors
        # Normal = (0, 0, 1), Tangent = (0, 1, 0), Bitangent = (1, 0, 0)
        self.register_buffer('canonical_normal', torch.tensor([0.0, 0.0, 1.0]))
        self.register_buffer('canonical_tangent', torch.tensor([0.0, 1.0, 0.0]))
        self.register_buffer('canonical_bitangent', torch.tensor([1.0, 0.0, 0.0]))
        
    def compute_svbrdf_pdf(self, albedo, roughness, metallic, wi, wo, normal):
        """
        Compute isotropic SVBRDF and PDF.
        
        Args:
            albedo: [N, 1] Albedo value
            roughness: [N, 1] Roughness
            metallic: [N, 1] Metallic factor
            wi: [N, 3] Incident direction (local space)
            wo: [N, 3] Outgoing direction (local space)
            normal: [N, 3] Normal (local space)
        Returns:
            brdf [N, 1], pdf [N, 1]
        """
        h = NF.normalize(wi + wo, dim=-1)
        NoL = (wi * normal).sum(-1, keepdim=True).relu()
        NoV = (wo * normal).sum(-1, keepdim=True).relu()
        VoH = (wo * h).sum(-1, keepdim=True).relu()
        NoH = (normal * h).sum(-1, keepdim=True).relu()

        D = D_GGX(NoH, roughness)
        pdf_spec = D.data / (4 * VoH.clamp_min(1e-4)) * NoH
        pdf_diff = NoL / math.pi
        pdf = 0.5 * pdf_spec + 0.5 * pdf_diff

        kd = albedo * (1 - metallic)
        ks = 0.04 * (1 - metallic) + albedo * metallic

        G = G_Smith(NoV, NoL, roughness)
        F = fresnelSchlick(VoH, ks)
        brdf_diff = kd / math.pi
        brdf_spec = D * G * F / 4.0

        brdf = brdf_diff + brdf_spec

        return brdf, pdf

    def compute_anisotropic_svbrdf_pdf(self,
                                       diffuse, ao, ax, ay, metallic, ior,
                                       wi, wo,
                                       normal, tangent):
        """
        Compute anisotropic SVBRDF and PDF.
        
        Args:
            diffuse: [N, 3] Diffuse (base-color)
            ao: [N, 1] Ambient Occlusion
            ax, ay: [N, 1] Directional roughness (tangent/bitangent)
            metallic: [N, 1] Metallic factor
            ior: [N, 1] Specular IOR
            wi, wo: [N, 3] Incident & outgoing directions (local space)
            normal: [N, 3] Surface normal (local space)
            tangent: [N, 3] Tangent vector (local space)
        Returns:
            brdf [N, 3], pdf [N, 1]
        """
        B = NF.normalize(torch.cross(normal, tangent, dim=-1), dim=-1)
        h = NF.normalize(wi + wo, dim=-1)

        NoL = (wi * normal).sum(-1, keepdim=True).clamp_min(0.0)
        NoV = (wo * normal).sum(-1, keepdim=True).clamp_min(0.0)
        VoH = (wo * h).sum(-1, keepdim=True).clamp_min(1e-4)
        NoH = (normal * h).sum(-1, keepdim=True).clamp_min(1e-4)

        # Specular distribution and geometry
        D = D_GGX_aniso(h, normal, tangent, B, ax, ay)
        G = G_Smith_aniso(wi, wo, normal, tangent, B, ax, ay)

        # Fresnel base reflectance
        F0_dielectric = ((ior - 1) / (ior + 1)).pow(2)
        F0 = F0_dielectric * (1 - metallic) + diffuse * metallic
        F = fresnelSchlick(VoH, F0)

        # Diffuse and Specular reflectances
        kd = diffuse * (1 - metallic) * ao  # apply AO only to diffuse
        ks = 1.0  # standard GGX specular strength

        # BRDF computation
        brdf_spec = ks * (D * G * F) / (4.0 * NoL * NoV + 1e-6)
        brdf_diff = kd / math.pi
        brdf = brdf_spec + brdf_diff

        # PDF (half diffuse, half specular)
        pdf_spec = D * NoH / (4.0 * VoH)
        pdf_diff = NoL / math.pi
        pdf = 0.5 * (pdf_spec + pdf_diff)

        return brdf, pdf

    def compute_disney_anisotropic_svbrdf_pdf(self,
                                              baseColor, ao, ax, ay, metallic, ior,
                                              specularTint, sheen, sheenTint,
                                              wi, wo,
                                              normal, tangent):
        """
        Disney Principled BRDF (anisotropic) with specularTint, Burley diffuse, and sheen.
        Reference: Burley 2012, WDAS BRDF Explorer disney.brdf

        Args:
            baseColor: [N, 3] Base color (linear space)
            ao: [N, 1] Ambient Occlusion
            ax, ay: [N, 1] Directional roughness (tangent/bitangent)
            metallic: [N, 1] Metallic factor
            ior: [N, 1] Specular IOR (controls dielectric F0 magnitude)
            specularTint: [N, 1] How much dielectric F0 takes baseColor hue (0=white, 1=tinted)
            sheen: [N, 1] Sheen strength (fabric edge glow)
            sheenTint: [N, 1] Sheen color (0=white, 1=baseColor tint)
            wi, wo: [N, 3] Incident & outgoing directions (local space)
            normal: [N, 3] Surface normal (local space)
            tangent: [N, 3] Tangent vector (local space)
        Returns:
            brdf [N, 3], pdf [N, 1]
        """
        B = NF.normalize(torch.cross(normal, tangent, dim=-1), dim=-1)
        h = NF.normalize(wi + wo, dim=-1)

        NoL = (wi * normal).sum(-1, keepdim=True).clamp_min(0.0)
        NoV = (wo * normal).sum(-1, keepdim=True).clamp_min(0.0)
        VoH = (wo * h).sum(-1, keepdim=True).clamp_min(1e-4)
        NoH = (normal * h).sum(-1, keepdim=True).clamp_min(1e-4)
        LdotH = (wi * h).sum(-1, keepdim=True).clamp_min(0.0)

        # --- Disney specularTint: colored dielectric F0 ---
        Cdlum = 0.3 * baseColor[:, 0:1] + 0.6 * baseColor[:, 1:2] + 0.1 * baseColor[:, 2:3]
        Ctint = baseColor / Cdlum.clamp(min=1e-6)

        F0_dielectric = ((ior - 1) / (ior + 1)).pow(2)
        Cspec0 = torch.lerp(
            F0_dielectric * torch.lerp(torch.ones_like(baseColor), Ctint, specularTint),
            baseColor,
            metallic
        )
        F = fresnelSchlick(VoH, Cspec0)

        # --- Specular: D * G * F (same NDF/G as anisotropic path) ---
        D = D_GGX_aniso(h, normal, tangent, B, ax, ay)
        G = G_Smith_aniso(wi, wo, normal, tangent, B, ax, ay)
        brdf_spec = (D * G * F) / (4.0 * NoL * NoV + 1e-6)

        # --- Burley diffuse (roughness-dependent retroreflection) ---
        FL = (1.0 - NoL).pow(5)
        FV = (1.0 - NoV).pow(5)
        roughness = (ax + ay) * 0.5  # average roughness for diffuse term
        Fd90 = 0.5 + 2.0 * LdotH * LdotH * roughness
        Fd = (1.0 + (Fd90 - 1.0) * FL) * (1.0 + (Fd90 - 1.0) * FV)

        kd = baseColor * (1.0 - metallic) * ao
        brdf_diff = kd / math.pi * Fd

        # --- Sheen lobe (fabric edge glow) ---
        FH = (1.0 - LdotH).pow(5)
        Csheen = torch.lerp(torch.ones_like(baseColor), Ctint, sheenTint)
        brdf_sheen = FH * sheen * Csheen * (1.0 - metallic)

        brdf = brdf_diff + brdf_sheen + brdf_spec

        # PDF (half diffuse, half specular) — same as anisotropic path
        pdf_spec = D * NoH / (4.0 * VoH)
        pdf_diff = NoL / math.pi
        pdf = 0.5 * (pdf_spec + pdf_diff)

        return brdf, pdf

    def forward(
        self,
        wi: torch.Tensor,
        wo: torch.Tensor,
        latent: torch.Tensor
    ):
        """
        Compute BRDF and PDF from directions and material latent.
        All inputs are in canonical/local space where normal=(0,0,1), tangent=(0,1,0).
        
        Args:
            wi: [N, 3] Incident light direction (local space)
            wo: [N, 3] Outgoing view direction (local space)
            latent: [N, C] Material properties latent
                Isotropic (C=6): [color(3), albedo(1), roughness(1), metallic(1)]
                Anisotropic (C=9): [diffuse(3), ao(1), roughness(1), metallic(1), ior(1), aniso_strength(1), aniso_rot(1)]
                Disney (C=12): [baseColor(3), ao(1), roughness(1), metallic(1), ior(1), aniso_strength(1), aniso_rot(1),
                                specularTint(1), sheen(1), sheenTint(1)]
        
        Returns:
            brdf: [N, 3] BRDF values
            pdf: [N, 1] PDF values
        """
        N = wi.shape[0]
        device = wi.device
        
        # Canonical basis vectors (expanded to batch size)
        normal = self.canonical_normal.expand(N, -1)       # [N, 3] = (0, 0, 1)
        tangent = self.canonical_tangent.expand(N, -1)     # [N, 3] = (0, 1, 0)
        
        # Check valid geometry
        NoL = (wi * normal).sum(-1, keepdim=True)
        NoV = (wo * normal).sum(-1, keepdim=True)
        valid_geometry = (NoL > 0) & (NoV > 0)
        
        if not valid_geometry.any():
            return torch.zeros_like(wi), torch.zeros(N, 1, device=device)
        
        if self.anisotropic or self.disney:
            # Parse anisotropic latent: [baseColor(3), ao(1), roughness(1), metallic(1), ior(1), aniso_strength(1), aniso_rot(1)]
            baseColor = latent[:, 0:3]
            ao = latent[:, 3:4]
            roughness = latent[:, 4:5]
            metallic = latent[:, 5:6]
            ior = latent[:, 6:7]
            aniso_strength = latent[:, 7:8]
            aniso_rot = latent[:, 8:9]
            
            # Apply constraints (shared between anisotropic and Disney)
            if self.soft_constraint:
                baseColor = torch.sigmoid(baseColor)
                ao = torch.sigmoid(ao)
                roughness = torch.sigmoid(roughness)
                metallic = torch.sigmoid(metallic)
                ior = 1.0 + torch.sigmoid(ior) * 1.5  # IOR range [1.0, 2.5]
                aniso_strength = torch.sigmoid(aniso_strength)
                aniso_rot = torch.sigmoid(aniso_rot)
            else:
                baseColor = torch.clamp(baseColor, 0.01, 0.99)
                ao = torch.clamp(ao, 0.01, 0.99)
                roughness = torch.clamp(roughness, 0.01, 0.99)
                metallic = torch.clamp(metallic, 0.01, 0.99)
                ior = torch.clamp(ior, 1.0, 2.5)
                aniso_strength = torch.clamp(aniso_strength, 0.0, 1.0)
                aniso_rot = torch.clamp(aniso_rot, 0.0, 1.0)
            
            # Compute ax and ay based on anisotropy strength
            ax = roughness * (1.0 + aniso_strength)
            ay = roughness * (1.0 - aniso_strength * 0.5)
            
            # Rotate tangent by aniso_rot in the tangent-bitangent plane
            rotation_angle = aniso_rot * 2.0 * torch.pi  # Convert [0,1] to [0, 2π]
            cos_theta = torch.cos(rotation_angle)
            sin_theta = torch.sin(rotation_angle)
            
            bitangent = self.canonical_bitangent.expand(N, -1)  # [N, 3] = (1, 0, 0)
            T_rot = cos_theta * tangent + sin_theta * bitangent
            T_rot = NF.normalize(T_rot, dim=-1)
            
            if self.disney:
                # Parse Disney-specific channels
                specularTint = latent[:, 9:10]
                sheen = latent[:, 10:11]
                sheenTint = latent[:, 11:12]
                
                if self.soft_constraint:
                    specularTint = torch.sigmoid(specularTint)
                    sheen = torch.sigmoid(sheen)
                    sheenTint = torch.sigmoid(sheenTint)
                else:
                    specularTint = torch.clamp(specularTint, 0.0, 1.0)
                    sheen = torch.clamp(sheen, 0.0, 1.0)
                    sheenTint = torch.clamp(sheenTint, 0.0, 1.0)
                
                brdf, pdf = self.compute_disney_anisotropic_svbrdf_pdf(
                    baseColor, ao, ax, ay, metallic, ior,
                    specularTint, sheen, sheenTint,
                    wi, wo, normal, T_rot
                )
            else:
                brdf, pdf = self.compute_anisotropic_svbrdf_pdf(
                    baseColor, ao, ax, ay, metallic, ior,
                    wi, wo, normal, T_rot
                )
            
        else:
            # Parse isotropic latent: [color(3), albedo(1), roughness(1), metallic(1)]
            # Matches original texture structure: color from [0:3], ARM from [3:6]
            color = latent[:, 0:3]
            albedo = latent[:, 3:4]
            roughness = latent[:, 4:5]
            metallic = latent[:, 5:6]
            
            # Apply constraints
            if self.soft_constraint:
                color = torch.sigmoid(color)
                albedo = torch.sigmoid(albedo)
                roughness = torch.sigmoid(roughness)
                metallic = torch.sigmoid(metallic)
            else:
                color = torch.clamp(color, 0.01, 0.99)
                albedo = torch.clamp(albedo, 0.01, 0.99)
                roughness = torch.clamp(roughness, 0.01, 0.99)
                metallic = torch.clamp(metallic, 0.01, 0.99)
            
            brdf, pdf = self.compute_svbrdf_pdf(
                albedo, roughness, metallic, wi, wo, normal
            )
            brdf = color * brdf  # Scale by color (same as original)
            
        return brdf, pdf

# ============================================================================
# 5. PROXY BRDF CLASS (for importance sampling)
# ============================================================================
class ProxyPBRBRDF(nn.Module):
    """
    Proxy PBR BRDF model for importance sampling.
    This is a placeholder - the actual implementation should come from your codebase.
    """
    def __init__(self):
        super().__init__()
    
    def sample_brdf(self, pos, sample1, sample2, wo, normal, roughness, batch_mask):
        """
        Sample directions from proxy BRDF distribution.
        
        Returns:
            wi: [B, 3] sampled directions
            pdf: [B, 1] probability density
        """
        # Placeholder - implement actual proxy BRDF sampling
        raise NotImplementedError("Implement proxy BRDF sampling from your codebase")


# ============================================================================
# 6. ANISOTROPIC LATENT TEXTURED MODEL - Combined material class
# ============================================================================
class AnisotropicLatentTexturedModel(LightningModule):
    """
    Complete material model combining all components.
    Provides the same API as the original class for compatibility.
    """
    def __init__(self, cfg):
        super().__init__()
        
        # Store configuration
        self.cfg = cfg
        self.latent_dim = cfg.latent_dim
        self.colorful_texture = cfg.colorful_texture
        self.larger_latent_dim = cfg.larger_latent_dim
        self.different_decoder = cfg.different_decoder
        self.predict_frame = cfg.predict_frame
        self.gt_frame = cfg.gt_frame
        self.anisotropic = True
        self.Gaussian_blur = cfg.Gaussian_blur
        self.learnable_factor = cfg.learnable_factor
        if self.learnable_factor:
            self.factor = nn.Parameter(torch.ones(3))

        self.mono_brdf = cfg.mono_brdf
        
        # Neural geometry settings
        self.neural_geometry_enabled = cfg.neural_geometry.enable
        self.geometry_latent_dim = cfg.neural_geometry.latent_dim if self.neural_geometry_enabled else 0
        self.local_wi_wo = cfg.neural_geometry.local_wi_wo if self.neural_geometry_enabled else False
        self.neural_geometry_pos_enc = cfg.neural_geometry.positional_encoding if self.neural_geometry_enabled else False
        self.recompute_frame = cfg.neural_geometry.recompute_frame if self.neural_geometry_enabled else False
        self.neural_geometry_factor = cfg.neural_geometry.factor if self.neural_geometry_enabled else 0.4
        # Calculate total latent dimension
        if self.colorful_texture and self.larger_latent_dim:
            brdf_latent_dim = self.latent_dim * 3
        elif self.different_decoder:
            brdf_latent_dim = self.latent_dim * 3
        else:
            brdf_latent_dim = self.latent_dim
        self.brdf_latent_dim = brdf_latent_dim

        # Add frame dimensions if predicting TBN
        total_latent_dim = brdf_latent_dim
        if self.predict_frame:
            total_latent_dim += 6  # normal (3) + tangent (3)
        
        # Add geometry latent dimensions
        if self.neural_geometry_enabled:
            total_latent_dim += self.geometry_latent_dim
        
        if self.mono_brdf:
            total_latent_dim += 3 # add three color channels
        # 1. Create LatentTexture
        self.texture_resolution = getattr(cfg, 'texture_resolution', 256)
        blur_config = {
            'blur_sigma0': getattr(cfg, 'blur_sigma0', 2.0),
            'blur_half_life': getattr(cfg, 'blur_half_life', 3333)
        } if self.Gaussian_blur else None

        self.latent_texture = LatentTexture(
            resolution=self.texture_resolution,
            latent_dim=total_latent_dim,
            predict_frame=self.predict_frame,
            init_std=0.1,
            blur_config=blur_config
        )
        
        # 2. Create BRDFDecoder
        self.decoder = BRDFDecoder(
            cfg=cfg.decoder,
            latent_dim=self.latent_dim,
            use_pos_enc=True,
            different_decoder=self.different_decoder
        )
        
        # 3. Create NeuralGeometry if enabled
        if self.neural_geometry_enabled:
            self.neural_geometry = NeuralGeometry(
                cfg=cfg.neural_geometry,
                geometry_latent_dim=self.geometry_latent_dim,
                use_local_wi_wo=self.local_wi_wo,
                use_pos_enc=self.neural_geometry_pos_enc
            )
        else:
            self.neural_geometry = None
        
        # 4. Proxy BRDF for importance sampling
        self.proxy_brdf = ProxyPBRBRDF()
        
        # 5. Latent bank mode (alternative to texture-based sampling)
        self.use_latent_bank = getattr(cfg, 'use_latent_bank', False)
        
        if self.use_latent_bank:
            observations_folder = cfg.observations_folder
            
            # Compute global point offset for this material
            self.global_point_offset = self._compute_global_point_offset(observations_folder)
            
            # Load point positions from observations
            self.point_positions = self._load_point_positions(observations_folder)
            num_points = self.point_positions.shape[0]
            print(f"[LatentBank] Loaded {num_points:,} point positions from {observations_folder}")
            
            # Build KD-tree for efficient nearest neighbor lookup
            from scipy.spatial import cKDTree
            self.kdtree = cKDTree(self.point_positions.numpy())
            print(f"[LatentBank] Built KD-tree for {num_points:,} points")
            
            # Latent bank will be loaded from checkpoint - don't initialize here
            self.point_latent_bank = None
            self.bank_latent_dim = brdf_latent_dim + (6 if self.predict_frame else 0)
            print(f"[LatentBank] Latent bank will be loaded from checkpoint (latent_dim={self.bank_latent_dim})")
    
    # ------------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------------
    def compute_uv(self, pos: torch.Tensor, width: float = 0.4, length: float = 0.4):
        """
        Compute UV coordinates from 3D positions.
        
        Args:
            pos: [B, 3] 3D positions
            width: Width of the surface
            length: Length of the surface
        
        Returns:
            uv: [B, 2] UV coordinates in [0, 1]
        """
        half_w = width * 0.5
        half_l = length * 0.5
        x, z = pos[:, 0], pos[:, 2]
        
        u = (x + half_w) / width
        v = (z + half_l) / length
        
        return torch.stack([u, v], dim=-1)
    
    def world_to_local(
        self,
        v: torch.Tensor,
        normal: torch.Tensor,
        tangent: torch.Tensor
    ):
        """
        Transform vector from world space to local tangent space.
        
        Args:
            v: [B, 3] vector in world space
            normal: [B, 3] normal in world space
            tangent: [B, 3] tangent in world space
        
        Returns:
            v_local: [B, 3] vector in local space
        """
        if tangent is None:
            # Generate arbitrary tangent perpendicular to normal
            tangent = torch.cross(
                normal,
                torch.tensor([0.0, 0.0, 1.0], device=normal.device).expand_as(normal)
            )
        
        tangent_len = tangent.norm(dim=-1, keepdim=True)
        tangent = tangent / (tangent_len + 1e-8)
        bitangent = torch.cross(normal, tangent)
        
        v_local = torch.stack([
            (v * tangent).sum(dim=-1),
            (v * bitangent).sum(dim=-1),
            (v * normal).sum(dim=-1)
        ], dim=-1)
        
        return v_local
    
    def extract_frame_from_latent(self, latent: torch.Tensor):
        """
        Extract and orthonormalize normal and tangent from latent code.
        
        Args:
            latent: [B, total_dim] latent code with last 6 dims as normal+tangent
        
        Returns:
            normal: [B, 3] normalized normal vector
            tangent: [B, 3] normalized tangent vector (orthogonal to normal)
        """
        predicted_normal = latent[..., -6:-3]
        predicted_tangent = latent[..., -3:]
        
        # Normalize
        predicted_normal = NF.normalize(predicted_normal, dim=-1)
        predicted_tangent = NF.normalize(predicted_tangent, dim=-1)
        
        # Gram-Schmidt orthogonalization
        predicted_tangent = predicted_tangent - \
            torch.sum(predicted_tangent * predicted_normal, dim=-1, keepdim=True) * predicted_normal
        predicted_tangent = NF.normalize(predicted_tangent, dim=-1)
        
        return predicted_normal, predicted_tangent
    
    def _sample_from_texture(self, uv: torch.Tensor, texture: torch.Tensor):
        """
        Sample latent from a given texture at UV coordinates.
        
        Args:
            uv: [B, 2] UV coordinates in [0, 1]
            texture: [1, D, H, W] texture to sample from
        
        Returns:
            latent: [B, D] sampled latent codes
        """
        # Convert UV to grid coordinates for F.grid_sample
        grid_coords = uv * 2.0 - 1.0  # [B, 2] -> [-1, 1]
        grid_coords = grid_coords.unsqueeze(1).unsqueeze(0)  # [1, 1, B, 2]
        
        # Bilinear sampling
        latent = NF.grid_sample(
            texture,  # [1, D, H, W]
            grid_coords,  # [1, 1, B, 2]
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # [1, D, 1, B]
        
        # Reshape to [B, D]
        latent = latent.squeeze(0).squeeze(-1).transpose(0, 1)
        
        return latent
    
    # ------------------------------------------------------------------------
    # Latent Bank Methods (for use_latent_bank mode)
    # ------------------------------------------------------------------------
    def _compute_global_point_offset(self, observations_folder: str) -> int:
        """
        Compute the global point offset for this material based on folder path.
        
        The observations folder path contains the material ID (e.g., 'data/0/observations/').
        The offset is the sum of num_points from all materials with ID < current material ID.
        
        Args:
            observations_folder: Path to observations folder (e.g., 'data/0/observations/')
        
        Returns:
            offset: int, global point offset for indexing into the latent bank
        """
        import json
        from pathlib import Path
        
        obs_folder = Path(observations_folder)
        material_folder = obs_folder.parent
        data_folder = material_folder.parent
        
        # Extract current material ID from folder name
        current_material_id = int(material_folder.name)
        
        # Find all material folders with ID < current material ID
        offset = 0
        for mat_id in range(current_material_id):
            mat_folder = data_folder / str(mat_id)
            metadata_path = mat_folder / "point_metadata.json"
            
            if not metadata_path.exists():
                print(f"[LatentBank] Warning: {metadata_path} not found, skipping material {mat_id}")
                continue
            
            try:
                with open(metadata_path, 'r') as f:
                    point_meta = json.load(f)
                num_points = point_meta['num_points']
                offset += num_points
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[LatentBank] Warning: Failed to read {metadata_path}: {e}, skipping material {mat_id}")
                continue
        
        print(f"[LatentBank] Material {current_material_id}: global point offset = {offset:,}")
        return offset
    
    def _load_point_positions(self, observations_folder: str) -> torch.Tensor:
        """
        Load point positions from point_positions.npz file.
        
        Args:
            observations_folder: Path to folder containing observation chunks
        
        Returns:
            point_positions: [num_unique_points, 3] tensor of unique point positions
        """
        import numpy as np
        from pathlib import Path
        
        obs_folder = Path(observations_folder)
        material_folder = obs_folder.parent
        positions_path = material_folder / "point_positions.npz"
        
        data = np.load(positions_path)
        point_positions = torch.from_numpy(data['positions']).float()
        print(f"[LatentBank] Loaded point positions: {point_positions.shape}")
        
        return point_positions
    
    def _find_nearest_point_ids(self, pos: torch.Tensor, k: int = 1) -> tuple:
        """
        Find K nearest point IDs for query positions using KD-tree.
        
        Args:
            pos: [B, 3] query positions
            k: number of nearest neighbors
        
        Returns:
            distances: [B, k] distances to nearest points
            point_ids: [B, k] tensor of nearest point IDs
        """
        # Query KD-tree (CPU operation)
        distances, point_ids = self.kdtree.query(pos.detach().cpu().numpy(), k=k)
        distances = torch.from_numpy(distances).to(pos.device).float()
        point_ids = torch.from_numpy(point_ids).to(pos.device)
        return distances, point_ids
    
    def _sample_from_latent_bank(self, pos: torch.Tensor, k: int = 8) -> torch.Tensor:
        """
        Sample latent codes from point bank using K-NN with inverse distance weighting.
        
        Args:
            pos: [B, 3] query positions
            k: number of nearest neighbors for interpolation
        
        Returns:
            latent: [B, latent_dim] interpolated latent codes
        """
        # Find K nearest point IDs and distances (local indices for this material)
        distances, local_point_ids = self._find_nearest_point_ids(pos, k=k)  # [B, k], [B, k]
        
        # Convert local point IDs to global point IDs for latent bank lookup
        global_point_ids = local_point_ids + self.global_point_offset
        
        # Lookup latents for all K neighbors using global IDs
        latents = self.point_latent_bank(global_point_ids)  # [B, k, latent_dim]
        
        # Inverse distance weighting
        eps = 1e-8
        weights = 1.0 / (distances + eps)  # [B, k]
        weights = weights / weights.sum(dim=-1, keepdim=True)  # normalize to sum to 1
        
        # Weighted sum of latents
        latent = (weights.unsqueeze(-1) * latents).sum(dim=1)  # [B, latent_dim]
        
        return latent
    
    # ------------------------------------------------------------------------
    # Main BRDF evaluation API
    # ------------------------------------------------------------------------
    def eval_brdf(
        self,
        gt_params,
        pos: torch.Tensor,
        wi: torch.Tensor,
        wo: torch.Tensor,
        normal: torch.Tensor,
        uv: torch.Tensor,
        TBN: torch.Tensor,
        latent=None,
        batch_mask=None,
        footprint_vis=None,
        dp_du=None,
        dp_dv=None
    ):
        """
        Evaluate BRDF at given geometry and directions.
        
        Args:
            gt_params: Ground truth parameters (optional)
            pos: [B, 3] 3D positions
            wi: [B, 3] incoming light directions (world space)
            wo: [B, 3] outgoing view directions (world space)
            normal: [B, 3] normals (world space)
            uv: [B, 2] UV coordinates
            TBN: [B, 3, 3] tangent-bitangent-normal frame
            latent: Ignored (latent is sampled internally)
            batch_mask: Batch mask for batched operations
            footprint_vis: Footprint for mipmap level selection
            dp_du, dp_dv: Ray differentials
        
        Returns:
            brdf: [B, 3] BRDF values
            pdf: [B, 1] probability density
            uv_offset: [B, 2] UV offset (if neural geometry enabled, else zeros)
        """
        NoL = (wi * normal).sum(-1, keepdim=True)
        NoV = (wo * normal).sum(-1, keepdim=True)
        
        # Initialize uv_offset (used in return, always needed)
        uv_offset = torch.zeros_like(uv)
        
        # =====================================================================
        # LATENT BANK MODE: Sample latent from point bank using position
        # =====================================================================
        if self.use_latent_bank:
            # Sample latent from latent bank using nearest neighbor lookup
            latent = self._sample_from_latent_bank(pos)
            
            # Extract frame from latent if predicting frame
            if self.predict_frame:
                predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)
            else:
                # Use geometry normal and tangent
                predicted_normal = normal
                predicted_tangent = TBN[:, :, 0] if TBN is not None else None
            
            # No neural geometry in latent bank mode
        
        # =====================================================================
        # TEXTURE MODE: Sample latent from texture using UV coordinates
        # =====================================================================
        else:
            # 1. Get the (optionally blurred) texture ONCE
            if self.training and self.Gaussian_blur:
                tex = self.latent_texture.apply_gaussian_blur(self.global_step)
            else:
                tex = self.latent_texture.params
            
            # 2. Sample latent from the texture
            latent = self._sample_from_texture(uv, tex)
            
            # 3. Extract frame from latent if predicting frame
            if self.predict_frame:
                predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)
            else:
                # Use geometry normal and tangent
                predicted_normal = normal
                predicted_tangent = TBN[:, :, 0] if TBN is not None else None
            
            # 4. Predict UV offset if neural geometry is enabled
            if self.neural_geometry_enabled:
                # Extract geometry latent
                geometry_latent = latent[..., -6-self.geometry_latent_dim:-6] if self.predict_frame else \
                                 latent[..., -self.geometry_latent_dim:]
                
                # Get directions for geometry network
                if self.local_wi_wo:
                    wi_for_geo = self.world_to_local(wi, predicted_normal, predicted_tangent)
                    wo_for_geo = self.world_to_local(wo, predicted_normal, predicted_tangent)
                else:
                    wi_for_geo = wi
                    wo_for_geo = wo
                
                # Predict UV offset
                uv_offset = self.neural_geometry(wi_for_geo, wo_for_geo, geometry_latent) * self.neural_geometry_factor
                uv = uv + uv_offset
                uv = ((uv%1)+1)%1
                # Sample from the SAME blurred texture
                latent = self._sample_from_texture(uv, tex)
                
                # Recompute frame if needed
                if self.recompute_frame and self.predict_frame:
                    predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)
        
        # =====================================================================
        # Common code path for both modes
        # =====================================================================
        # 5. Transform directions to local space
        wi_local = self.world_to_local(wi, predicted_normal, predicted_tangent)
        wo_local = self.world_to_local(wo, predicted_normal, predicted_tangent)
        local_normal = torch.zeros_like(wi_local)
        local_normal[..., 2] = 1.0  # (0, 0, 1) in local space
        
        # 5. Encode directions
        enc_dir = self.decoder.encode_directions(wi_local, wo_local, local_normal)     
        
        # 6. Extract BRDF latent and decode
        if self.different_decoder:
            brdf = self.decoder(enc_dir, latent[..., :self.brdf_latent_dim])
        elif self.colorful_texture:
            brdf = self.decoder(enc_dir, latent[..., :self.latent_dim])
        else:
            brdf = self.decoder(enc_dir, latent[..., :self.latent_dim])
            brdf = brdf.repeat(1, 3)  # Replicate to RGB
        
        # 7. Calculate PDF (cosine-weighted)
        pdf = NoL / math.pi
        if self.learnable_factor:
            brdf = brdf * self.factor
        
        return brdf, predicted_normal, pdf, uv_offset
    
    def sample_brdf(
        self,
        params,
        pos: torch.Tensor,
        sample1: torch.Tensor,
        sample2: torch.Tensor,
        wo: torch.Tensor,
        normal: torch.Tensor,
        latent=None,
        batch_mask=None
    ):
        """
        Importance sample BRDF using proxy BRDF and evaluate with neural BRDF.
        
        Args:
            params: Dictionary with 'roughness' for proxy BRDF
            pos: [B, 3] 3D positions
            sample1: [B] uniform samples [0,1] for diffuse/specular choice
            sample2: [B, 2] uniform samples for hemisphere sampling
            wo: [B, 3] outgoing view directions (world space)
            normal: [B, 3] normals (world space)
            latent: Ignored (not used)
            batch_mask: Batch mask
        
        Returns:
            wi: [B, 3] sampled incoming directions (world space)
            pdf: [B, 1] sampling probability from proxy BRDF
            brdf_weight: [B, 3] BRDF/PDF ratio
        """
        # Sample direction from proxy BRDF
        wi_proxy, pdf_proxy = self.proxy_brdf.sample_brdf(
            pos, sample1, sample2, wo, normal,
            params['roughness'], batch_mask
        )
        
        # Evaluate neural BRDF at sampled direction
        # Note: Need to compute UV and TBN here
        uv = self.compute_uv(pos)
        TBN = None  # Compute TBN if needed
        
        mlp_brdf, _, _ = self.eval_brdf(
            params, pos, wi_proxy, wo, normal, uv, TBN,
            latent, batch_mask
        )
        
        # Calculate importance weight
        stop_gradient_pdf_proxy = pdf_proxy.detach()
        brdf_weight = torch.where(
            pdf_proxy > 0,
            mlp_brdf / (stop_gradient_pdf_proxy + 1e-8),
            torch.zeros_like(mlp_brdf)
        )
        
        return wi_proxy, stop_gradient_pdf_proxy, brdf_weight
    
    # ------------------------------------------------------------------------
    # Save/Load functionality
    # ------------------------------------------------------------------------
    def save_latent(self, path: str):
        """Save latent texture to file"""
        self.latent_texture.save(path)
    
    def load_latent(self, path: str):
        """Load latent texture from file"""
        self.latent_texture.load(path)

# ============================================================================
# 7. MULTI-MATERIAL LATENT BRDF - Auto-decoder for multiple materials
# ============================================================================
class MultiMaterialLatentBRDF(LightningModule):
    """
    Multi-material BRDF model using auto-decoder architecture.
    - Per-material latent codes (M materials)
    - Per-point latent codes (sum of points across all materials)
    - Shared MLP decoder across all materials
    
    Expected folder structure:
        data_folder/
            0/                      # material_id = 0
                point_metadata.json # contains {"num_points": N, ...}
            1/                      # material_id = 1
                point_metadata.json
            ...
    """
    def __init__(self, cfg):
        super().__init__()
        
        # Store configuration
        self.cfg = cfg
        data_folder = getattr(cfg, 'data_folder', None)
        
        # Read training list from txt file
        self.training_list_path = getattr(cfg, 'training_list_path', None)

        # Optimizer / sparse-embedding config (mirrors BonnLatentBRDF)
        self.optimizer_name = getattr(cfg, 'optimizer', {}).get('name', 'SparseAdam')
        self.use_sparse_adam = (self.optimizer_name in ('SparseAdam', 'SparseAdam8bit'))

        # Latent dimensions
        self.latent_dim = cfg.latent_dim
        self.predict_frame = cfg.predict_frame
        self.different_decoder = cfg.different_decoder
        self.brdf_latent_dim = self.latent_dim * 3 if self.different_decoder else self.latent_dim
        self.total_latent_dim = self.brdf_latent_dim + (6 if self.predict_frame else 0)
        # BRDF decoder settings
        self.use_pos_enc = cfg.use_pos_enc

        # Point subsampling ratio — must match cfg.data.point_subsample_ratio used
        # by MultiMaterialDenseDataset so the latent bank size aligns with the
        # point_ids emitted by the dataloader. Wired via Hydra interpolation in
        # config/material/multi_material_latent.yaml.
        self.point_subsample_ratio = float(getattr(cfg, 'point_subsample_ratio', 1.0))

        # Load point metadata from material subfolders
        print(f"Loading point metadata from {data_folder}...")
        self.metadata = self._load_point_metadata(data_folder)
        
        num_materials = self.metadata['num_materials']
        total_points = self.metadata['total_points']
        
        print(f"Loaded {num_materials} materials with {total_points:,} total points")

        is_sparse = self.use_sparse_adam
        if not is_sparse:
            print("Dense Adam or SGD selected. Setting point_latent_bank sparse=False.")

        self.point_latent_bank = nn.Embedding(
            num_embeddings=total_points,
            embedding_dim=self.total_latent_dim,
            sparse=is_sparse,
        )
        nn.init.normal_(self.point_latent_bank.weight, mean=0.0, std=cfg.init_std)
        
        if self.predict_frame:
            with torch.no_grad():
                self.point_latent_bank.weight[:, -6:-3] = torch.tensor([0.0, 0.0, 1.0])
                self.point_latent_bank.weight[:, -3:] = torch.tensor([0.0, 1.0, 0.0])
        
        # Shared BRDF decoder
        self.decoder = BRDFDecoder(
            cfg=cfg.decoder,
            latent_dim=self.latent_dim,
            use_pos_enc=self.use_pos_enc,
            different_decoder=self.different_decoder
        )
        
        # L2 gradient smoothness regularization on BRDF lobe
        self.smooth_reg = getattr(cfg.decoder, 'smooth_reg', False)
        self.smooth_reg_eps = getattr(cfg.decoder, 'smooth_reg_eps', 0.01)
        
        print("Initialization complete!")
    
    def _load_point_metadata(self, data_folder):
        """
        Load point metadata from material subfolders.
        
        Expected structure:
        data_folder/
            0/                      # material_id = 0
                point_metadata.json # contains {"num_points": N, "num_observations": M, ...}
            1/                      # material_id = 1
                point_metadata.json
            ...
        
        Returns:
            metadata: dict with keys:
                - num_materials: int
                - total_points: int (sum across all materials)
                - materials: List[dict] with material info including point_range
                - material_point_offsets: dict mapping material_id -> global point offset
        """
        import json
        import os
        from pathlib import Path

        root = Path(data_folder)
        
        # Build material folders from training list
        training_list = []
        with open(self.training_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    training_list.append(int(line))

        # Optional: "pretend the bad↔backup material swap never happened" — for
        # each swapped slot in training_list, read num_points from the swap
        # partner's folder (which physically holds this slot's pre-swap data).
        # Keeps latent_bank shape and per-material offsets identical to a saved
        # pre-swap checkpoint. Mirror of MultiMaterialDenseDataset's flag.
        swap_partner = {}
        legacy_swap_indexing = bool(getattr(self.cfg, 'legacy_swap_indexing', False))
        if legacy_swap_indexing:
            rl_path = getattr(self.cfg, 'replace_list_path', None) or str(root / 'replace_list.json')
            if os.path.exists(rl_path):
                with open(rl_path) as _rl_f:
                    _rl = json.load(_rl_f)
                for _r in _rl.get('records', []):
                    if _r.get('backup_id') is None or _r.get('replaces') is None:
                        continue
                    swap_partner[int(_r['backup_id'])] = int(_r['replaces'])
                    swap_partner[int(_r['replaces'])] = int(_r['backup_id'])
                _affected = sum(1 for m in training_list if m in swap_partner)
                print(f"[legacy_swap_indexing] enabled: {_affected}/{len(training_list)} training-list slots remap to partner folder ({rl_path})")
            else:
                print(f"[legacy_swap_indexing] enabled but replace_list.json not found at {rl_path}; no remap applied")

        # When a slot is swap-paired AND legacy mode is on, read metadata from
        # the partner's folder. material_id (used for latent indexing) stays as
        # the training_list slot id — see the loop below.
        material_folders = [root / str(swap_partner.get(mid, mid)) for mid in training_list]
        materials = []
        material_point_offsets = {}
        global_point_offset = 0

        print(f"Training list path: {self.training_list_path}")
        print(f"Loaded {len(training_list)} materials from training list: {training_list}")

        for _i_mat, mat_folder in enumerate(material_folders):
            # Use training_list slot id (NOT folder.name) so swap remapping
            # doesn't change the material_id used by point_latent_bank.
            material_id = training_list[_i_mat]
            metadata_path = mat_folder / "point_metadata.json"
            
            # Skip if metadata file doesn't exist or isn't readable
            if not metadata_path.exists():
                print(f"  Warning: {metadata_path} not found, skipping material {material_id}")
                continue
            
            try:
                with open(metadata_path, 'r') as f:
                    point_meta = json.load(f)

                num_points = point_meta['num_points']
                num_observations = point_meta.get('num_observations', 0)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Warning: Failed to read {metadata_path}: {e}, skipping material {material_id}")
                continue

            # Apply point subsampling: mirrors MultiMaterialDenseDataset, which
            # keeps max(1, int(len(valid_v) * ratio)) points per material. When
            # cfg.data.filter_observations is False (default for points_dense)
            # len(valid_v) == num_points and the two counts agree exactly. If
            # XY filtering is ever enabled, the dataloader will emit fewer
            # point_ids than the bank allocates here (wasted capacity, no crash).
            if self.point_subsample_ratio < 1.0:
                num_points = max(1, int(num_points * self.point_subsample_ratio))
                num_observations = int(num_observations * self.point_subsample_ratio)

            # Store material info
            materials.append({
                'material_id': material_id,
                'name': mat_folder.name,
                'num_points': num_points,
                'num_observations': num_observations,
                'point_range': (global_point_offset, global_point_offset + num_points),
                'folder': str(mat_folder)
            })

            material_point_offsets[material_id] = global_point_offset
            global_point_offset += num_points

            print(f"  Material {material_id}: {num_points:,} points, {num_observations:,} observations")
        
        if len(materials) == 0:
            raise ValueError(f"No valid material folders found in {data_folder}")
        
        metadata = {
            'num_materials': len(materials),
            'total_points': global_point_offset,
            'materials': materials,
            'material_point_offsets': material_point_offsets,
        }
        
        # Build offset tensor for efficient indexing
        # offset_tensor[material_id] = global point offset for that material
        # Use max material ID + 1 to handle non-contiguous material IDs (e.g., 0, 1, 3, 5)
        max_mat_id = max(mat_info['material_id'] for mat_info in materials)
        offset_tensor = torch.zeros(max_mat_id + 1, dtype=torch.long)
        for mat_info in materials:
            mat_id = mat_info['material_id']
            offset_tensor[mat_id] = material_point_offsets[mat_id]
        
        # Register as buffer so it moves with the model to GPU
        self.register_buffer('material_offset_tensor', offset_tensor)
        
        return metadata
    
    def get_global_point_id(self, material_id, local_point_id):
        """
        Convert local point ID (per-material) to global point ID.
        
        Args:
            material_id: (1, N) or int - material indices
            local_point_id: (1, N) or int - local point indices within material
        
        Returns:
            global_point_id: (1, N) or int - global point indices for latent bank lookup
        """
        # Use pre-computed offset tensor for vectorized lookup
        offsets = self.material_offset_tensor[material_id]
        return local_point_id + offsets

    def extract_frame_from_latent(self, latent: torch.Tensor):
        """
        Extract and orthonormalize normal and tangent from latent code.
        
        Args:
            latent: [B, total_dim] latent code with last 6 dims as normal+tangent
        
        Returns:
            normal: [B, 3] normalized normal vector
            tangent: [B, 3] normalized tangent vector (orthogonal to normal)
        """
        predicted_normal = latent[..., -6:-3]
        predicted_tangent = latent[..., -3:]
        
        # Normalize
        predicted_normal = NF.normalize(predicted_normal, dim=-1)
        predicted_tangent = NF.normalize(predicted_tangent, dim=-1)
        
        # Gram-Schmidt orthogonalization
        predicted_tangent = predicted_tangent - \
            torch.sum(predicted_tangent * predicted_normal, dim=-1, keepdim=True) * predicted_normal
        predicted_tangent = NF.normalize(predicted_tangent, dim=-1)
        if torch.isnan(predicted_tangent).any():
            print("predicted_tangent is nan")
        
        return predicted_normal, predicted_tangent
    
    def world_to_local(self, v, normal, tangent=None):
        """
        Transform vector from world space to local tangent space.
        
        Args:
            v: [B, 3] vector in world space
            normal: [B, 3] normal in world space
            tangent: [B, 3] tangent in world space (optional)
        
        Returns:
            v_local: [B, 3] vector in local space
        """
        if tangent is None:
            # Generate arbitrary tangent perpendicular to normal
            up = torch.tensor([0.0, 1.0, 0.0], device=normal.device).expand_as(normal)
            tangent = torch.cross(up, normal)
            tangent_len = tangent.norm(dim=-1, keepdim=True)
            
            # Handle collinear case
            collinear_mask = tangent_len.squeeze(-1) < 1e-6
            if collinear_mask.any():
                right = torch.tensor([1.0, 0.0, 0.0], device=normal.device).expand_as(normal)
                tangent[collinear_mask] = torch.cross(right[collinear_mask], normal[collinear_mask])
                tangent_len = tangent.norm(dim=-1, keepdim=True)
        else:
            tangent_len = tangent.norm(dim=-1, keepdim=True)
        
        tangent = tangent / (tangent_len + 1e-8)
        bitangent = torch.cross(normal, tangent)
        
        v_local = torch.stack([
            (v * tangent).sum(dim=-1),
            (v * bitangent).sum(dim=-1),
            (v * normal).sum(dim=-1)
        ], dim=-1)
        
        return v_local
    
    def eval_brdf(
        self,
        pos,
        wi,
        wo,
        normal,
        latent=None,
        point_ids=None,
        material_ids=None,
    ):
        """
        Evaluate BRDF at given geometry and directions.
        
        Args:
            pos: [B, 3] 3D positions
            wi: [B, 3] incoming light directions (world space)
            wo: [B, 3] outgoing view directions (world space)
            normal: [B, 3] normals (world space)
            latent: Ignored (latents retrieved from banks)
            point_ids: [B] LOCAL point indices (per-material, from dataloader)
            material_ids: [B] material indices (required for global point ID computation)
        
        Returns:
            brdf: [B, 3] BRDF values
            normal: [B, 3] normals (local space)
            pdf: [B, 1] probability density
        """
        
        # point_ids and material_ids should be provided by the dataloader
        if point_ids is None or material_ids is None:
            raise ValueError("point_ids and material_ids must be provided (from dataloader)")
        
        # Convert local point IDs to global point IDs
        global_point_ids = self.get_global_point_id(material_ids, point_ids)
        
        # Retrieve latents from banks
        latent = self.point_latent_bank(global_point_ids)        # [B, latent_dim]
  
        if self.predict_frame:
            predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)
        NoL = (wi * predicted_normal).sum(-1, keepdim=True)
        NoV = (wo * predicted_normal).sum(-1, keepdim=True)
        wi_local = self.world_to_local(wi, predicted_normal, predicted_tangent)
        wo_local = self.world_to_local(wo, predicted_normal, predicted_tangent)
        normal_local = torch.zeros_like(wi_local)
        normal_local[..., 2] = 1.0  # Normal is always (0,0,1) in local space
        
        # Encode directions
        enc_dir = self.decoder.encode_directions(wi_local, wo_local, normal_local)

        # Decode BRDF
        brdf_lat = latent[:, : self.brdf_latent_dim]
        brdf = self.decoder(enc_dir, brdf_lat)
        if not self.different_decoder and brdf.shape[-1] == 1:
            brdf = brdf.expand(-1, 3)  # Expand to RGB

        # L2 gradient smoothness regularization via geodesic finite differences
        if self.smooth_reg:
            eps = self.smooth_reg_eps
            # Random axis perpendicular to wi_local (tangent plane of S²)
            rand_vec = torch.randn_like(wi_local)
            rand_vec = rand_vec - (rand_vec * wi_local).sum(-1, keepdim=True) * wi_local
            axis = NF.normalize(rand_vec, dim=-1)
            # Geodesic perturbation via Rodrigues' rotation
            wi_perturbed = wi_local * math.cos(eps) + torch.cross(axis, wi_local, dim=-1) * math.sin(eps)
            # Evaluate BRDF at perturbed direction
            enc_pert = self.decoder.encode_directions(wi_perturbed, wo_local, normal_local)
            brdf_pert = self.decoder(enc_pert, brdf_lat)
            if not self.different_decoder and brdf_pert.shape[-1] == 1:
                brdf_pert = brdf_pert.expand(-1, 3)
            # L2 squared gradient: || (f(wi+eps) - f(wi)) / eps ||^2
            smooth_loss = ((brdf_pert - brdf) / eps).pow(2).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=wi.device)
        
        # Simple diffuse PDF (can be improved with importance sampling)
        pdf = NoL.clamp(min=0) / math.pi
        if torch.isnan(brdf).any():
            print("brdf is nan")
        if torch.isnan(predicted_normal).any():
            print("normal is nan")
        return brdf, predicted_normal, pdf, smooth_loss
    
    def sample_brdf(
        self,
        params,
        pos,
        sample1,
        sample2,
        wo,
        normal,
        latent=None,
        batch_mask=None,
        point_ids=None,
        material_ids=None
    ):
        """
        Sample BRDF using importance sampling.
        
        Args:
            params: Ground truth parameters (not used)
            pos: [B, 3] 3D positions
            sample1: [B] uniform samples [0,1]
            sample2: [B, 2] uniform samples [0,1]^2
            wo: [B, 3] outgoing view directions (world space)
            normal: [B, 3] normals (world space)
            latent: Ignored
            batch_mask: Batch mask
            point_ids: [B] LOCAL point indices (per-material, from dataloader)
            material_ids: [B] material indices
        
        Returns:
            wi: [B, 3] sampled incoming light directions
            pdf: [B, 1] probability density
            brdf_weight: [B, 3] BRDF / pdf
        """
        # Simple cosine-weighted hemisphere sampling (can use proxy BRDF)
        # This is a placeholder - implement proper importance sampling if needed
        
        # Cosine-weighted sampling
        theta = torch.asin(torch.sqrt(sample2[..., 0]))
        phi = 2 * math.pi * sample2[..., 1]
        
        # Local space directions
        wi_local = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ], dim=-1)
        
        # Transform to world space
        # Build TBN frame
        up = torch.tensor([0.0, 1.0, 0.0], device=normal.device).expand_as(normal)
        tangent = torch.cross(up, normal)
        tangent_len = tangent.norm(dim=-1, keepdim=True)
        tangent = tangent / (tangent_len + 1e-8)
        bitangent = torch.cross(normal, tangent)
        
        # Local to world
        wi = (wi_local[..., 0:1] * tangent + 
              wi_local[..., 1:2] * bitangent + 
              wi_local[..., 2:3] * normal)
        
        # Evaluate BRDF at sampled direction
        brdf, pdf, _ = self.eval_brdf(
            None, pos, wi, wo, normal,
            point_ids=point_ids,
            material_ids=material_ids
        )
        
        # BRDF weight = BRDF / pdf (for Monte Carlo integration)
        brdf_weight = brdf / pdf.clamp(min=1e-6)

        return wi, pdf, brdf_weight


# ============================================================================
# MULTI-MATERIAL PBR LATENT BRDF — PBR sanity-check sibling of
# MultiMaterialLatentBRDF. Identical metadata loader (training_list +
# point_metadata.json per material) but swaps the learned BRDFDecoder for an
# analytical PBRDecoder. Latents now directly parameterize PBR channels.
# ============================================================================


# 7. BONN LATENT BRDF - Auto-decoder for Bonn SVBRDF dataset
# ============================================================================
class BonnLatentBRDF(LightningModule):
    """BRDF model for the Bonn SVBRDF dataset (UBOFAB19).

    Same architecture as MultiMaterialLatentBRDF:
    - Per-point latent codes (sum of H*W across all materials)
    - Shared MLP decoder across all materials
    - Optional predicted normal + tangent frame from latent

    Metadata is loaded from a single ``bonn_point_metadata.json`` in the
    data folder (no per-material subfolders or training list needed).

    Generate the metadata file with::

        python scripts/comparisons/generate_bonn_metadata.py /path/to/Bonn_train
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        data_folder = getattr(cfg, 'data_folder', None)

        self.latent_dim = cfg.latent_dim
        self.predict_frame = cfg.predict_frame
        self.different_decoder = cfg.different_decoder
        self.brdf_latent_dim = self.latent_dim * 3 if self.different_decoder else self.latent_dim
        self.total_latent_dim = self.brdf_latent_dim + 6  # always reserve 6 dims: [-6:-3] normal, [-3:] tangent
        self.use_pos_enc = cfg.use_pos_enc

        self.learnable_factor = getattr(cfg, 'learnable_factor', False)
        if self.learnable_factor:
            self.factor = nn.Parameter(torch.ones(3))

        # Per-material point subsampling ratio. Must match cfg.data.point_subsample_ratio
        # so the bank size lines up with the dataloader's compacted point_ids
        # (BonnDataset / BonnValDataset emit dense 0..N_sub-1 per material).
        # Stage-2 single-material mode keeps the full bank — subsampling is
        # only meaningful for multi-material stage-1 training.
        self.point_subsample_ratio = float(getattr(cfg, 'point_subsample_ratio', 1.0))

        # Single-material mode: load only one material from the metadata
        # file and use a dense (non-sparse) embedding for dense Adam.
        self.single_material_id = getattr(cfg, 'single_material_id', None)
        self.single_material = self.single_material_id is not None

        self.optimizer_name = getattr(cfg, 'optimizer', {}).get('name', 'SparseAdam')
        self.use_sparse_adam = (self.optimizer_name in ('SparseAdam', 'SparseAdam8bit'))

        if self.single_material:
            total_points, self._mat_H, self._mat_W = self._load_single_material_metadata(
                data_folder, self.single_material_id)
            print(f"BonnLatentBRDF single-material mode: mat{self.single_material_id:04d}, "
                  f"{total_points:,} points")
            self.point_latent_bank = nn.Embedding(
                num_embeddings=total_points,
                embedding_dim=self.total_latent_dim,
                sparse=False,
            )
        else:
            print(f"Loading Bonn point metadata from {data_folder} ...")
            self.metadata = self._load_point_metadata(data_folder)

            num_materials = self.metadata['num_materials']
            total_points = self.metadata['total_points']
            print(f"Loaded {num_materials} materials with {total_points:,} total points")
            
            is_sparse = self.use_sparse_adam
            if not is_sparse:
                print("Dense Adam or SGD selected. Setting point_latent_bank sparse=False.")
            
            self.point_latent_bank = nn.Embedding(
                num_embeddings=total_points,
                embedding_dim=self.total_latent_dim,
                sparse=is_sparse,
            )

        nn.init.normal_(self.point_latent_bank.weight, mean=0.0, std=cfg.init_std)

        with torch.no_grad():
            self.point_latent_bank.weight[:, -6:-3] = torch.tensor([0.0, 0.0, 1.0])
            self.point_latent_bank.weight[:, -3:]   = torch.tensor([0.0, 1.0, 0.0])

        self.init_normal_from_gt = getattr(cfg, 'init_normal_from_gt', False)
        if self.init_normal_from_gt:
            self._initialize_normals_from_gt(data_folder)

        self.decoder = BRDFDecoder(
            cfg=cfg.decoder,
            latent_dim=self.latent_dim,
            use_pos_enc=self.use_pos_enc,
            different_decoder=self.different_decoder,
        )

        self.smooth_reg = getattr(cfg.decoder, 'smooth_reg', False)
        self.smooth_reg_eps = getattr(cfg.decoder, 'smooth_reg_eps', 0.01)
        print("BonnLatentBRDF initialisation complete!")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def _load_point_metadata(self, data_folder):
        """Load from ``bonn_point_metadata.json``.

        JSON format:  { "1": {"H": 512, "W": 512, "num_points": 262144}, ... }
        All materials in the file are included.
        """
        import json
        from pathlib import Path

        meta_path = Path(data_folder) / "bonn_point_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found.  Run:\n"
                f"  python scripts/comparisons/generate_bonn_metadata.py {data_folder}")

        with open(meta_path) as f:
            raw = json.load(f)

        # In debug mode, only use the first debug_num materials
        debug = getattr(self.cfg, 'debug', False)
        debug_num = getattr(self.cfg, 'debug_num', 1)
        debug_rotate = getattr(self.cfg, 'debug_rotate', False)
        debug_swap_channels = getattr(self.cfg, 'debug_swap_channels', False)

        sorted_keys = sorted(raw.keys(), key=lambda k: int(k))
        if debug:
            sorted_keys = sorted_keys[:debug_num]
            print(f"[DEBUG] Using only {debug_num} material(s) for latent bank")

        materials = []
        material_point_offsets = {}
        global_offset = 0

        for mat_id_str in sorted_keys:
            mat_id = int(mat_id_str)
            entry = raw[mat_id_str]
            num_points = entry['num_points']

            # Apply per-material point subsampling to align with the dataloader,
            # which compacts point_ids to dense 0..N_sub-1 per material. Same
            # max(1, int(N * ratio)) recipe as MultiMaterialDenseDataset and
            # BonnDataset._load_single_material.
            if self.point_subsample_ratio < 1.0:
                num_points = max(1, int(num_points * self.point_subsample_ratio))

            materials.append({
                'material_id': mat_id,
                'name': f'mat{mat_id:04d}',
                'H': entry['H'],
                'W': entry['W'],
                'num_points': num_points,
                'num_observations': 0,
                'point_range': (global_offset, global_offset + num_points),
                'folder': str(data_folder),
            })
            material_point_offsets[mat_id] = global_offset
            global_offset += num_points
            print(f"  mat{mat_id:04d}: {entry['H']}x{entry['W']} = {num_points:,} points")

        # Debug pair: add a synthetic second material with same num_points as the first
        if (debug_rotate or debug_swap_channels) and materials:
            first = materials[0]
            fake_id = first['material_id'] + 1
            num_points = first['num_points']
            materials.append({
                'material_id': fake_id,
                'name': f'mat{fake_id:04d}',
                'H': first.get('H'),
                'W': first.get('W'),
                'num_points': num_points,
                'num_observations': 0,
                'point_range': (global_offset, global_offset + num_points),
                'folder': str(data_folder),
            })
            material_point_offsets[fake_id] = global_offset
            global_offset += num_points
            print(f"  mat{fake_id:04d} (debug copy): {num_points:,} points")

        if not materials:
            raise ValueError(f"No materials found in {meta_path}")

        metadata = {
            'num_materials': len(materials),
            'total_points': global_offset,
            'materials': materials,
            'material_point_offsets': material_point_offsets,
        }

        max_mat_id = max(m['material_id'] for m in materials)
        offset_tensor = torch.zeros(max_mat_id + 1, dtype=torch.long)
        for m in materials:
            offset_tensor[m['material_id']] = material_point_offsets[m['material_id']]
        self.register_buffer('material_offset_tensor', offset_tensor)

        return metadata

    @staticmethod
    def _load_single_material_metadata(data_folder, mat_id):
        """Read metadata for *one* material from ``bonn_point_metadata.json``.

        Returns (num_points, H, W).
        """
        import json
        from pathlib import Path

        meta_path = Path(data_folder) / "bonn_point_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found.  Run:\n"
                f"  python scripts/comparisons/generate_bonn_metadata.py {data_folder}")
        with open(meta_path) as f:
            raw = json.load(f)
        key = str(mat_id)
        if key not in raw:
            raise KeyError(
                f"Material {mat_id} not found in {meta_path}. "
                f"Available: {sorted(raw.keys(), key=lambda k: int(k))}")
        entry = raw[key]
        return entry['num_points'], entry['H'], entry['W']

    # ------------------------------------------------------------------
    # GT normal initialization
    # ------------------------------------------------------------------
    def _initialize_normals_from_gt(self, data_folder):
        """Initialize latent bank normal slots from AxF-decoded GT normal maps."""
        from pathlib import Path
        import numpy as np
        from datasets.bonn import _read_gt_normal_map

        svfresnel_dir = Path(data_folder) / 'Bonn_svfresnel'
        count = 0

        with torch.no_grad():
            if self.single_material:
                gt_n = _read_gt_normal_map(
                    svfresnel_dir, self.single_material_id,
                    self._mat_H, self._mat_W)
                if gt_n is not None:
                    self.point_latent_bank.weight[:, -6:-3] = torch.from_numpy(gt_n)
                    count += 1
                else:
                    print(f"  [Warning] GT normal not found for mat{self.single_material_id:04d}, "
                          f"keeping default (0,0,1)")
            else:
                for mat_info in self.metadata['materials']:
                    mat_id = mat_info['material_id']
                    H, W = mat_info['H'], mat_info['W']
                    offset = mat_info['point_range'][0]
                    n_pts = mat_info['num_points']

                    gt_n = _read_gt_normal_map(svfresnel_dir, mat_id, H, W)
                    if gt_n is None:
                        print(f"  [Warning] GT normal not found for mat{mat_id:04d}, "
                              f"keeping default (0,0,1)")
                        continue

                    # Subsample with the same seed (mat_id) and ratio used by
                    # the dataloader so the slot at local index k holds the
                    # same physical pixel's GT normal that the dataloader emits.
                    if self.point_subsample_ratio < 1.0 and gt_n.shape[0] != n_pts:
                        full_n = gt_n.shape[0]
                        rng = np.random.default_rng(mat_id)
                        sub_idx = np.sort(
                            rng.choice(full_n, size=n_pts, replace=False))
                        gt_n = gt_n[sub_idx]

                    self.point_latent_bank.weight[offset:offset + n_pts, -6:-3] = \
                        torch.from_numpy(gt_n)
                    count += 1

        print(f"Initialized normals from GT for {count} material(s)")

    # ------------------------------------------------------------------
    # Point-ID mapping
    # ------------------------------------------------------------------
    def get_global_point_id(self, material_id, local_point_id):
        if self.single_material:
            return local_point_id
        offsets = self.material_offset_tensor[material_id]
        return local_point_id + offsets

    # ------------------------------------------------------------------
    # Frame helpers
    # ------------------------------------------------------------------
    def extract_frame_from_latent(self, latent: torch.Tensor, input_normal=None):
        # When input_normal is provided (predict_frame=False), use it as the normal
        # and only use the latent for the tangent direction.
        if input_normal is not None:
            predicted_normal = NF.normalize(input_normal, dim=-1)
        else:
            predicted_normal = NF.normalize(latent[..., -6:-3], dim=-1)

        predicted_tangent = NF.normalize(latent[..., -3:], dim=-1)

        predicted_tangent = predicted_tangent - \
            torch.sum(predicted_tangent * predicted_normal, dim=-1, keepdim=True) * predicted_normal
        predicted_tangent = NF.normalize(predicted_tangent, dim=-1)
        if torch.isnan(predicted_tangent).any():
            print("predicted_tangent is nan")

        return predicted_normal, predicted_tangent

    def world_to_local(self, v, normal, tangent=None):
        if tangent is None:
            up = torch.tensor([0.0, 1.0, 0.0], device=normal.device).expand_as(normal)
            tangent = torch.cross(up, normal)
            tangent_len = tangent.norm(dim=-1, keepdim=True)
            collinear_mask = tangent_len.squeeze(-1) < 1e-6
            if collinear_mask.any():
                right = torch.tensor([1.0, 0.0, 0.0], device=normal.device).expand_as(normal)
                tangent[collinear_mask] = torch.cross(right[collinear_mask], normal[collinear_mask])
                tangent_len = tangent.norm(dim=-1, keepdim=True)
        else:
            tangent_len = tangent.norm(dim=-1, keepdim=True)

        tangent = tangent / (tangent_len + 1e-8)
        bitangent = torch.cross(normal, tangent)

        return torch.stack([
            (v * tangent).sum(dim=-1),
            (v * bitangent).sum(dim=-1),
            (v * normal).sum(dim=-1),
        ], dim=-1)

    # ------------------------------------------------------------------
    # BRDF evaluation
    # ------------------------------------------------------------------
    def eval_brdf(
        self,
        pos,
        wi,
        wo,
        normal,
        latent=None,
        point_ids=None,
        material_ids=None,
        return_wi_local=False,
    ):
        if point_ids is None or material_ids is None:
            raise ValueError("point_ids and material_ids must be provided")

        global_point_ids = self.get_global_point_id(material_ids, point_ids)
        latent = self.point_latent_bank(global_point_ids)

        input_normal = None if self.predict_frame else normal
        predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent, input_normal)

        NoL = (wi * predicted_normal).sum(-1, keepdim=True)
        NoV = (wo * predicted_normal).sum(-1, keepdim=True)
        wi_local = self.world_to_local(wi, predicted_normal, predicted_tangent)
        wo_local = self.world_to_local(wo, predicted_normal, predicted_tangent)
        normal_local = torch.zeros_like(wi_local)
        normal_local[..., 2] = 1.0

        enc_dir = self.decoder.encode_directions(wi_local, wo_local, normal_local)

        brdf_lat = latent[:, : self.brdf_latent_dim]
        brdf = self.decoder(enc_dir, brdf_lat)
        if not self.different_decoder and brdf.shape[-1] == 1:
            brdf = brdf.expand(-1, 3)

        if self.smooth_reg:
            eps = self.smooth_reg_eps
            rand_vec = torch.randn_like(wi_local)
            rand_vec = rand_vec - (rand_vec * wi_local).sum(-1, keepdim=True) * wi_local
            axis = NF.normalize(rand_vec, dim=-1)
            wi_perturbed = wi_local * math.cos(eps) + torch.cross(axis, wi_local, dim=-1) * math.sin(eps)
            enc_pert = self.decoder.encode_directions(wi_perturbed, wo_local, normal_local)
            brdf_pert = self.decoder(enc_pert, brdf_lat)
            if not self.different_decoder and brdf_pert.shape[-1] == 1:
                brdf_pert = brdf_pert.expand(-1, 3)
            smooth_loss = ((brdf_pert - brdf) / eps).pow(2).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=wi.device)

        if self.learnable_factor:
            brdf = brdf * self.factor

        pdf = NoL.clamp(min=0) / math.pi
        if torch.isnan(brdf).any():
            print("brdf is nan")
        if torch.isnan(predicted_normal).any():
            print("normal is nan")
        if return_wi_local:
            return brdf, predicted_normal, pdf, smooth_loss, wi_local
        return brdf, predicted_normal, pdf, smooth_loss

    # ------------------------------------------------------------------
    # BRDF sampling (cosine-weighted hemisphere)
    # ------------------------------------------------------------------
    def sample_brdf(
        self, params, pos, sample1, sample2, wo, normal,
        latent=None, batch_mask=None, point_ids=None, material_ids=None,
    ):
        theta = torch.asin(torch.sqrt(sample2[..., 0]))
        phi = 2 * math.pi * sample2[..., 1]

        wi_local = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ], dim=-1)

        up = torch.tensor([0.0, 1.0, 0.0], device=normal.device).expand_as(normal)
        tangent = torch.cross(up, normal)
        tangent = tangent / (tangent.norm(dim=-1, keepdim=True) + 1e-8)
        bitangent = torch.cross(normal, tangent)

        wi = (wi_local[..., 0:1] * tangent +
              wi_local[..., 1:2] * bitangent +
              wi_local[..., 2:3] * normal)

        brdf, pdf, _ = self.eval_brdf(
            None, pos, wi, wo, normal,
            point_ids=point_ids, material_ids=material_ids)

        brdf_weight = brdf / pdf.clamp(min=1e-6)
        return wi, pdf, brdf_weight


class MERLBRDF(LightningModule):
    """
    Multi-material BRDF model using auto-decoder architecture.
    - Per-material latent codes (M materials)
    - Per-point latent codes (sum of points across all materials)
    - Shared MLP decoder across all materials
    
    Expected folder structure:
        data_folder/
            0/                      # material_id = 0
                point_metadata.json # contains {"num_points": N, ...}
            1/                      # material_id = 1
                point_metadata.json
            ...
    """
    def __init__(self, cfg):
        super().__init__()
        
        # Store configuration
        self.cfg = cfg
        self.num_materials = cfg.num_materials
        data_folder = getattr(cfg, 'data_folder', None)
        self.start_material_id = cfg.start_material_id
        # Latent dimensions
        self.latent_dim = cfg.latent_dim
        self.predict_frame = cfg.predict_frame
        self.different_decoder = cfg.different_decoder
        self.brdf_latent_dim = self.latent_dim * 3 if self.different_decoder else self.latent_dim
        self.total_latent_dim = self.brdf_latent_dim + (6 if self.predict_frame else 0)
        # BRDF decoder settings
        self.use_pos_enc = cfg.use_pos_enc

        total_points=120
        self.point_latent_bank = nn.Embedding(
            num_embeddings=total_points,
            embedding_dim=self.total_latent_dim
        )
        nn.init.normal_(self.point_latent_bank.weight, mean=0.0, std=cfg.init_std)
        
        if self.predict_frame:
            # Last 6 dimensions: normal (0,0,1) and tangent (0,1,0)
            with torch.no_grad():
                self.point_latent_bank.weight[:, -6:-3] = torch.tensor([0.0, 0.0, 1.0])  # normal
                self.point_latent_bank.weight[:, -3:] = torch.tensor([0.0, 1.0, 0.0])    # tangent
        
        # Shared BRDF decoder
        self.decoder = BRDFDecoder(
            cfg=cfg.decoder,
            latent_dim=self.latent_dim,
            use_pos_enc=self.use_pos_enc,
            different_decoder=self.different_decoder
        )
        
        print("Initialization complete!")
    
    def eval_brdf(
        self,
        wi,
        wo,
        material_id,
    ):
        """
        Evaluate BRDF at given geometry and directions.
        
        Args:
            pos: [B, 3] 3D positions
            wi: [B, 3] incoming light directions (world space)
            wo: [B, 3] outgoing view directions (world space)
            normal: [B, 3] normals (world space)
            latent: Ignored (latents retrieved from banks)
            point_ids: [B] LOCAL point indices (per-material, from dataloader)
            material_ids: [B] material indices (required for global point ID computation)
        
        Returns:
            brdf: [B, 3] BRDF values
            normal: [B, 3] normals (local space)
            pdf: [B, 1] probability density
        """
        
        # Retrieve latents from banks
        #print("material_id",material_id)
        latent = self.point_latent_bank(material_id)        # [B, latent_dim]

        normal_local = torch.zeros_like(wi)
        normal_local[..., 2] = 1.0  # Normal is always (0,0,1) in local space
        
        # print("latent",torch.mean(latent),torch.std(latent))
        # Encode directions
        enc_dir = self.decoder.encode_directions(wi, wo, normal_local)
        
        # Decode BRDF
        brdf_lat = latent[:, : self.brdf_latent_dim]
        brdf = self.decoder(enc_dir, brdf_lat)
        if not self.different_decoder and brdf.shape[-1] == 1:
            brdf = brdf.expand(-1, 3)  # Expand to RGB
        
        
        return brdf
    
    def directions_to_rusinkiewicz(self, wi, wo):
        """
        Convert incoming (wi) and outgoing (wo) directions to Rusinkiewicz parameterization.
        
        This follows the MERL BRDF reference implementation (BRDFRead.cpp).
        
        Rusinkiewicz parameterization:
        - theta_h: angle between half-vector and normal [0, π/2]
        - theta_d: polar angle of rotated incoming vector [0, π/2]
        - phi_d: azimuthal angle of rotated incoming vector [0, π]
        
        The key insight is that theta_d is NOT simply half the angle between wi and wo.
        Instead, we rotate wi into the half-vector's coordinate frame and measure its angle.
        
        Args:
            wi: [B, 3] incoming light directions (normalized)
            wo: [B, 3] outgoing view directions (normalized)
        
        Returns:
            theta_h: [B] half-angle [0, pi/2]
            theta_d: [B] difference angle [0, pi/2]
            phi_d: [B] azimuthal difference [0, pi]
        """
        # Normalize directions
        wi = NF.normalize(wi, dim=-1)
        wo = NF.normalize(wo, dim=-1)
        
        # Compute half-vector
        h = NF.normalize(wi + wo, dim=-1)
        
        # theta_h: angle between half-vector and surface normal (z-axis)
        theta_h = torch.acos(torch.clamp(h[..., 2], -1.0, 1.0))
        
        # phi_h: azimuthal angle of half-vector
        phi_h = torch.atan2(h[..., 1], h[..., 0])
        
        # Rotate wi into half-vector coordinate frame
        # Step 1: Rotate by -phi_h around z-axis (normal)
        cos_ph = torch.cos(-phi_h)
        sin_ph = torch.sin(-phi_h)
        wi_rot1_x = wi[..., 0] * cos_ph - wi[..., 1] * sin_ph
        wi_rot1_y = wi[..., 0] * sin_ph + wi[..., 1] * cos_ph
        wi_rot1_z = wi[..., 2]
        
        # Step 2: Rotate by -theta_h around y-axis (binormal)
        cos_th = torch.cos(-theta_h)
        sin_th = torch.sin(-theta_h)
        diff_x = wi_rot1_x * cos_th + wi_rot1_z * sin_th
        diff_y = wi_rot1_y
        diff_z = -wi_rot1_x * sin_th + wi_rot1_z * cos_th
        
        # theta_d: polar angle of the rotated difference vector
        theta_d = torch.acos(torch.clamp(diff_z, -1.0, 1.0))
        
        # phi_d: azimuthal angle of the rotated difference vector
        phi_d = torch.atan2(diff_y, diff_x)
        
        # MERL uses reciprocity: phi_d in [0, π] only
        # If phi_d < 0, add π; if phi_d > π, subtract π
        phi_d = torch.where(phi_d < 0, phi_d + math.pi, phi_d)
        
        return theta_h, theta_d, phi_d
    
    def sample_brdf(
        self,
        params,
        pos,
        sample1,
        sample2,
        wo,
        normal,
        latent=None,
        batch_mask=None,
        point_ids=None,
        material_ids=None
    ):
        """
        Sample BRDF using importance sampling.
        
        Args:
            params: Ground truth parameters (not used)
            pos: [B, 3] 3D positions
            sample1: [B] uniform samples [0,1]
            sample2: [B, 2] uniform samples [0,1]^2
            wo: [B, 3] outgoing view directions (world space)
            normal: [B, 3] normals (world space)
            latent: Ignored
            batch_mask: Batch mask
            point_ids: [B] LOCAL point indices (per-material, from dataloader)
            material_ids: [B] material indices
        
        Returns:
            wi: [B, 3] sampled incoming light directions
            pdf: [B, 1] probability density
            brdf_weight: [B, 3] BRDF / pdf
        """
        # Simple cosine-weighted hemisphere sampling (can use proxy BRDF)
        # This is a placeholder - implement proper importance sampling if needed
        
        # Cosine-weighted sampling
        theta = torch.asin(torch.sqrt(sample2[..., 0]))
        phi = 2 * math.pi * sample2[..., 1]
        
        # Local space directions
        wi_local = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ], dim=-1)
        
        # Transform to world space
        # Build TBN frame
        up = torch.tensor([0.0, 1.0, 0.0], device=normal.device).expand_as(normal)
        tangent = torch.cross(up, normal)
        tangent_len = tangent.norm(dim=-1, keepdim=True)
        tangent = tangent / (tangent_len + 1e-8)
        bitangent = torch.cross(normal, tangent)
        
        # Local to world
        wi = (wi_local[..., 0:1] * tangent + 
              wi_local[..., 1:2] * bitangent + 
              wi_local[..., 2:3] * normal)
        
        # Evaluate BRDF at sampled direction
        brdf, pdf, _ = self.eval_brdf(
            None, pos, wi, wo, normal,
            point_ids=point_ids,
            material_ids=material_ids
        )
        
        # BRDF weight = BRDF / pdf (for Monte Carlo integration)
        brdf_weight = brdf / pdf.clamp(min=1e-6)
        
        return wi, pdf, brdf_weight
    
    # ============================================================================
# 6. ANISOTROPIC LATENT TEXTURED MODEL - Combined material class
# ============================================================================
class LearnablePBRTexturedModel(LightningModule):
    """
    Complete material model combining all components.
    Provides the same API as the original class for compatibility.
    
    Uses PBRDecoder with latent structure:
        Isotropic (6 channels): [color(3), albedo(1), roughness(1), metallic(1)]
        Anisotropic (9 channels): [diffuse(3), ao(1), roughness(1), metallic(1), ior(1), aniso_strength(1), aniso_rot(1)]
        Disney (12 channels): [baseColor(3), ao(1), roughness(1), metallic(1), ior(1), aniso_strength(1), aniso_rot(1),
                                specularTint(1), sheen(1), sheenTint(1)]
    """
    def __init__(self, cfg):
        super().__init__()
        
        # Store configuration
        self.cfg = cfg
        self.predict_frame = cfg.predict_frame
        self.gt_frame = cfg.gt_frame
        self.anisotropic = getattr(cfg, 'anisotropic', False)
        self.disney = getattr(cfg, 'disney', False)
        self.Gaussian_blur = cfg.Gaussian_blur
        self.learnable_factor = cfg.learnable_factor
        self.soft_constraint = getattr(cfg, 'soft_constraint', True)
        self.init_std = 0.1
        
        if self.learnable_factor:
            self.factor = nn.Parameter(torch.tensor(1.0))
        
        # Neural geometry settings
        self.neural_geometry_enabled = cfg.neural_geometry.enable
        self.geometry_latent_dim = cfg.neural_geometry.latent_dim if self.neural_geometry_enabled else 0
        self.local_wi_wo = cfg.neural_geometry.local_wi_wo if self.neural_geometry_enabled else False
        self.neural_geometry_pos_enc = cfg.neural_geometry.positional_encoding if self.neural_geometry_enabled else False
        self.recompute_frame = cfg.neural_geometry.recompute_frame if self.neural_geometry_enabled else False
        self.neural_geometry_factor = cfg.neural_geometry.factor if self.neural_geometry_enabled else 0.4
        
        # Calculate PBR latent dimension based on model type
        if self.disney:
            brdf_latent_dim = 12
        elif self.anisotropic:
            brdf_latent_dim = 9
        else:
            brdf_latent_dim = 6
        self.brdf_latent_dim = brdf_latent_dim
        
        # Add frame dimensions if predicting TBN
        total_latent_dim = brdf_latent_dim
        if self.predict_frame:
            total_latent_dim += 6  # normal (3) + tangent (3)
        
        # Add geometry latent dimensions
        if self.neural_geometry_enabled:
            total_latent_dim += self.geometry_latent_dim
        
        self.total_latent_dim = total_latent_dim
        
        # 1. Create LatentTexture
        self.texture_resolution = getattr(cfg, 'texture_resolution', 256)
        blur_config = {
            'blur_sigma0': 2.0,
            'blur_half_life': 3333
        } if self.Gaussian_blur else None
        
        self.latent_texture = LatentTexture(
            resolution=self.texture_resolution,
            latent_dim=total_latent_dim,
            predict_frame=self.predict_frame,
            init_std=self.init_std,
            blur_config=blur_config
        )
        
        # 2. Create PBRDecoder
        self.decoder = PBRDecoder(
            cfg=cfg,
            soft_constraint=self.soft_constraint
        )
        
        # 3. Create NeuralGeometry if enabled
        if self.neural_geometry_enabled:
            self.neural_geometry = NeuralGeometry(
                cfg=cfg.neural_geometry,
                geometry_latent_dim=self.geometry_latent_dim,
                use_local_wi_wo=self.local_wi_wo,
                use_pos_enc=self.neural_geometry_pos_enc
            )
        else:
            self.neural_geometry = None
        
        # 4. Proxy BRDF for importance sampling
        self.proxy_brdf = ProxyPBRBRDF()
        
        # 5. Latent bank mode (alternative to texture-based sampling)
        self.use_latent_bank = getattr(cfg, 'use_latent_bank', False)
        
        if self.use_latent_bank:
            observations_folder = cfg.observations_folder
            
            # Compute global point offset for this material
            self.global_point_offset = self._compute_global_point_offset(observations_folder)
            
            # Load point positions from observations
            self.point_positions = self._load_point_positions(observations_folder)
            num_points = self.point_positions.shape[0]
            print(f"[LatentBank] Loaded {num_points:,} point positions from {observations_folder}")
            
            # Build KD-tree for efficient nearest neighbor lookup
            from scipy.spatial import cKDTree
            self.kdtree = cKDTree(self.point_positions.numpy())
            print(f"[LatentBank] Built KD-tree for {num_points:,} points")
            
            # Latent bank will be loaded from checkpoint - don't initialize here
            self.point_latent_bank = None
            self.bank_latent_dim = brdf_latent_dim + (6 if self.predict_frame else 0)
            print(f"[LatentBank] Latent bank will be loaded from checkpoint (latent_dim={self.bank_latent_dim})")
    
    # ------------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------------
    def compute_uv(self, pos: torch.Tensor, width: float = 0.4, length: float = 0.4):
        """
        Compute UV coordinates from 3D positions.
        
        Args:
            pos: [B, 3] 3D positions
            width: Width of the surface
            length: Length of the surface
        
        Returns:
            uv: [B, 2] UV coordinates in [0, 1]
        """
        half_w = width * 0.5
        half_l = length * 0.5
        x, z = pos[:, 0], pos[:, 2]
        
        u = (x + half_w) / width
        v = (z + half_l) / length
        
        return torch.stack([u, v], dim=-1)
    
    def world_to_local(
        self,
        v: torch.Tensor,
        normal: torch.Tensor,
        tangent: torch.Tensor
    ):
        """
        Transform vector from world space to local tangent space.
        
        Args:
            v: [B, 3] vector in world space
            normal: [B, 3] normal in world space
            tangent: [B, 3] tangent in world space
        
        Returns:
            v_local: [B, 3] vector in local space
        """
        if tangent is None:
            # Generate arbitrary tangent perpendicular to normal
            tangent = torch.cross(
                normal,
                torch.tensor([0.0, 0.0, 1.0], device=normal.device).expand_as(normal)
            )
        
        tangent_len = tangent.norm(dim=-1, keepdim=True)
        tangent = tangent / (tangent_len + 1e-8)
        bitangent = torch.cross(normal, tangent)
        
        v_local = torch.stack([
            (v * tangent).sum(dim=-1),
            (v * bitangent).sum(dim=-1),
            (v * normal).sum(dim=-1)
        ], dim=-1)
        
        return v_local
    
    def extract_frame_from_latent(self, latent: torch.Tensor):
        """
        Extract and orthonormalize normal and tangent from latent code.
        
        Args:
            latent: [B, total_dim] latent code with last 6 dims as normal+tangent
        
        Returns:
            normal: [B, 3] normalized normal vector
            tangent: [B, 3] normalized tangent vector (orthogonal to normal)
        """
        predicted_normal = latent[..., -6:-3]
        predicted_tangent = latent[..., -3:]
        
        # Normalize
        predicted_normal = NF.normalize(predicted_normal, dim=-1)
        predicted_tangent = NF.normalize(predicted_tangent, dim=-1)
        
        # Gram-Schmidt orthogonalization
        predicted_tangent = predicted_tangent - \
            torch.sum(predicted_tangent * predicted_normal, dim=-1, keepdim=True) * predicted_normal
        predicted_tangent = NF.normalize(predicted_tangent, dim=-1)
        
        return predicted_normal, predicted_tangent
    
    def _sample_from_texture(self, uv: torch.Tensor, texture: torch.Tensor):
        """
        Sample latent from a given texture at UV coordinates.
        
        Args:
            uv: [B, 2] UV coordinates in [0, 1]
            texture: [1, D, H, W] texture to sample from
        
        Returns:
            latent: [B, D] sampled latent codes
        """
        # Convert UV to grid coordinates for F.grid_sample
        grid_coords = uv * 2.0 - 1.0  # [B, 2] -> [-1, 1]
        grid_coords = grid_coords.unsqueeze(1).unsqueeze(0)  # [1, 1, B, 2]
        
        # Bilinear sampling
        latent = NF.grid_sample(
            texture,  # [1, D, H, W]
            grid_coords,  # [1, 1, B, 2]
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # [1, D, 1, B]
        
        # Reshape to [B, D]
        latent = latent.squeeze(0).squeeze(-1).transpose(0, 1)
        
        return latent
    
    # ------------------------------------------------------------------------
    # Latent Bank Methods (for use_latent_bank mode)
    # ------------------------------------------------------------------------
    def _compute_global_point_offset(self, observations_folder: str) -> int:
        """
        Compute the global point offset for this material based on folder path.
        
        The observations folder path contains the material ID (e.g., 'data/0/observations/').
        The offset is the sum of num_points from all materials with ID < current material ID.
        
        Args:
            observations_folder: Path to observations folder (e.g., 'data/0/observations/')
        
        Returns:
            offset: int, global point offset for indexing into the latent bank
        """
        import json
        from pathlib import Path
        
        obs_folder = Path(observations_folder)
        material_folder = obs_folder.parent
        data_folder = material_folder.parent
        
        # Extract current material ID from folder name
        current_material_id = int(material_folder.name)
        
        # Find all material folders with ID < current material ID
        offset = 0
        for mat_id in range(current_material_id):
            mat_folder = data_folder / str(mat_id)
            metadata_path = mat_folder / "point_metadata.json"
            
            if not metadata_path.exists():
                print(f"[LatentBank] Warning: {metadata_path} not found, skipping material {mat_id}")
                continue
            
            try:
                with open(metadata_path, 'r') as f:
                    point_meta = json.load(f)
                num_points = point_meta['num_points']
                offset += num_points
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[LatentBank] Warning: Failed to read {metadata_path}: {e}, skipping material {mat_id}")
                continue
        
        print(f"[LatentBank] Material {current_material_id}: global point offset = {offset:,}")
        return offset
    
    def _load_point_positions(self, observations_folder: str) -> torch.Tensor:
        """
        Load point positions from point_positions.npz file.
        
        Args:
            observations_folder: Path to folder containing observation chunks
        
        Returns:
            point_positions: [num_unique_points, 3] tensor of unique point positions
        """
        import numpy as np
        from pathlib import Path
        
        obs_folder = Path(observations_folder)
        material_folder = obs_folder.parent
        positions_path = material_folder / "point_positions.npz"
        
        data = np.load(positions_path)
        point_positions = torch.from_numpy(data['positions']).float()
        print(f"[LatentBank] Loaded point positions: {point_positions.shape}")
        
        return point_positions
    
    def _find_nearest_point_ids(self, pos: torch.Tensor, k: int = 1) -> tuple:
        """
        Find K nearest point IDs for query positions using KD-tree.
        
        Args:
            pos: [B, 3] query positions
            k: number of nearest neighbors
        
        Returns:
            distances: [B, k] distances to nearest points
            point_ids: [B, k] tensor of nearest point IDs
        """
        # Query KD-tree (CPU operation)
        distances, point_ids = self.kdtree.query(pos.detach().cpu().numpy(), k=k)
        distances = torch.from_numpy(distances).to(pos.device).float()
        point_ids = torch.from_numpy(point_ids).to(pos.device)
        return distances, point_ids
    
    def _sample_from_latent_bank(self, pos: torch.Tensor, k: int = 8) -> torch.Tensor:
        """
        Sample latent codes from point bank using K-NN with inverse distance weighting.
        
        Args:
            pos: [B, 3] query positions
            k: number of nearest neighbors for interpolation
        
        Returns:
            latent: [B, latent_dim] interpolated latent codes
        """
        # Find K nearest point IDs and distances (local indices for this material)
        distances, local_point_ids = self._find_nearest_point_ids(pos, k=k)  # [B, k], [B, k]
        
        # Convert local point IDs to global point IDs for latent bank lookup
        global_point_ids = local_point_ids + self.global_point_offset
        
        # Lookup latents for all K neighbors using global IDs
        latents = self.point_latent_bank(global_point_ids)  # [B, k, latent_dim]
        
        # Inverse distance weighting
        eps = 1e-8
        weights = 1.0 / (distances + eps)  # [B, k]
        weights = weights / weights.sum(dim=-1, keepdim=True)  # normalize to sum to 1
        
        # Weighted sum of latents
        latent = (weights.unsqueeze(-1) * latents).sum(dim=1)  # [B, latent_dim]
        
        return latent
    
    # ------------------------------------------------------------------------
    # Main BRDF evaluation API
    # ------------------------------------------------------------------------
    def eval_brdf(
        self,
        gt_params,
        pos: torch.Tensor,
        wi: torch.Tensor,
        wo: torch.Tensor,
        normal: torch.Tensor,
        uv: torch.Tensor,
        TBN: torch.Tensor,
        latent=None,
        batch_mask=None,
        footprint_vis=None,
        dp_du=None,
        dp_dv=None
    ):
        """
        Evaluate BRDF at given geometry and directions.
        
        Args:
            gt_params: Ground truth parameters (optional)
            pos: [B, 3] 3D positions
            wi: [B, 3] incoming light directions (world space)
            wo: [B, 3] outgoing view directions (world space)
            normal: [B, 3] normals (world space)
            uv: [B, 2] UV coordinates
            TBN: [B, 3, 3] tangent-bitangent-normal frame
            latent: Ignored (latent is sampled internally)
            batch_mask: Batch mask for batched operations
            footprint_vis: Footprint for mipmap level selection
            dp_du, dp_dv: Ray differentials
        
        Returns:
            brdf: [B, 3] BRDF values
            pdf: [B, 1] probability density
            uv_offset: [B, 2] UV offset (if neural geometry enabled, else zeros)
        """
        NoL = (wi * normal).sum(-1, keepdim=True)
        NoV = (wo * normal).sum(-1, keepdim=True)
        
        # Initialize uv_offset (used in return, always needed)
        uv_offset = torch.zeros_like(uv)
        
        # =====================================================================
        # LATENT BANK MODE: Sample latent from point bank using position
        # =====================================================================
        if self.use_latent_bank:
            # Sample latent from latent bank using nearest neighbor lookup
            latent = self._sample_from_latent_bank(pos)
            
            # Extract frame from latent if predicting frame
            if self.predict_frame:
                predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)
            else:
                # Use geometry normal and tangent
                predicted_normal = normal
                predicted_tangent = TBN[:, :, 0] if TBN is not None else None
            
            # No neural geometry in latent bank mode
        
        # =====================================================================
        # TEXTURE MODE: Sample latent from texture using UV coordinates
        # =====================================================================
        else:
            # 1. Get the (optionally blurred) texture ONCE
            if self.training and self.Gaussian_blur:
                tex = self.latent_texture.apply_gaussian_blur(self.global_step)
            else:
                tex = self.latent_texture.params
            
            # 2. Sample latent from the texture
            latent = self._sample_from_texture(uv, tex)
            
            # 3. Extract frame from latent if predicting frame
            if self.predict_frame:
                predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)
            else:
                # Use geometry normal and tangent
                predicted_normal = normal
                predicted_tangent = TBN[:, :, 0] if TBN is not None else None
            
            # 4. Predict UV offset if neural geometry is enabled
            if self.neural_geometry_enabled:
                # Extract geometry latent
                geometry_latent = latent[..., -6-self.geometry_latent_dim:-6] if self.predict_frame else \
                                 latent[..., -self.geometry_latent_dim:]
                
                # Get directions for geometry network
                if self.local_wi_wo:
                    wi_for_geo = self.world_to_local(wi, predicted_normal, predicted_tangent)
                    wo_for_geo = self.world_to_local(wo, predicted_normal, predicted_tangent)
                else:
                    wi_for_geo = wi
                    wo_for_geo = wo
                
                # Predict UV offset
                uv_offset = self.neural_geometry(wi_for_geo, wo_for_geo, geometry_latent) * self.neural_geometry_factor
                uv = uv + uv_offset
                uv = ((uv%1)+1)%1
                # Sample from the SAME blurred texture
                latent = self._sample_from_texture(uv, tex)
                
                # Recompute frame if needed
                if self.recompute_frame and self.predict_frame:
                    predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)
        
        # =====================================================================
        # Common code path for both modes
        # =====================================================================
        # 5. Transform directions to local space (canonical space: normal=(0,0,1), tangent=(0,1,0))
        wi_local = self.world_to_local(wi, predicted_normal, predicted_tangent)
        wo_local = self.world_to_local(wo, predicted_normal, predicted_tangent)
        
        # 6. Extract BRDF latent (first brdf_latent_dim channels)
        brdf_latent = latent[..., :self.brdf_latent_dim]
        
        # 7. Evaluate PBR BRDF using the decoder
        # PBRDecoder takes (wi, wo, latent) in canonical space and returns (brdf, pdf)
        brdf, pdf = self.decoder(wi_local, wo_local, brdf_latent)
        
        # 8. Apply learnable factor if enabled
        if self.learnable_factor:
            brdf = brdf * self.factor
        
        return brdf, predicted_normal, pdf, uv_offset
    
    def sample_brdf(
        self,
        params,
        pos: torch.Tensor,
        sample1: torch.Tensor,
        sample2: torch.Tensor,
        wo: torch.Tensor,
        normal: torch.Tensor,
        latent=None,
        batch_mask=None
    ):
        """
        Importance sample BRDF using proxy BRDF and evaluate with neural BRDF.
        
        Args:
            params: Dictionary with 'roughness' for proxy BRDF
            pos: [B, 3] 3D positions
            sample1: [B] uniform samples [0,1] for diffuse/specular choice
            sample2: [B, 2] uniform samples for hemisphere sampling
            wo: [B, 3] outgoing view directions (world space)
            normal: [B, 3] normals (world space)
            latent: Ignored (not used)
            batch_mask: Batch mask
        
        Returns:
            wi: [B, 3] sampled incoming directions (world space)
            pdf: [B, 1] sampling probability from proxy BRDF
            brdf_weight: [B, 3] BRDF/PDF ratio
        """
        # Sample direction from proxy BRDF
        wi_proxy, pdf_proxy = self.proxy_brdf.sample_brdf(
            pos, sample1, sample2, wo, normal,
            params['roughness'], batch_mask
        )
        
        # Evaluate neural BRDF at sampled direction
        # Note: Need to compute UV and TBN here
        uv = self.compute_uv(pos)
        TBN = None  # Compute TBN if needed
        
        mlp_brdf, _, _ = self.eval_brdf(
            params, pos, wi_proxy, wo, normal, uv, TBN,
            latent, batch_mask
        )
        
        # Calculate importance weight
        stop_gradient_pdf_proxy = pdf_proxy.detach()
        brdf_weight = torch.where(
            pdf_proxy > 0,
            mlp_brdf / (stop_gradient_pdf_proxy + 1e-8),
            torch.zeros_like(mlp_brdf)
        )
        
        return wi_proxy, stop_gradient_pdf_proxy, brdf_weight
    
    # ------------------------------------------------------------------------
    # Save/Load functionality
    # ------------------------------------------------------------------------
    def save_latent(self, path: str):
        """Save latent texture to file"""
        self.latent_texture.save(path)
    
    def load_latent(self, path: str):
        """Load latent texture from file"""
        self.latent_texture.load(path)
        
    


# ============================================================================
# 10. UBO LATENT BRDF - Auto-decoder for UBO2014 BTF dataset
# ============================================================================
class UBOLatentBRDF(LightningModule):
    """BRDF model for the Bonn UBO2014 BTF dataset.

    Single-material auto-decoder with per-texel latent codes.  The BTF
    data is a flat sample with normal = (0, 0, 1), so all directions are
    already in local space and no world-to-local transform is needed.

    Key differences from BonnLatentBRDF:
      - No material_ids (single material only, no global offset mapping)
      - No xyz positions or world-to-local transforms
      - Latent bank size is determined from BTF spatial resolution
      - No GT normal initialization (flat sample, normal is always up)
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.latent_dim = cfg.latent_dim
        self.predict_frame = getattr(cfg, 'predict_frame', False)
        self.different_decoder = getattr(cfg, 'different_decoder', False)
        self.brdf_latent_dim = self.latent_dim * 3 if self.different_decoder else self.latent_dim
        self.use_pos_enc = getattr(cfg, 'use_pos_enc', True)

        self.learnable_factor = getattr(cfg, 'learnable_factor', False)
        if self.learnable_factor:
            self.factor = nn.Parameter(torch.ones(3))

        # Determine latent bank size from BTF file
        btf_path = getattr(cfg, 'btf_path', None)
        if btf_path is not None:
            from btf_extractor import Ubo2014
            btf = Ubo2014(btf_path)
            H, W, _ = btf.img_shape
            total_points = H * W
            del btf
            print(f"UBOLatentBRDF: BTF {H}×{W} = {total_points:,} texels")
        else:
            # Fallback: use config values
            H = getattr(cfg, 'img_height', 400)
            W = getattr(cfg, 'img_width', 400)
            total_points = H * W
            print(f"UBOLatentBRDF: using config {H}×{W} = {total_points:,} texels")

        self._H = H
        self._W = W

        # Frame dims: always reserve 6 dims for normal+tangent when predict_frame
        if self.predict_frame:
            self.total_latent_dim = self.brdf_latent_dim + 6
        else:
            self.total_latent_dim = self.brdf_latent_dim

        self.point_latent_bank = nn.Embedding(
            num_embeddings=total_points,
            embedding_dim=self.total_latent_dim,
            sparse=False,  # always dense for stage-2 / Adam
        )

        nn.init.normal_(self.point_latent_bank.weight, mean=0.0, std=cfg.init_std)

        # Initialize frame slots if predict_frame
        if self.predict_frame:
            with torch.no_grad():
                self.point_latent_bank.weight[:, -6:-3] = torch.tensor([0.0, 0.0, 1.0])
                self.point_latent_bank.weight[:, -3:]   = torch.tensor([0.0, 1.0, 0.0])

        # Decoder
        self.decoder = BRDFDecoder(
            cfg=cfg.decoder,
            latent_dim=self.latent_dim,
            use_pos_enc=self.use_pos_enc,
            different_decoder=self.different_decoder,
        )

        self.smooth_reg = getattr(cfg.decoder, 'smooth_reg', False)
        self.smooth_reg_eps = getattr(cfg.decoder, 'smooth_reg_eps', 0.01)
        print("UBOLatentBRDF initialisation complete!")

    # ------------------------------------------------------------------
    # Frame helpers (only used when predict_frame=True)
    # ------------------------------------------------------------------
    def extract_frame_from_latent(self, latent: torch.Tensor):
        predicted_normal = NF.normalize(latent[..., -6:-3], dim=-1)
        predicted_tangent = NF.normalize(latent[..., -3:], dim=-1)

        # Gram-Schmidt orthogonalise tangent w.r.t. normal
        predicted_tangent = predicted_tangent - \
            torch.sum(predicted_tangent * predicted_normal, dim=-1, keepdim=True) * predicted_normal
        predicted_tangent = NF.normalize(predicted_tangent, dim=-1)

        return predicted_normal, predicted_tangent

    def world_to_local(self, v, normal, tangent):
        bitangent = torch.cross(normal, tangent, dim=-1)
        return torch.stack([
            (v * tangent).sum(dim=-1),
            (v * bitangent).sum(dim=-1),
            (v * normal).sum(dim=-1),
        ], dim=-1)

    # ------------------------------------------------------------------
    # BRDF evaluation
    # ------------------------------------------------------------------
    def eval_brdf(self, wi, wo, point_ids=None, return_wi_local=False):
        """Evaluate BRDF for given directions and point IDs.

        Args:
            wi: [B, 3] light directions (local frame for flat BTF sample)
            wo: [B, 3] view directions (local frame for flat BTF sample)
            point_ids: [B] texel indices
            return_wi_local: if True, also return wi after rotation into the
                predicted local frame (needed if the caller wants to multiply
                by NoL = wi_local.z against the predicted shading normal)

        Returns:
            brdf: [B, 3] BRDF values
            smooth_loss: scalar smoothness regularisation loss
            wi_local: [B, 3] (only if return_wi_local=True)
        """
        if point_ids is None:
            raise ValueError("point_ids must be provided")

        latent = self.point_latent_bank(point_ids)  # [B, total_latent_dim]

        if self.predict_frame:
            predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)
            wi_local = self.world_to_local(wi, predicted_normal, predicted_tangent)
            wo_local = self.world_to_local(wo, predicted_normal, predicted_tangent)
        else:
            # BTF directions are already in local frame (flat sample)
            wi_local = wi
            wo_local = wo

        normal_local = torch.zeros_like(wi_local)
        normal_local[..., 2] = 1.0

        enc_dir = self.decoder.encode_directions(wi_local, wo_local, normal_local)

        brdf_lat = latent[:, :self.brdf_latent_dim]
        brdf = self.decoder(enc_dir, brdf_lat)
        if not self.different_decoder and brdf.shape[-1] == 1:
            brdf = brdf.expand(-1, 3)

        # Smoothness regularisation
        if self.smooth_reg:
            eps = self.smooth_reg_eps
            rand_vec = torch.randn_like(wi_local)
            rand_vec = rand_vec - (rand_vec * wi_local).sum(-1, keepdim=True) * wi_local
            axis = NF.normalize(rand_vec, dim=-1)
            wi_perturbed = wi_local * math.cos(eps) + torch.cross(axis, wi_local, dim=-1) * math.sin(eps)
            enc_pert = self.decoder.encode_directions(wi_perturbed, wo_local, normal_local)
            brdf_pert = self.decoder(enc_pert, brdf_lat)
            if not self.different_decoder and brdf_pert.shape[-1] == 1:
                brdf_pert = brdf_pert.expand(-1, 3)
            smooth_loss = ((brdf_pert - brdf) / eps).pow(2).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=wi.device)

        if self.learnable_factor:
            brdf = brdf * self.factor

        if return_wi_local:
            return brdf, smooth_loss, wi_local
        return brdf, smooth_loss


# ============================================================================
# PBR LATENT CLASSES FOR BONN AND UBO DATASETS
# ============================================================================

class BonnPBRLatentBRDF(LightningModule):
    """PBR BRDF model for the Bonn SVBRDF dataset.

    Same latent-bank architecture as BonnLatentBRDF but replaces the learned
    MLP decoder (BRDFDecoder) with an analytical PBRDecoder.  The latent
    codes now encode physical material properties directly:

        Isotropic  (6):  [color(3), albedo(1), roughness(1), metallic(1)]
        Anisotropic(9):  [diffuse(3), ao(1), roughness(1), metallic(1),
                          ior(1), aniso_strength(1), aniso_rot(1)]
        Disney     (12): anisotropic + [specularTint(1), sheen(1), sheenTint(1)]

    The last 6 latent dimensions always store the predicted normal+tangent
    frame, identical to BonnLatentBRDF.
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        data_folder = getattr(cfg, 'data_folder', None)

        self.anisotropic = getattr(cfg, 'anisotropic', False)
        self.disney = getattr(cfg, 'disney', False)
        self.soft_constraint = getattr(cfg, 'soft_constraint', True)
        self.predict_frame = cfg.predict_frame

        # PBR latent dim is fixed by model type
        if self.disney:
            self.brdf_latent_dim = 12
        elif self.anisotropic:
            self.brdf_latent_dim = 9
        else:
            self.brdf_latent_dim = 6

        # Always reserve 6 dims for normal + tangent frame
        self.total_latent_dim = self.brdf_latent_dim + 6

        self.learnable_factor = getattr(cfg, 'learnable_factor', False)
        if self.learnable_factor:
            self.factor = nn.Parameter(torch.ones(3))

        # Single-material mode
        self.single_material_id = getattr(cfg, 'single_material_id', None)
        self.single_material = self.single_material_id is not None

        if self.single_material:
            total_points, self._mat_H, self._mat_W = BonnLatentBRDF._load_single_material_metadata(
                data_folder, self.single_material_id)
            print(f"BonnPBRLatentBRDF single-material mode: mat{self.single_material_id:04d}, "
                  f"{total_points:,} points")
            self.point_latent_bank = nn.Embedding(
                num_embeddings=total_points,
                embedding_dim=self.total_latent_dim,
                sparse=False,
            )
        else:
            # Reuse metadata loader from BonnLatentBRDF (needs self.cfg for debug flags)
            print(f"Loading Bonn point metadata from {data_folder} ...")
            self.metadata = self._load_point_metadata(data_folder)

            total_points = self.metadata['total_points']
            print(f"Loaded {self.metadata['num_materials']} materials with {total_points:,} total points")

            self.point_latent_bank = nn.Embedding(
                num_embeddings=total_points,
                embedding_dim=self.total_latent_dim,
                sparse=False,  # PBR always dense (no encode_directions)
            )

        nn.init.normal_(self.point_latent_bank.weight, mean=0.0, std=cfg.init_std)

        # Initialize frame slots: normal=(0,0,1), tangent=(0,1,0)
        with torch.no_grad():
            self.point_latent_bank.weight[:, -6:-3] = torch.tensor([0.0, 0.0, 1.0])
            self.point_latent_bank.weight[:, -3:]   = torch.tensor([0.0, 1.0, 0.0])

        # Expose latent_dim for compatibility with trainer visualization code
        self.latent_dim = self.brdf_latent_dim

        # PBRDecoder (no trainable parameters — purely analytical)
        self.decoder = PBRDecoder(
            cfg=cfg,
            soft_constraint=self.soft_constraint,
        )

        print("BonnPBRLatentBRDF initialisation complete!")

    # ------------------------------------------------------------------
    # Metadata (delegate to BonnLatentBRDF's implementation)
    # ------------------------------------------------------------------
    def _load_point_metadata(self, data_folder):
        """Load from ``bonn_point_metadata.json``.

        Reuses the same JSON format as BonnLatentBRDF.
        """
        import json
        from pathlib import Path

        meta_path = Path(data_folder) / "bonn_point_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found.  Run:\n"
                f"  python scripts/comparisons/generate_bonn_metadata.py {data_folder}")

        with open(meta_path) as f:
            raw = json.load(f)

        debug = getattr(self.cfg, 'debug', False)
        debug_num = getattr(self.cfg, 'debug_num', 1)

        sorted_keys = sorted(raw.keys(), key=lambda k: int(k))
        if debug:
            sorted_keys = sorted_keys[:debug_num]
            print(f"[DEBUG] Using only {debug_num} material(s) for latent bank")

        materials = []
        material_point_offsets = {}
        global_offset = 0

        for mat_id_str in sorted_keys:
            mat_id = int(mat_id_str)
            entry = raw[mat_id_str]
            num_points = entry['num_points']

            materials.append({
                'material_id': mat_id,
                'name': f'mat{mat_id:04d}',
                'H': entry['H'],
                'W': entry['W'],
                'num_points': num_points,
                'num_observations': 0,
                'point_range': (global_offset, global_offset + num_points),
                'folder': str(data_folder),
            })
            material_point_offsets[mat_id] = global_offset
            global_offset += num_points
            print(f"  mat{mat_id:04d}: {entry['H']}x{entry['W']} = {num_points:,} points")

        if not materials:
            raise ValueError(f"No materials found in {meta_path}")

        metadata = {
            'num_materials': len(materials),
            'total_points': global_offset,
            'materials': materials,
            'material_point_offsets': material_point_offsets,
        }

        max_mat_id = max(m['material_id'] for m in materials)
        offset_tensor = torch.zeros(max_mat_id + 1, dtype=torch.long)
        for m in materials:
            offset_tensor[m['material_id']] = material_point_offsets[m['material_id']]
        self.register_buffer('material_offset_tensor', offset_tensor)

        return metadata

    # ------------------------------------------------------------------
    # Point-ID mapping
    # ------------------------------------------------------------------
    def get_global_point_id(self, material_id, local_point_id):
        if self.single_material:
            return local_point_id
        offsets = self.material_offset_tensor[material_id]
        return local_point_id + offsets

    # ------------------------------------------------------------------
    # Frame helpers (same as BonnLatentBRDF)
    # ------------------------------------------------------------------
    def extract_frame_from_latent(self, latent: torch.Tensor, input_normal=None):
        if input_normal is not None:
            predicted_normal = NF.normalize(input_normal, dim=-1)
        else:
            predicted_normal = NF.normalize(latent[..., -6:-3], dim=-1)

        predicted_tangent = NF.normalize(latent[..., -3:], dim=-1)
        predicted_tangent = predicted_tangent - \
            torch.sum(predicted_tangent * predicted_normal, dim=-1, keepdim=True) * predicted_normal
        predicted_tangent = NF.normalize(predicted_tangent, dim=-1)

        return predicted_normal, predicted_tangent

    def world_to_local(self, v, normal, tangent=None):
        if tangent is None:
            up = torch.tensor([0.0, 1.0, 0.0], device=normal.device).expand_as(normal)
            tangent = torch.cross(up, normal)
            tangent_len = tangent.norm(dim=-1, keepdim=True)
            collinear_mask = tangent_len.squeeze(-1) < 1e-6
            if collinear_mask.any():
                right = torch.tensor([1.0, 0.0, 0.0], device=normal.device).expand_as(normal)
                tangent[collinear_mask] = torch.cross(right[collinear_mask], normal[collinear_mask])
                tangent_len = tangent.norm(dim=-1, keepdim=True)
        else:
            tangent_len = tangent.norm(dim=-1, keepdim=True)

        tangent = tangent / (tangent_len + 1e-8)
        bitangent = torch.cross(normal, tangent)

        return torch.stack([
            (v * tangent).sum(dim=-1),
            (v * bitangent).sum(dim=-1),
            (v * normal).sum(dim=-1),
        ], dim=-1)

    # ------------------------------------------------------------------
    # BRDF evaluation
    # ------------------------------------------------------------------
    def eval_brdf(
        self,
        pos,
        wi,
        wo,
        normal,
        latent=None,
        point_ids=None,
        material_ids=None,
        return_wi_local=False,
    ):
        if point_ids is None or material_ids is None:
            raise ValueError("point_ids and material_ids must be provided")

        global_point_ids = self.get_global_point_id(material_ids, point_ids)
        latent = self.point_latent_bank(global_point_ids)

        input_normal = None if self.predict_frame else normal
        predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent, input_normal)

        NoL = (wi * predicted_normal).sum(-1, keepdim=True)
        NoV = (wo * predicted_normal).sum(-1, keepdim=True)

        # Transform to local frame
        wi_local = self.world_to_local(wi, predicted_normal, predicted_tangent)
        wo_local = self.world_to_local(wo, predicted_normal, predicted_tangent)

        # PBR latent: first brdf_latent_dim channels
        brdf_lat = latent[:, :self.brdf_latent_dim]

        # PBRDecoder takes (wi, wo, latent) directly — no encode_directions needed
        brdf, pdf = self.decoder(wi_local, wo_local, brdf_lat)

        if self.learnable_factor:
            brdf = brdf * self.factor

        # No smooth_loss for PBR (analytical model, no learned function to regularize)
        smooth_loss = torch.tensor(0.0, device=wi.device)

        if torch.isnan(brdf).any():
            print("brdf is nan")
        if torch.isnan(predicted_normal).any():
            print("normal is nan")

        if return_wi_local:
            return brdf, predicted_normal, pdf, smooth_loss, wi_local
        return brdf, predicted_normal, pdf, smooth_loss

    # ------------------------------------------------------------------
    # BRDF sampling (cosine-weighted hemisphere)
    # ------------------------------------------------------------------
    def sample_brdf(
        self, params, pos, sample1, sample2, wo, normal,
        latent=None, batch_mask=None, point_ids=None, material_ids=None,
    ):
        theta = torch.asin(torch.sqrt(sample2[..., 0]))
        phi = 2 * math.pi * sample2[..., 1]

        wi_local = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ], dim=-1)

        up = torch.tensor([0.0, 1.0, 0.0], device=normal.device).expand_as(normal)
        tangent = torch.cross(up, normal)
        tangent = tangent / (tangent.norm(dim=-1, keepdim=True) + 1e-8)
        bitangent = torch.cross(normal, tangent)

        wi = (wi_local[..., 0:1] * tangent +
              wi_local[..., 1:2] * bitangent +
              wi_local[..., 2:3] * normal)

        brdf, _, pdf, _ = self.eval_brdf(
            pos, wi, wo, normal,
            point_ids=point_ids, material_ids=material_ids)

        brdf_weight = brdf / pdf.clamp(min=1e-6)
        return wi, pdf, brdf_weight


class UBOPBRLatentBRDF(LightningModule):
    """PBR BRDF model for the Bonn UBO2014 BTF dataset.

    Same per-texel latent-bank architecture as UBOLatentBRDF but replaces the
    learned MLP decoder (BRDFDecoder) with an analytical PBRDecoder.

    Latent codes encode physical material properties:
        Isotropic  (6):  [color(3), albedo(1), roughness(1), metallic(1)]
        Anisotropic(9):  [diffuse(3), ao(1), roughness(1), metallic(1),
                          ior(1), aniso_strength(1), aniso_rot(1)]
        Disney     (12): anisotropic + [specularTint(1), sheen(1), sheenTint(1)]
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.anisotropic = getattr(cfg, 'anisotropic', False)
        self.disney = getattr(cfg, 'disney', False)
        self.soft_constraint = getattr(cfg, 'soft_constraint', True)
        self.predict_frame = getattr(cfg, 'predict_frame', False)

        # PBR latent dim is fixed by model type
        if self.disney:
            self.brdf_latent_dim = 12
        elif self.anisotropic:
            self.brdf_latent_dim = 9
        else:
            self.brdf_latent_dim = 6

        # Frame dims
        if self.predict_frame:
            self.total_latent_dim = self.brdf_latent_dim + 6
        else:
            self.total_latent_dim = self.brdf_latent_dim

        self.learnable_factor = getattr(cfg, 'learnable_factor', False)
        if self.learnable_factor:
            self.factor = nn.Parameter(torch.ones(3))

        # Determine latent bank size from BTF file
        btf_path = getattr(cfg, 'btf_path', None)
        if btf_path is not None:
            from btf_extractor import Ubo2014
            btf = Ubo2014(btf_path)
            H, W, _ = btf.img_shape
            total_points = H * W
            del btf
            print(f"UBOPBRLatentBRDF: BTF {H}x{W} = {total_points:,} texels")
        else:
            H = getattr(cfg, 'img_height', 400)
            W = getattr(cfg, 'img_width', 400)
            total_points = H * W
            print(f"UBOPBRLatentBRDF: using config {H}x{W} = {total_points:,} texels")

        self._H = H
        self._W = W

        self.point_latent_bank = nn.Embedding(
            num_embeddings=total_points,
            embedding_dim=self.total_latent_dim,
            sparse=False,
        )

        nn.init.normal_(self.point_latent_bank.weight, mean=0.0, std=cfg.init_std)

        # Initialize frame slots if predict_frame
        if self.predict_frame:
            with torch.no_grad():
                self.point_latent_bank.weight[:, -6:-3] = torch.tensor([0.0, 0.0, 1.0])
                self.point_latent_bank.weight[:, -3:]   = torch.tensor([0.0, 1.0, 0.0])

        # Expose latent_dim for compatibility with trainer visualization code
        self.latent_dim = self.brdf_latent_dim

        # PBRDecoder (no trainable parameters — purely analytical)
        self.decoder = PBRDecoder(
            cfg=cfg,
            soft_constraint=self.soft_constraint,
        )

        print("UBOPBRLatentBRDF initialisation complete!")

    # ------------------------------------------------------------------
    # Frame helpers (only used when predict_frame=True)
    # ------------------------------------------------------------------
    def extract_frame_from_latent(self, latent: torch.Tensor):
        predicted_normal = NF.normalize(latent[..., -6:-3], dim=-1)
        predicted_tangent = NF.normalize(latent[..., -3:], dim=-1)

        predicted_tangent = predicted_tangent - \
            torch.sum(predicted_tangent * predicted_normal, dim=-1, keepdim=True) * predicted_normal
        predicted_tangent = NF.normalize(predicted_tangent, dim=-1)

        return predicted_normal, predicted_tangent

    def world_to_local(self, v, normal, tangent):
        bitangent = torch.cross(normal, tangent, dim=-1)
        return torch.stack([
            (v * tangent).sum(dim=-1),
            (v * bitangent).sum(dim=-1),
            (v * normal).sum(dim=-1),
        ], dim=-1)

    # ------------------------------------------------------------------
    # BRDF evaluation
    # ------------------------------------------------------------------
    def eval_brdf(self, wi, wo, point_ids=None, return_wi_local=False):
        """Evaluate PBR BRDF for given directions and point IDs.

        Args:
            wi: [B, 3] light directions (local frame for flat BTF sample)
            wo: [B, 3] view directions (local frame for flat BTF sample)
            point_ids: [B] texel indices
            return_wi_local: if True, also return wi after rotation into the
                predicted local frame (for NoL = wi_local.z weighting).

        Returns:
            brdf: [B, 3] BRDF values
            smooth_loss: scalar (always 0 for PBR — no learned function)
            wi_local: [B, 3] (only if return_wi_local=True)
        """
        if point_ids is None:
            raise ValueError("point_ids must be provided")

        latent = self.point_latent_bank(point_ids)

        if self.predict_frame:
            predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)
            wi_local = self.world_to_local(wi, predicted_normal, predicted_tangent)
            wo_local = self.world_to_local(wo, predicted_normal, predicted_tangent)
        else:
            wi_local = wi
            wo_local = wo

        brdf_lat = latent[:, :self.brdf_latent_dim]

        # PBRDecoder takes (wi, wo, latent) directly
        brdf, pdf = self.decoder(wi_local, wo_local, brdf_lat)

        if self.learnable_factor:
            brdf = brdf * self.factor

        smooth_loss = torch.tensor(0.0, device=wi.device)

        if return_wi_local:
            return brdf, smooth_loss, wi_local
        return brdf, smooth_loss
