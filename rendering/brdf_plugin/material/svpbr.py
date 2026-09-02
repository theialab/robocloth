import torch
import torch.nn as nn
import torch.nn.functional as NF
import math
import numpy as np
from pytorch_lightning import LightningModule
import sys
from brdf_plugin.utils.ops import *
from brdf_plugin.utils.cuda_manage import print_cuda_memory_info
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

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
            print("tex",tex.shape)
        
        return tex.permute(1, 2, 0).contiguous()  # [H,W,C] => [U,V,C]
    except Exception as e:
        print(f"Error loading PBR textures: {e}")
        # Return a default texture if loading fails
        H, W = 1024, 1024
        return torch.ones(H, W, 12)  # Default to 12 channels for compatibility

def compute_tbn(pos, uv, width: float = 0.4, length: float = 0.4):
    """
    Build a fixed T-B-N frame for each point.

    Args
    ----
    pos   : (B, 3)  xyz positions (only used for device / dtype)
    uv    : (B, 2)  – not used here but kept for API compatibility
    width : float   tangent scale along +X
    length: float   bitangent scale along +Z
    """
    batch  = pos.shape[0]
    device = pos.device
    dtype  = pos.dtype

    # Constant tangent  (width, 0, 0)
    T = pos.new_tensor([width, 0.0, 0.0]).repeat(batch, 1)  # (B, 3)

    # Constant bitangent (0, 0, length)
    B = pos.new_tensor([0.0, 0.0, length]).repeat(batch, 1)  # (B, 3)

    # Normal = B × T   (right-handed)
    N = torch.cross(B, T, dim=-1)

    # Normalise
    T = NF.normalize(T, dim=-1)
    B = NF.normalize(B, dim=-1)
    N = NF.normalize(N, dim=-1)

    return T, B, N

def sample_texture(params, uv):
    B, H, W, C = params.shape
    
    # UV coordinates adjustment for grid_sample
    uv_grid = uv.unsqueeze(0).unsqueeze(2) * 2 - 1  # [1,N,1,2]

    # Sample the entire texture (all channels)
    sampled_texture = torch.nn.functional.grid_sample(
        params.permute(0, 3, 1, 2), uv_grid, mode='bilinear', align_corners=True
    ).squeeze(-1).permute(0, 2, 1).squeeze(0)  # [N,C]

    return sampled_texture

def local_to_world_normal(sampled_normal, T, B, N):
    # Adjust normal map from [0,1] to [-1,1]
    sampled_normal = sampled_normal * 2 - 1
    sampled_normal = torch.nn.functional.normalize(sampled_normal, dim=-1)
    #print("sampled_normal",sampled_normal.shape)
    #print("T",T.shape)
    n_world = (
        sampled_normal[:, 0:1] * T +
        sampled_normal[:, 1:2] * B +
        sampled_normal[:, 2:3] * N
    )
    n_world = torch.nn.functional.normalize(n_world, dim=-1)
    return n_world

class SvPBRBRDF(nn.Module):
    """ Base BRDF class """
    def __init__(self, cfg, albedo=torch.ones(1, 3)):
        super(SvPBRBRDF,self).__init__()
        # Initialize learnable material parameters
        
        self.albedo = nn.Parameter(albedo)
        self.anisotropic = cfg.anisotropic
        self.perlin_scale = cfg.perlin_scale
        self.perlin_randomness = cfg.perlin_randomness
        self.scale_factor = cfg.scale_factor
        self.normal_map = cfg.normal_map
        self.pbr_texture = load_pbr_texture(cfg.pbr_texture_path).unsqueeze(0).cuda()
    def diffuse_sampler(self, sample2, normal):
        """ sampling diffuse lobe: wi ~ NoV/math.pi 
        Args:
            sample2: Bx2 unIform samples
            normal: Bx3 normal
        Return:d
            wi: Bx3 sampled direction in world space
        """
        theta = torch.asin(sample2[...,0].sqrt())
        phi = math.pi*2*sample2[...,1]
        wi = angle2xyz(theta,phi)
        
        Nmat = get_normal_space(normal)
        wi = (wi[:,None]@Nmat.permute(0,2,1)).squeeze(1)    
        if wi.isnan().any():
            print("wi is nan")
        return wi


    def specular_sampler(self, sample2,roughness, wo, normal):
        """ sampling ggx lobe: h ~ D/(VoH*4)*NoH
        Args:
            sample2: Bx3 uniform samples
            roughness: Bx1 roughness
            wo: Bx3 viewing direction
            normal: Bx3 normal
        Return:
            wi: Bx3 sampled direction in world space
        """
        alpha = (roughness * roughness).squeeze(-1)
        
        # sample half vector
        theta = (1-sample2[...,0])/((sample2[...,0]*(alpha*alpha-1)+1))
        theta = torch.acos(theta.sqrt())

        phi = 2*math.pi*sample2[...,1]
        wh = angle2xyz(theta,phi)

        # half vector to wi
        Nmat = get_normal_space(normal)
        wh = (wh[:,None]@Nmat.permute(0,2,1)).squeeze(1)
        wi = 2*(wo*wh).sum(-1,keepdim=True)*wh-wo
        wi = NF.normalize(wi,dim=-1)
        return wi

    def compute_svbrdf_pdf(self, albedo, roughness, metallic, wi, wo, normal):
        h = NF.normalize(wi+wo,dim=-1)
        NoL = (wi*normal).sum(-1,keepdim=True).relu()
        NoV = (wo*normal).sum(-1,keepdim=True).relu()
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()

        D = D_GGX(NoH,roughness)
        pdf_spec = D.data/(4*VoH.clamp_min(1e-4))*NoH
        pdf_diff = NoL/math.pi
        pdf = 0.5*pdf_spec + 0.5*pdf_diff

        kd = albedo*(1-metallic)
        ks = 0.04*(1-metallic) + albedo*metallic

        G = G_Smith(NoV,NoL,roughness)
        F = fresnelSchlick(VoH,ks)
        brdf_diff = kd/math.pi*NoL
        brdf_spec = D*G*F/4.0*NoL

        brdf = brdf_diff + brdf_spec

        return brdf, pdf

    def compute_anisotropic_svbrdf_pdf(self,
                                    diffuse, ao, ax, ay, metallic, ior,
                                    wi, wo,
                                    normal, tangent):
        """
        Inputs:
            diffuse  : [N,3] Diffuse (base-color)
            ao       : [N,1] Ambient Occlusion
            ax, ay   : [N,1] Directional roughness
            metallic : [N,1] Metallic factor
            ior      : [N,1] Specular IOR
            wi, wo   : [N,3] Incident & outgoing directions
            normal   : [N,3] Surface normal
            tangent  : [N,3] Tangent vector (bitangent computed internally)
        Returns:
            brdf [N,3] , pdf [N,1]
        """
        B = NF.normalize(torch.cross(normal, tangent, dim=-1), dim=-1)
        h = NF.normalize(wi + wo, dim=-1)

        NoL = (wi * normal).sum(-1, keepdim=True).clamp_min(0.0)
        NoV = (wo * normal).sum(-1, keepdim=True).clamp_min(0.0)
        VoH = (wo * h).sum(-1, keepdim=True).clamp_min(1e-4)
        NoH = (normal * h).sum(-1, keepdim=True).clamp_min(1e-4)

        # --- specular distribution and geometry -------------------
        D = D_GGX_aniso(h, normal, tangent, B, ax, ay)
        G = G_Smith_aniso(wi, wo, normal, tangent, B, ax, ay)

        # --- Fresnel base reflectance -----------------------------
        F0_dielectric = ((ior - 1) / (ior + 1)).pow(2)
        F0 = F0_dielectric * (1 - metallic) + diffuse * metallic
        F = fresnelSchlick(VoH, F0)

        # --- Diffuse and Specular reflectances --------------------
        kd = diffuse * (1 - metallic) * ao  # apply AO only to diffuse
        ks = 1.0                            # standard GGX specular strength (no scaling)

        # --- BRDF computation -------------------------------------
        brdf_spec = ks * (D * G * F) / (4.0 * NoL * NoV + 1e-6)
        brdf_diff = kd / math.pi
        brdf = brdf_spec + brdf_diff

        # --- PDF (half diffuse, half specular) --------------------
        pdf_spec = D * NoH / (4.0 * VoH)
        pdf_diff = NoL / math.pi
        pdf = 0.5 * (pdf_spec + pdf_diff)

        return brdf, pdf

    

    def eval_brdf(self, params, pos, wi, wo, normal,tangent,uv, latent=None, batch_mask=None):
        #print("using pbr model")
        wi=local2world(wi,tangent,normal)
        wo=local2world(wo,tangent,normal)
        #NoL=wi[:,2:3].repeat(1, 3)
        #NoV=wo[:,2:3].repeat(1, 3)
        NoL = (wi*normal).sum(-1, keepdim=True)
        NoV = (wo*normal).sum(-1, keepdim=True)

        valid_geometry = (NoL > 0) & (NoV > 0)
        
        if not valid_geometry.any():
            return torch.zeros_like(wi), torch.zeros(wi.shape[0], 1, device=wi.device)
        radius = 0.2
        factor = self.scale_factor
        params = self.pbr_texture
        H, W = params.shape[1], params.shape[2]
        crop_h = int((H - H * factor) // 2)
        crop_w = int((W - W * factor) // 2)
        new_h = int(H * factor)
        new_w = int(W * factor)
        params = params[:, crop_h:crop_h+new_h, crop_w:crop_w+new_w, :]
        #uv = compute_uv(pos, 0.8, 0.8)
        # Step 1: TBN frame
        #T, B, N_geo = compute_tbn(pos, uv, 0.8, 0.8)
        T=tangent
        B=torch.cross(normal,tangent, dim=-1)
        N_geo=normal
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
            base_roughness = roughness       # Clamp roughness to valid range
            
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
            
            # Evaluate anisotropic BRDF
            brdf, pdf = self.compute_anisotropic_svbrdf_pdf(
                diffuse, ao, ax, ay, metallic, ior,
                wi, wo, n_world, T_ortho
            )
            
        else:
            arm = sampled_texture[:, 6:9]      # ARM channels
            color = sampled_texture[:, 0:3]    # Color channels
            normal_local = sampled_texture[:, 10:13]  # Normal channels
            albedo, roughness, metallic = arm[:, 0:1], arm[:, 1:2], arm[:, 2:3]

            #roughness[:] = 0.2
            #metallic[:] = 0.8
            # Step 4: Transform local normal to world
            n_world = local_to_world_normal(normal_local, T, B, N_geo)

            # Evaluate BRDF with mapped parameters
            if self.normal_map:
                brdf, pdf = self.compute_svbrdf_pdf(albedo, roughness, metallic, wi, wo, n_world)
            else:
                brdf, pdf = self.compute_svbrdf_pdf(albedo, roughness, metallic, wi, wo, normal)
            brdf = color * brdf
            
        return brdf, pdf
    
    def sample_brdf(self, params, pos, sample1, sample2, wo, normal, latent=None, batch_mask=None):
        """ TODO """
        B = sample1.shape[0]
        device = sample1.device

        # Compute blending mask
        mask = perlin_mask(pos, scale=self.perlin_scale, randomness=self.perlin_randomness).view(-1)

        # Sample diffuse directions (same across both materials)
        wi_diffuse = self.diffuse_sampler(sample2, normal)

        # Sample specular directions for both materials
        roughness0 = params['roughness'][:, 0]
        roughness1 = params['roughness'][:, 1]
        wi_spec0 = self.specular_sampler(sample2, roughness0, wo, normal)
        wi_spec1 = self.specular_sampler(sample2, roughness1, wo, normal)

        # Mix sampled directions based on mask
        wi_spec = wi_spec1

        # Choose whether to use diffuse or specular
        select_diff = sample1 > 0.5
        wi = torch.zeros(B, 3, device=device)
        wi[select_diff] = wi_diffuse[select_diff]
        wi[~select_diff] = wi_spec[~select_diff]

        # Evaluate blended BRDF
        brdf, pdf = self.eval_brdf(params, pos, wi, wo, normal, latent, batch_mask)
        brdf_weight = torch.where(pdf > 0, brdf / pdf, torch.zeros_like(brdf))
        brdf_weight[brdf_weight.isnan()] = 0

        return wi, pdf, brdf_weight


def _load_axf_brdf_core():
    """Import axf_brdf_core built from axf/eval (CMake + pybind11).

    Resolution order:
    1. ``AXF_BRDF_PY_PATH`` / ``AXF_EVAL_BUILD`` (directories containing the .so)
    2. ``<this file>/../../../axf/eval/build`` (sibling ``axf`` next to the repository)
    3. Walk parents of ``brdf_plugin/material`` for ``axf/eval/build``
    4. If a matching ``axf_brdf_core*.so`` is found, load via ``spec_from_file_location`` (works without sys.path).

    Rebuild for your interpreter::

        cd /path/to/axf/eval/build
        cmake .. -DPython3_EXECUTABLE="$(which python)" -DBUILD_AXF_PYBIND=ON
        cmake --build .
    """
    import glob
    import importlib.util
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []

    def _add(p):
        p = os.path.abspath(os.path.expanduser(p))
        if p and os.path.isdir(p) and p not in candidates:
            candidates.append(p)

    for key in ("AXF_BRDF_PY_PATH", "AXF_EVAL_BUILD"):
        raw = os.environ.get(key, "")
        if not raw:
            continue
        for part in raw.split(os.pathsep):
            _add(part.strip())

    _add(os.path.join(here, "../../../axf/eval/build"))
    p = here
    for _ in range(8):
        _add(os.path.join(p, "axf", "eval", "build"))
        p = os.path.dirname(p)

    ver_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    tried = []
    for d in candidates:
        tried.append(d)
        pattern = os.path.join(d, "axf_brdf_core*.so")
        matches = sorted(glob.glob(pattern))
        if not matches:
            continue
        preferred = [m for m in matches if ver_tag in os.path.basename(m)]
        so_path = preferred[0] if preferred else matches[-1]
        spec = importlib.util.spec_from_file_location("axf_brdf_core", so_path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            return mod
        except Exception as ex:
            raise ImportError(
                f"Found {so_path} but failed to load (wrong Python version?). "
                f"Rebuild with: cmake -DPython3_EXECUTABLE=$(which python) .. && cmake --build .\n"
                f"Original error: {ex}"
            ) from ex

    for d in candidates:
        if d not in sys.path:
            sys.path.insert(0, d)
    try:
        import importlib

        return importlib.import_module("axf_brdf_core")
    except ModuleNotFoundError as e:
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        nl = "\n"
        raise ModuleNotFoundError(
            f"No axf_brdf_core extension for Python {py_ver}. "
            f"Tried directories:{nl}{nl.join(tried) or '(none)'}{nl}"
            f"Set AXF_BRDF_PY_PATH to the folder that contains axf_brdf_core*.so, "
            f"or build: cd axf/eval/build && cmake .. -DPython3_EXECUTABLE=$(which python) "
            f"-DBUILD_AXF_PYBIND=ON && cmake --build ."
        ) from e


class AXFBRDF(nn.Module):
    """
    Measured AxF BRDF via CPUDecoder::eval (same path as axf/eval/eval.cpp).

    wi, wo must be in local tangent space with N = +Z (same convention as SvPBRBRDF
    *before* local2world). The C++ SDK applies normalization and returns linear RGB
    including the cosine term (BRDF * <N, L>).

    Requires the Python extension ``axf_brdf_core`` (build axf/eval with BUILD_AXF_PYBIND=ON)
    and ``LD_LIBRARY_PATH`` / rpath so ``libAxFDecoding.so`` can load.
    """

    def __init__(self, cfg,axf_path):
        super().__init__()
        self.axf_path = axf_path
        if not self.axf_path:
            raise ValueError("AXFBRDF: config must set axf_path to an .axf file")
        material_id = getattr(cfg, "material_id", "") or ""
        core = _load_axf_brdf_core()
        self._core = core.AXFBRDFCore(axf_path, material_id)

    def eval_brdf(self, wi, wo,  uv, latent=None, batch_mask=None):
        # AxF expects local directions (tangent, bitangent, normal) = (x, y, z); do not local2world.
        NoL = wi[:, 2:3]
        NoV = wo[:, 2:3]
        valid = (NoL > 0) & (NoV > 0)
        if not valid.any():
            return torch.zeros_like(wi), torch.zeros(wi.shape[0], 1, device=wi.device, dtype=wi.dtype)

        wi_np = wi.detach().cpu().numpy().astype(np.float32)
        wo_np = wo.detach().cpu().numpy().astype(np.float32)
        uv_np = uv.detach().cpu().numpy().astype(np.float32)
        rgb_np = self._core.eval_brdf_batch(wi_np, wo_np, uv_np)
        rgb = torch.from_numpy(np.asarray(rgb_np, dtype=np.float32)).to(
            device=wi.device, dtype=wi.dtype
        )
        valid_3 = valid.expand(-1, 3)
        rgb = torch.where(valid_3, rgb, torch.zeros_like(rgb))
        # Cosine-weighted hemisphere PDF proxy (training only; not AxF importance sampling)
        pdf = NoL / math.pi
        pdf = torch.where(valid, pdf, torch.zeros_like(pdf))
        return rgb, pdf

    def sample_brdf(self, params, pos, sample1, sample2, wo, normal, latent=None, batch_mask=None):
        raise NotImplementedError(
            "AXFBRDF.sample_brdf is not implemented; use explicit wi sampling or extend with a proposal PDF."
        )
