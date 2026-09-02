import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import os
import yaml
import torch.nn.functional as NF

# Root of the rendering/ package: material yamls live in <root>/configs/material.
_RENDER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_checkpoint_io():
    """Load the shared fail-closed checkpoint policy (training/utils/checkpoint_io.py).

    rendering/ runs with cwd=rendering and never imports the training package,
    so the module is loaded by file path — the same sys.path-free technique
    svpbr.py uses for axf_brdf_core — instead of duplicating the policy here.
    The module is pure torch, so it works unchanged in the rendering env.
    """
    import importlib.util
    import sys
    name = "robocloth_checkpoint_io"
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(os.path.dirname(_RENDER_ROOT), "training", "utils", "checkpoint_io.py")
    if not os.path.isfile(path):
        raise ImportError(
            f"shared checkpoint loader not found at {path}; rendering/ must stay "
            "next to training/ inside the RoboCloth repository")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


checkpoint_io = _import_checkpoint_io()

# Material-namespace keys that training writes but the renderer deliberately
# does not model — the ONLY tensors a checkpoint may carry that the strict
# load below does not require the model to have. Trainer-side state
# (gt_material.*, emitter.*, pan_weights) lives outside 'material.' and is
# split off before loading, so it is not listed here.
#   material_offset_tensor: integer per-material offsets into the multi-
#   material latent bank, registered by the training-side Bonn classes in
#   multi-material mode (training/models/neural_brdf_refactored.py,
#   BonnLatentBRDF / BonnPBRLatentBRDF._load_point_metadata); the renderer
#   wrappers have no such buffer and never index by material id.
# If a checkpoint family ever needs another one, add its key prefix HERE
# (explicitly) — a non-strict load is never the answer (it rendered
# incompatible ckpts as noise).
IGNORED_MATERIAL_KEY_PREFIXES = ('material_offset_tensor',)

from brdf_plugin.material.anisotropicLatent import AnisotropicLatentTexturedModel,MERLBRDF,MERLInterface,MerlTorch
from brdf_plugin.material.isotropicLatent import LatentTexturedModel
from brdf_plugin.material.svpbr import SvPBRBRDF,AXFBRDF
from brdf_plugin.material.compressed import CompressedModel
from brdf_plugin.utils.cuda_manage import print_cuda_memory_info
from brdf_plugin.material.moeLatent import MoeLatentModel
from brdf_plugin.material.ubo_latent import UBOLatentBRDF, UBOLatentBRDFDebugCos, UBOLatentBRDFDebugBinary
from brdf_plugin.material.ubo_pbr import UBOPBRLatentBRDF
from brdf_plugin.material.anisotropic_pbr import LearnablePBRTexturedModel
from brdf_plugin.material.bonn_latent import BonnLatentBRDF, BonnPBRLatentBRDF
from brdf_plugin.material.BTF import UBOBTFInterpolator

#from mitsuba.render import Integrator, SurfaceInteraction3f, BSDFContext, Ray3f

# from pytorch_model.bvpnet import SingleBVPNet

def create_anisotropic_model(model_path=None, material_type="AnisotropicLatentTexturedModel",
                             use_btf=False, btf_path=""):
    """
    Create or load AnisotropicLatentTexturedModel

    Args:
        model_path: Model weight file path, if None create new model
        material_type: Material type key (matches a yaml in configs/material/)
        use_btf: If True, return a ground-truth UBOBTFInterpolator instead of the neural decoder
        btf_path: Path to a UBO ``.btf`` file (used only when use_btf=True)

    Returns:
        model: BRDF model instance (either a torch nn.Module or UBOBTFInterpolator)
    """
    if use_btf:
        if not btf_path or not os.path.exists(btf_path):
            raise FileNotFoundError(f"use_btf=True but btf_path is missing/invalid: {btf_path!r}")
        print(f"Using ground-truth BTF: {btf_path}")
        return UBOBTFInterpolator(btf_path)

    # Fail closed before any GPU allocation: a requested checkpoint must exist.
    # (model_path=None is the explicit random-init path via use_anisotropic;
    # "" is the PBR-without-weights convention — neither names a checkpoint.)
    if model_path and not os.path.isfile(model_path):
        raise FileNotFoundError(f"checkpoint not found: {model_path}")

    config_path = os.path.join(_RENDER_ROOT, "configs", "material", f"{material_type}.yaml")
    print(f"Loading configuration from config file: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Create configuration class with recursive dict conversion
    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                if key != 'type':  # Skip type field
                    # Recursively convert nested dicts to Config objects
                    if isinstance(value, dict):
                        setattr(self, key, Config(value))
                    else:
                        setattr(self, key, value)
    
    config = Config(config_dict)
    #print(f"Model configuration: latent_dim={config.latent_dim}, texture_resolution={config.texture_resolution}")
    #print(f"hidden_layers={config.hidden_layers}, activation={config.activation}")
    
    if material_type == "AnisotropicLatentTexturedModel":
        model = AnisotropicLatentTexturedModel(config).cuda()
    elif material_type == "LatentTexturedModel":
        model = LatentTexturedModel(config)
    elif material_type == "SvPBRBRDF":
        model = SvPBRBRDF(config)
    elif material_type == "CompressedModel":
        model = CompressedModel(config)
    elif material_type == "MoeLatentModel":
        model = MoeLatentModel(config).cuda()
    elif material_type == "MERLBRDF":
        model = MERLBRDF(config).cuda()
    elif material_type == "AXFBRDF":
        axf_path="/absolute/path/to/axf/mat0001.axf"
        model = AXFBRDF(config,axf_path).cuda()
    elif material_type == "UBOLatentBRDF":
        model = UBOLatentBRDF(config).cuda()
    elif material_type == "UBOLatentBRDFDebugCos":
        model = UBOLatentBRDFDebugCos(config).cuda()
    elif material_type == "UBOLatentBRDFDebugBinary":
        model = UBOLatentBRDFDebugBinary(config).cuda()
    elif material_type == "UBOPBRLatentBRDF":
        model = UBOPBRLatentBRDF(config).cuda()
    elif material_type == "LearnablePBRTexturedModel":
        model = LearnablePBRTexturedModel(config).cuda()
    elif material_type == "BonnLatentBRDF":
        model = BonnLatentBRDF(config).cuda()
    elif material_type == "BonnPBRLatentBRDF":
        model = BonnPBRLatentBRDF(config).cuda()
    else:
        raise ValueError(f"Unsupported material type: {material_type}")
    
    if model_path=="":
        print("Using PBR model")
        return model.cuda()

    if model_path is None:
        print("Using randomly initialized AnisotropicLatentTexturedModel")
        model.eval()
        return model.cuda()

    # Load pretrained weights. Fail closed (beta item B2): an unreadable file,
    # an unknown format or any missing / unexpected / shape-mismatched tensor
    # raises — there is no non-strict retry and no random-init fallback.
    print(f"Loading pretrained model: {model_path}")
    checkpoint = checkpoint_io.load_checkpoint_file(model_path, map_location='cuda')
    state_dict = checkpoint_io.extract_state_dict(checkpoint, source=model_path)
    print(f"Checkpoint contains {len(state_dict)} keys")

    # Lightning trainer checkpoints hold the whole LightningModule. The BRDF
    # model is the 'material.' sub-tree; the other keys (gt_material.*,
    # emitter.*, per-trainer buffers such as pan_weights) are trainer state,
    # not material weights — they are split off explicitly and reported.
    # Bare model checkpoints keep the legacy handling: an optional 'model.'
    # prefix is stripped and everything else is loaded as-is.
    material_state_dict, trainer_state_dict = checkpoint_io.split_namespace(state_dict, 'material.')
    if material_state_dict:
        print(f"Found {len(material_state_dict)} keys with 'material.' prefix; skipping "
              f"{len(trainer_state_dict)} trainer-state keys: {sorted(trainer_state_dict)[:8]}"
              + (" ..." if len(trainer_state_dict) > 8 else ""))
        state_dict = material_state_dict
    else:
        state_dict = {(k[6:] if k.startswith('model.') else k): v for k, v in state_dict.items()}
    print(f"Example keys: {list(state_dict.keys())[:5]}")

    # Bonn models store a per-material H*W latent bank (resolution varies per
    # material). Resize the embedding to match the checkpoint BEFORE loading,
    # using H,W from bonn_point_metadata.json — otherwise the bank cannot load
    # (size mismatch) and the strict load below rejects the checkpoint.
    # Material id is the parent folder of the ckpt (.../Bonn/<id>/X.ckpt).
    if material_type in ("BonnLatentBRDF", "BonnPBRLatentBRDF"):
        import json
        mat_id = os.path.basename(os.path.dirname(model_path))
        meta_path = os.path.join(
            getattr(config, "bonn_dataset_folder", "/absolute/path/to/Bonn_val"),
            "bonn_point_metadata.json")
        with open(meta_path, "r") as mf:
            _meta = json.load(mf)
        if str(mat_id) not in _meta:
            raise KeyError(f"Bonn material '{mat_id}' not in {meta_path}")
        H, W = int(_meta[str(mat_id)]["H"]), int(_meta[str(mat_id)]["W"])
        if "point_latent_bank.weight" not in state_dict:
            raise KeyError(f"Bonn mat {mat_id}: ckpt has no point_latent_bank.weight")
        bank_n = state_dict["point_latent_bank.weight"].shape[0]
        if H * W != bank_n:
            raise ValueError(
                f"Bonn mat {mat_id}: metadata H*W={H*W} != ckpt bank {bank_n}")
        model.set_grid(H, W)
        print(f"[Bonn] mat {mat_id}: latent grid {H}x{W} = {bank_n} points")

    report = checkpoint_io.load_state_dict_strict(
        model, state_dict, ignore_prefixes=IGNORED_MATERIAL_KEY_PREFIXES, source=model_path)
    print(report.summary())
    print("Model weights loaded successfully")

    model.eval()

    return model.cuda()

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=3):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        #self._init_weights()
        
    def _init_weights(self):
    # Initialize all weights and biases to fixed values (e.g., 0.1 and 0.0)
        init.constant_(self.fc1.weight, 10)
        init.constant_(self.fc1.bias, 0.0)

        init.constant_(self.fc2.weight, 10)
        init.constant_(self.fc2.bias, 0.0)

        init.constant_(self.fc3.weight, 10)
        init.constant_(self.fc3.bias, 0.0)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def _prop_bool(props, name, default):
    """Read a boolean BSDF property, accepting XML string values ("true"/"false")."""
    v = props.get(name, default)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


class MLPBRDF(mi.BSDF):
    """Neural-BRDF Mitsuba plugin ("mlpbrdf").

    Properties:
        model_path        checkpoint to load (see ``create_anisotropic_model``)
        material_type     wrapper class key (yaml in configs/material/)
        two_sided         True (default): mirror back-face hits so an open drape's
                          interior renders fabric instead of black. False: mask
                          back faces to zero (legacy one-sided behavior).
        uv_tiling         texture repeats across the mesh UV (default 5.0)
        uv_inset          apply the legacy uv = (uv*0.8 + 0.1) % 1 remap after
                          tiling (default False; the teaser scenes used True)
        grazing_mask_deg  one-sided mode only: zero the BRDF where the light
                          direction exceeds this angle from the normal (0 = off)
        apply_cosine_at_eval  False for checkpoints whose training baked cos
                          into the network output (read by some wrappers)
        use_btf/btf_path  evaluate a ground-truth UBO BTF instead of a network
    """

    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        self.eta = props.get('eta', 1.33)

        use_btf = _prop_bool(props, 'use_btf', False)
        btf_path = props.get('btf_path', '')
        # Whether eval should multiply by NoL.  False for checkpoints whose
        # training already baked cos into the network output (bonn-style).
        apply_cosine_at_eval = _prop_bool(props, 'apply_cosine_at_eval', True)
        # UV tiling: texture repeats this many times across the mesh UV
        # (eval does uv = (uv * uv_tiling) % 1). 5.0 is the established baseline.
        self.uv_tiling = float(props.get('uv_tiling', 5.0))
        # Legacy teaser-scene UV remap applied after tiling: uv = (uv*0.8+0.1) % 1.
        self.uv_inset = _prop_bool(props, 'uv_inset', False)
        # Native two-sided evaluation (see eval below). two_sided=False
        # reproduces the legacy one-sided path: back faces render black.
        self.two_sided = _prop_bool(props, 'two_sided', True)
        # One-sided mode only: mask brdf to zero where the incoming-light
        # direction (BRDF's wi = Mitsuba's wo) makes an angle > grazing_mask_deg
        # with the local +z (geometric normal). 0 = disabled.
        grazing_deg = float(props.get('grazing_mask_deg', '0') or 0)
        self.grazing_mask_cos = math.cos(math.radians(grazing_deg)) if grazing_deg > 0 else 0.0

        # Check if using AnisotropicLatentTexturedModel
        if props.has_property('model_path'):
            # Load AnisotropicLatentTexturedModel from path
            model_path = props.get('model_path', '')  # Provide default value
            print(f"MLPBRDF: Loading AnisotropicLatentTexturedModel from {model_path}")
            # Get config path from props, use default if not provided
            material_type = props.get('material_type', '')
            self.anisotropic_model = create_anisotropic_model(
                model_path, material_type, use_btf=use_btf, btf_path=btf_path
            )
            # Forward the cos-at-eval flag onto the model so its eval_brdf can read it.
            try:
                self.anisotropic_model.apply_cosine_at_eval = apply_cosine_at_eval
            except (AttributeError, RuntimeError):
                pass
            torch.cuda.empty_cache()
            self.use_anisotropic = True
            print(f"MLPBRDF: AnisotropicLatentTexturedModel loaded (apply_cosine_at_eval={apply_cosine_at_eval})")
        elif props.has_property('use_anisotropic') and props.get('use_anisotropic', False):
            # Use randomly initialized AnisotropicLatentTexturedModel
            print("MLPBRDF: Creating randomly initialized AnisotropicLatentTexturedModel")
            # Get config path from props, use default if not provided
            material_type = props.get('material_type', '')
            self.anisotropic_model = create_anisotropic_model(
                None, material_type, use_btf=use_btf, btf_path=btf_path
            )
            try:
                self.anisotropic_model.apply_cosine_at_eval = apply_cosine_at_eval
            except (AttributeError, RuntimeError):
                pass
            self.use_anisotropic = True
            print("MLPBRDF: AnisotropicLatentTexturedModel created")
        else:
            # Use simple MLP and traditional weights
            self.use_anisotropic = False
            if props.has_property('w1'):
                # Use passed weights
                self.w1 = props['w1']
                self.b1 = props['b1']
                self.w2 = props['w2']
                self.b2 = props['b2']
                self.w3 = props['w3']
                self.b3 = props['b3']
            
            self.model = SimpleMLP().cuda()
            print("MLPBRDF: Using SimpleMLP")
        
        self.m_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide

        #self.set_point_lights(props)

    def __del__(self):
        """
        Destructor: Explicitly release CUDA memory
        """
        try:
            if hasattr(self, 'use_anisotropic') and self.use_anisotropic:
                if hasattr(self, 'anisotropic_model'):
                    print("MLPBRDF: Releasing AnisotropicLatentTexturedModel CUDA memory")
                    # Delete model reference
                    del self.anisotropic_model
            
            if hasattr(self, 'model'):
                # Release SimpleMLP model
                del self.model
            
            # Force clean up PyTorch CUDA cache
            torch.cuda.empty_cache()
            
            # Clean up DrJit cache
            dr.flush_malloc_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            print("MLPBRDF: CUDA memory release completed")
            
        except Exception as e:
            # Exceptions in destructor should not be thrown
            print(f"MLPBRDF destructor warning: {e}")

    def flip(self,si,mask):
        si.wi[mask] = -si.wi[mask]
        si.sh_frame.n[mask] = -si.sh_frame.n[mask]
        si.sh_frame.s[mask] = -si.sh_frame.s[mask]  # flip tangent
        si.sh_frame.t[mask] = -si.sh_frame.t[mask]  # flip bitangent

    def sample(self, ctx, si, sample1, sample2, active):
        # 1. Determine the number of rays/samples in the current batch
        batch_size = dr.width(sample2)

        # 2. Generate samples in PyTorch (e.g., Uniform 0 to 1)
        # Ensure the device matches your Mitsuba variant (usually 'cuda')
        torch_samples = torch.rand((batch_size, 2), device='cuda', dtype=torch.float32)

        # 3. Transform PyTorch tensor back to Mitsuba type (mi.Point2f)
        # We split the tensor columns and wrap them into Dr.Jit arrays
        sample2_torch=mi.Vector2f(mi.Float(torch_samples[:, 0].contiguous()), mi.Float(torch_samples[:, 1].contiguous()))
        # 4. Proceed with standard Mitsuba warping
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        active &= cos_theta_i > 0
        #self.flip(si, ~active)

        bs = mi.BSDFSample3f()
        # Now using the PyTorch-generated samples
        bs.wo = mi.warp.square_to_cosine_hemisphere(sample2_torch)
        #bs.wo=si.wi
        bs.pdf = mi.warp.square_to_cosine_hemisphere_pdf(bs.wo)
        #bs.pdf=0.1
        bs.eta = 1.0
        bs.sampled_type = +mi.BSDFFlags.GlossyReflection
        bs.sampled_component = 0

        brdf_res=self.eval(ctx, si, bs.wo, active)
        #print("brdf_res:", brdf_res)
        value = brdf_res / bs.pdf
        
        # Only clear cache after large batch sampling
        sample_count = dr.width(si.wi.x) if hasattr(dr, 'width') else 1
        if sample_count > 1500:  # Medium threshold, as sample might have medium frequency
            torch.cuda.empty_cache()
        return ( bs, dr.select(active & (bs.pdf > 0.0), value, mi.Vector3f(0)) )
        #return ( bs, value )   
        
    def pdf(self, ctx, si, wo, active=True):
        
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        pdf = dr.clamp(mi.Frame3f.cos_theta(wo),1e-5,1)*1/dr.pi
        # pdf = mi.warp.square_to_beckmann_pdf(wo,0.08)

        return dr.select((cos_theta_i > 0.0) & (cos_theta_o > 0.0), pdf, 0.0)
    
    
    def eval_pdf(self, ctx, si, wo, active=True):
        return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)

    def eval(self, ctx, si, wo, active):
        # --- Native two-sided handling (opaque fabric looks identical from
        # both faces) -------------------------------------------------------
        # Decide the hit side from the GEOMETRIC shading-frame cosine BEFORE
        # touching anything. cos_theta(si.wi) > 0 iff the ray hit the FRONT
        # face; < 0 iff it hit the BACK (the hollow interior of an open drape).
        # We use the geometric local +z, NOT the model's predicted normal: the
        # predicted normal is a UV-space texture and can flip into the back
        # hemisphere at grazing angles on curved geometry, masking valid
        # front-facing pixels to black.
        #
        # A valid OPAQUE reflection requires wi and wo on the SAME hemisphere.
        # For back-side lanes we mirror the shading frame + wi + wo so the model
        # always evaluates a front-facing configuration, then return that same
        # felt value (the fabric is the same on both faces). FRONT-side lanes are
        # left untouched, so the front shading is byte-identical to the legacy
        # one-sided path. This is done INSIDE the BSDF on purpose: Mitsuba's stock
        # `twosided` wrapper re-dispatches through a canonicalized frame that
        # silently DIMS the front-side response of this custom neural/BTF BSDF
        # (verified on the convex sphere, which has no back-faces yet darkened).
        #
        # two_sided=False skips the mirroring and instead masks back faces (and
        # optionally grazing light directions) to black — the legacy behavior
        # the pre-authored teaser scenes were tuned against.
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        if self.two_sided:
            mask = (cos_theta_i * cos_theta_o) > 0.0
            back = cos_theta_i < 0.0
            self.flip(si, back)              # mirror back-side lanes (in place)
            wo = dr.select(back, -wo, wo)    # mirror wo on the same lanes (new array)
        else:
            # grazing_mask_cos >= 0 subsumes the back-face mask; when set above
            # 0 it also masks the trained-decoder's grazing-angle predictions.
            thresh = max(self.grazing_mask_cos, 0.0)
            mask = (cos_theta_i > 0.0) & (cos_theta_o > thresh)

        if self.use_anisotropic:
            value, _cos_i_pred, _cos_o_pred = self.eval_anisotropic_mlp(si, wo)
        else:
            value = self.mlp(si, wo)

        if self.two_sided:
            self.flip(si, back)  # restore si for the caller (flip is its own inverse)

        # Only clear cache after large batch evaluation
        sample_count = dr.width(si.wi.x) if hasattr(dr, 'width') else 1
        if sample_count > 2000:  # Higher threshold, as eval might be called frequently
            torch.cuda.empty_cache()
        return dr.select(mask, value, mi.Vector3f(0.0))
    
    def eval_anisotropic_mlp(self, si, wo):
        """Evaluate BRDF using AnisotropicLatentTexturedModel"""
        # Get sample count
        sample_count = dr.width(si.wi.x)

        # Immediately extract all required data and convert to independent PyTorch tensors
        # This allows releasing references to DrJit arrays
        
        # Mitsuba 3 returns Vector*.torch() in channels-first layout (K, N);
        # the model expects batch-first (N, K), so we transpose at the boundary.
        def _bf(t):
            return t.T.contiguous() if t.dim() == 2 and t.shape[0] in (2, 3) and t.shape[0] != t.shape[1] else t

        # 1. World coordinate position - create independent copy
        pos_torch = _bf(si.p.torch().detach())

        # 2. Geometric normal (world coordinate) - create independent copy
        normal_torch = NF.normalize(_bf(si.sh_frame.n.torch().detach()), dim=-1)
        tangent_torch = NF.normalize(_bf(si.sh_frame.s.torch().detach()), dim=-1)
        bitangent_torch = NF.normalize(_bf(si.sh_frame.t.torch().detach()), dim=-1)
        TBN_torch = torch.stack([tangent_torch, bitangent_torch, normal_torch], dim=-1)  # [B, 3, 3]

        # 3. UV texture coordinates - create independent copy
        uv_torch = _bf(si.uv.torch().detach())
        uv_torch = (uv_torch * self.uv_tiling) % 1.0
        if self.uv_inset:
            uv_torch = (uv_torch * 0.8 + 0.1) % 1.0

        # 4. Incident direction (world coordinate) - create independent copy
        wi_torch = _bf(si.wi.torch().detach())

        # 5. Exit direction (world coordinate) - create independent copy
        wo_torch = _bf(wo.torch().detach())
        
        # Now we have independent tensor copies, we can release some memory in eval_brdf
        # Force Python garbage collection, clean up any temporary DrJit array references
        #normal_torch = double_sided(wo_torch,normal_torch)
        del si
        
        # Evaluate BRDF using AnisotropicLatentTexturedModel
        #flip_mask=wi_torch[:,2]<0
        #print("flip_mask",torch.sum(flip_mask))
        #wi_torch[flip_mask]=-wi_torch[flip_mask]
        with torch.no_grad():
            # NOTE: wi_torch / wo_torch are intentionally passed swapped here
            # (the neural model's "wi" slot receives our wo_torch and vice versa).
            brdf, _, _, _, cos_i_internal, cos_o_internal = self.anisotropic_model.eval_brdf(
                {}, pos_torch.cuda(), wo_torch.cuda(), wi_torch.cuda(),
                normal_torch.cuda(), uv_torch.cuda(), TBN_torch.cuda()
            )
            #brdf,_ = self.anisotropic_model.eval_brdf(wo_torch.cuda(), wi_torch.cuda(),uv_torch.cuda())

        # Keep tensors on the active device. Under cuda_ad_rgb, mi.Float wraps
        # a CUDA torch tensor in-place (no copy); under llvm_ad_rgb mi.Float
        # auto-host-copies, so calling .cpu() here is redundant work either way.
        brdf = brdf.detach()
        # Swap back: model's internal cos_theta_i used its swapped wi (= our wo), so
        # our external cos_theta_i corresponds to the model's internal cos_theta_o.
        cos_theta_i_torch = cos_o_internal.squeeze(-1).detach()
        cos_theta_o_torch = cos_i_internal.squeeze(-1).detach()

        del pos_torch, normal_torch, uv_torch, wi_torch, wo_torch, TBN_torch

        # Convert directly from PyTorch to DrJit (stay on CUDA)

        result=mi.Vector3f(
            mi.Float(brdf[:, 0].contiguous()),
            mi.Float(brdf[:, 1].contiguous()),
            mi.Float(brdf[:, 2].contiguous())
        )
        cos_theta_i_mi = mi.Float(cos_theta_i_torch.contiguous())
        cos_theta_o_mi = mi.Float(cos_theta_o_torch.contiguous())
        # Delete BRDF tensor reference

        del brdf, cos_theta_i_torch, cos_theta_o_torch
        torch.cuda.empty_cache()
        # Clean up DrJit cache
        dr.flush_malloc_cache()

        # Force garbage collection
        import gc
        gc.collect()
        return result, cos_theta_i_mi, cos_theta_o_mi

    def mlp(self, si, wo):
        # Prepare input tensor: manually construct tensor from wi and wo components
        x = dr.zeros(mi.TensorXf, 6)
        x[0] = si.wi.x
        x[1] = si.wi.y 
        x[2] = si.wi.z
        x[3] = wo.x
        x[4] = wo.y
        x[5] = wo.z
        x=x.torch().cuda()
        out=self.model(x)
        return mi.Vector3f(float(out[0]), float(out[1]), float(out[2]))
