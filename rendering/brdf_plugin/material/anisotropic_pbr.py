"""
LearnablePBRTexturedModel — Disney-PBR variant of AnisotropicLatentTexturedModel
for the 'Ours' (real-captured) dataset.

It reuses AnisotropicLatentTexturedModel's machinery verbatim — the latent
texture (grid_sample), the NeuralGeometry UV-offset MLP, predicted-frame
extraction and world->local transform — but replaces the learned MLP
``BRDFDecoder`` with the analytic Disney ``PBRDecoder`` (from ubo_pbr.py).

This mirrors the training code's ``training/models/neural_brdf_refactored.LearnablePBRTexturedModel``
(trained via scripts/jobs/run_stage2_ours_PBR.sh: material=learnable_pbr_texture_model,
disney=True, latent_dim=24, texture_resolution=2048, neural_geometry.factor=0.08),
so checkpoints from checkpoints/stage2/RoboCloth/<mat>/PBR_epoch*.ckpt (released layout)
load cleanly.

Latent-texture channel layout (total 34 for disney + predict_frame + geom16):
    [0:12]   Disney SVBRDF params (parsed by PBRDecoder):
             baseColor(3), ao(1), roughness(1), metallic(1), ior(1),
             aniso_strength(1), aniso_rot(1), specularTint(1), sheen(1), sheenTint(1)
    [12:28]  16-D neural-geometry latent
    [28:34]  predicted frame: normal(3) + tangent(3)

eval_brdf follows the convention used by anisotropicLatent.py /
ubo_pbr.py: it returns the cosine-weighted brdf and the 6-tuple
(brdf, predicted_normal, pdf, uv_offset, cos_theta_i_pred, cos_theta_o_pred)
that brdf_plugin/mlp.py expects, so the renderer config uses
apply_cosine_at_eval: false (the cosine is applied here).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as NF
from pytorch_lightning import LightningModule

from brdf_plugin.material.anisotropicLatent import LatentTexture, NeuralGeometry
from brdf_plugin.material.ubo_pbr import PBRDecoder


class LearnablePBRTexturedModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.predict_frame = cfg.predict_frame
        self.gt_frame = getattr(cfg, 'gt_frame', False)
        self.anisotropic = getattr(cfg, 'anisotropic', False)
        self.disney = getattr(cfg, 'disney', False)
        self.Gaussian_blur = cfg.Gaussian_blur
        self.learnable_factor = cfg.learnable_factor
        self.soft_constraint = getattr(cfg, 'soft_constraint', True)
        self.init_std = 0.1

        # Scalar learnable factor (matches ckpt: material.factor has shape ()).
        if self.learnable_factor:
            self.factor = nn.Parameter(torch.tensor(1.0))

        # Neural-geometry settings
        self.neural_geometry_enabled = cfg.neural_geometry.enable
        self.geometry_latent_dim = cfg.neural_geometry.latent_dim if self.neural_geometry_enabled else 0
        self.local_wi_wo = cfg.neural_geometry.local_wi_wo if self.neural_geometry_enabled else False
        self.neural_geometry_pos_enc = cfg.neural_geometry.positional_encoding if self.neural_geometry_enabled else False
        self.recompute_frame = cfg.neural_geometry.recompute_frame if self.neural_geometry_enabled else False
        self.neural_geometry_factor = cfg.neural_geometry.factor if self.neural_geometry_enabled else 0.4
        self.use_neural_geometry_at_eval = (
            getattr(cfg.neural_geometry, 'use_at_eval', True) if self.neural_geometry_enabled else False
        )
        # Inference-only toggle (parity with anisotropicLatent / ubo_pbr).
        self.use_predicted_frame_at_eval = getattr(cfg, 'use_predicted_frame_at_eval', True)

        # Disney => 12 SVBRDF params (fixed by the disney/anisotropic flags).
        if self.disney:
            brdf_latent_dim = 12
        elif self.anisotropic:
            brdf_latent_dim = 9
        else:
            brdf_latent_dim = 6
        self.brdf_latent_dim = brdf_latent_dim

        total_latent_dim = brdf_latent_dim
        if self.predict_frame:
            total_latent_dim += 6
        if self.neural_geometry_enabled:
            total_latent_dim += self.geometry_latent_dim
        self.total_latent_dim = total_latent_dim

        # 1. Latent texture (same module/param name as the anisotropic model).
        self.texture_resolution = getattr(cfg, 'texture_resolution', 256)
        blur_config = {'blur_sigma0': 2.0, 'blur_half_life': 3333} if self.Gaussian_blur else None
        self.latent_texture = LatentTexture(
            resolution=self.texture_resolution,
            latent_dim=total_latent_dim,
            predict_frame=self.predict_frame,
            init_std=self.init_std,
            blur_config=blur_config,
        )

        # 2. Analytic Disney decoder (registers canonical_{normal,tangent,bitangent}
        #    buffers -> matches ckpt keys decoder.canonical_*; no trainable params).
        self.decoder = PBRDecoder(cfg=cfg, soft_constraint=self.soft_constraint)

        # 3. Neural geometry (same module/param names: neural_geometry.mlp.*).
        if self.neural_geometry_enabled:
            self.neural_geometry = NeuralGeometry(
                cfg=cfg.neural_geometry,
                geometry_latent_dim=self.geometry_latent_dim,
                use_local_wi_wo=self.local_wi_wo,
                use_pos_enc=self.neural_geometry_pos_enc,
            )
        else:
            self.neural_geometry = None

        self.use_latent_bank = getattr(cfg, 'use_latent_bank', False)
        self.latent_dim = self.brdf_latent_dim  # parity w/ trainer-visualization code
        print("LearnablePBRTexturedModel initialisation complete!")

    # ------------------------------------------------------------------
    # Helpers (identical to AnisotropicLatentTexturedModel)
    # ------------------------------------------------------------------
    def world_to_local(self, v, normal, tangent):
        if tangent is None:
            tangent = torch.cross(
                normal, torch.tensor([0.0, 0.0, 1.0], device=normal.device).expand_as(normal)
            )
        tangent = tangent / (tangent.norm(dim=-1, keepdim=True) + 1e-8)
        bitangent = torch.cross(normal, tangent)
        return torch.stack([
            (v * tangent).sum(dim=-1),
            (v * bitangent).sum(dim=-1),
            (v * normal).sum(dim=-1),
        ], dim=-1)

    def extract_frame_from_latent(self, latent):
        predicted_normal = NF.normalize(latent[..., -6:-3], dim=-1)
        predicted_tangent = NF.normalize(latent[..., -3:], dim=-1)
        predicted_tangent = predicted_tangent - \
            torch.sum(predicted_tangent * predicted_normal, dim=-1, keepdim=True) * predicted_normal
        predicted_tangent = NF.normalize(predicted_tangent, dim=-1)
        return predicted_normal, predicted_tangent

    def _sample_from_texture(self, uv, texture):
        grid_coords = uv * 2.0 - 1.0
        grid_coords = grid_coords.unsqueeze(1).unsqueeze(0)
        latent = NF.grid_sample(
            texture, grid_coords, mode='bilinear', padding_mode='border', align_corners=False
        )
        return latent.squeeze(0).squeeze(-1).transpose(0, 1)

    # ------------------------------------------------------------------
    # Main BRDF evaluation (anisotropic flow + Disney decode + cosine weight)
    # ------------------------------------------------------------------
    def eval_brdf(self, gt_params, pos, wi, wo, normal, uv, TBN, latent=None,
                  batch_mask=None, footprint_vis=None, dp_du=None, dp_dv=None):
        uv_offset = torch.zeros_like(uv)

        # 1. Sample latent from texture (no blur at eval).
        tex = self.latent_texture.params
        latent = self._sample_from_texture(uv, tex)

        # 2. Predicted frame (or fall back to geometric).
        if self.predict_frame and self.use_predicted_frame_at_eval:
            predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)
        else:
            predicted_normal = normal
            predicted_tangent = TBN[:, :, 0] if TBN is not None else None

        # 3. Neural-geometry UV offset + re-sample.
        if self.neural_geometry_enabled and self.use_neural_geometry_at_eval:
            geometry_latent = latent[..., -6 - self.geometry_latent_dim:-6] if self.predict_frame \
                else latent[..., -self.geometry_latent_dim:]
            if self.local_wi_wo:
                wi_for_geo = self.world_to_local(wi, predicted_normal, predicted_tangent)
                wo_for_geo = self.world_to_local(wo, predicted_normal, predicted_tangent)
            else:
                wi_for_geo, wo_for_geo = wi, wo
            uv_offset = self.neural_geometry(wi_for_geo, wo_for_geo, geometry_latent) * self.neural_geometry_factor
            uv = ((uv + uv_offset) % 1 + 1) % 1
            latent = self._sample_from_texture(uv, tex)
            if self.recompute_frame and self.predict_frame and self.use_predicted_frame_at_eval:
                predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)

        # 4. World -> local and Disney decode.
        wi_local = self.world_to_local(wi, predicted_normal, predicted_tangent)
        wo_local = self.world_to_local(wo, predicted_normal, predicted_tangent)
        brdf_latent = latent[..., :self.brdf_latent_dim]
        brdf, _pdf_lobe = self.decoder(wi_local, wo_local, brdf_latent)

        if self.learnable_factor:
            brdf = brdf * self.factor

        # 5. Cosine weight (clamped >=0; see commit e896799) + 6-tuple return.
        cos_theta_i_pred = (wi * predicted_normal).sum(-1, keepdim=True).clamp(min=0.0)
        cos_theta_o_pred = (wo * predicted_normal).sum(-1, keepdim=True).clamp(min=0.0)
        pdf = cos_theta_i_pred / math.pi
        brdf = brdf * cos_theta_i_pred
        return brdf, predicted_normal, pdf, uv_offset, cos_theta_i_pred, cos_theta_o_pred

    def save_latent(self, path):
        self.latent_texture.save(path)

    def load_latent(self, path):
        self.latent_texture.load(path)
