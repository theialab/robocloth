"""Scene builders and camera helpers for the exp-015 synthetic sequences.

Three scenes share one camera/light description:
  * ``material``      — the exp-005 neural-material rectangle (RoboCloth ``mlpbrdf`` BSDF) loaded
                        from the exp-005 ``scene.xml`` + ``materials.json`` through RoboCloth's loader;
  * ``white_lambert`` — the same rectangle with a white diffuse BSDF (albedo 1), used for the
                        G0 irradiance / silhouette checks;
  * ``ball``          — trajectory visualisation: grey principled sphere (radius 0.5, centred at the
                        origin) on a grey ground plane at y = -0.5, plus a small emissive marker
                        sphere at the light position.
Sample frame = world frame (exp-005): origin at the sample centre, +Y normal, +X = u, +Z = v.
"""
from __future__ import annotations

import math

import numpy as np

SAMPLE_HALF_EXTENT = 0.75          # exp-005: rectangle rotated -90° about X, scaled 0.75


def look_at_matrix(origin, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
    """Mitsuba-convention camera-to-world (camera looks along +Z, +Y up, +X to the LEFT of the
    image — Mitsuba's perspective sensor mirrors X; the metadata records this string)."""
    import mitsuba as mi
    return mi.ScalarTransform4f().look_at(origin=list(map(float, origin)), target=list(map(float, target)),
                                          up=list(map(float, up)))


def intrinsics(width, height, fov_deg, fov_axis="y"):
    if fov_axis == "y":
        fy = (height / 2.0) / math.tan(math.radians(fov_deg) / 2.0); fx = fy
    else:
        fx = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0); fy = fx
    return [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]]


def transform_to_list(T):
    return np.array(T.matrix, dtype=np.float64).reshape(4, 4).tolist()


def sensor_dict(camera_pos, fov_deg, width, height, rfilter=None):
    film = {"type": "hdrfilm", "width": int(width), "height": int(height),
            "pixel_format": "rgb", "component_format": "float32"}
    if rfilter:   # exp-005 / material renders keep Mitsuba's default (gaussian); checks use "box"
        film["rfilter"] = {"type": rfilter}
    return {
        "type": "perspective", "fov": float(fov_deg), "fov_axis": "y",
        "to_world": look_at_matrix(camera_pos),
        "sampler": {"type": "independent"},
        "film": film,
    }


BALL_CENTER = [-1.15, 0.35, 0.0]
BALL_RADIUS = 0.35


def ball_scene_dict(camera_pos, light_pos, fov_deg, width, height, intensity=20.0,
                    ball_roughness=0.3, ground_roughness=0.15, max_depth=4):
    """Visualisation scene with the SAME geometry as the material scene: ground plane at y = 0 with a
    glossy patch of the sample's size (half-extent 0.75) at the origin, so the light's reflection appears
    at the mirror point exactly where the material sample would show its specular peak; a grey ball to
    the side (centre (-1.15, 0.35, 0), radius 0.35) for shading and cast-shadow cues; the exp-005 point
    light. The light position is drawn as an overlay in make_visualization.py."""
    import mitsuba as mi
    T = mi.ScalarTransform4f
    return {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": int(max_depth)},
        "sensor": sensor_dict(camera_pos, fov_deg, width, height),
        "ground": {"type": "rectangle", "to_world": T().rotate([1, 0, 0], -90).scale(10.0),
                   "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.28, 0.28, 0.28]}}},
        "sample_patch": {"type": "rectangle",
                         "to_world": T().translate([0.0, 0.002, 0.0]).rotate([1, 0, 0], -90).scale(SAMPLE_HALF_EXTENT),
                         "bsdf": {"type": "principled", "base_color": {"type": "rgb", "value": [0.5, 0.5, 0.5]},
                                  "roughness": float(ground_roughness), "specular": 0.6}},
        "ball": {"type": "sphere", "radius": BALL_RADIUS, "center": BALL_CENTER,
                 "bsdf": {"type": "principled", "base_color": {"type": "rgb", "value": [0.6, 0.6, 0.6]},
                          "roughness": float(ball_roughness), "specular": 0.5}},
        "light": {"type": "point", "position": [float(x) for x in light_pos],
                  "intensity": {"type": "rgb", "value": [intensity] * 3}},
    }


def marker_radiance(intensity, radius):
    return float(intensity / (math.pi * radius * radius))


def white_lambert_scene_dict(camera_pos, light_pos, fov_deg, width, height, intensity=20.0, max_depth=4):
    import mitsuba as mi
    T = mi.ScalarTransform4f
    return {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": int(max_depth)},
        "sensor": sensor_dict(camera_pos, fov_deg, width, height, rfilter="box"),
        "cloth": {"type": "rectangle", "to_world": T().rotate([1, 0, 0], -90).scale(SAMPLE_HALF_EXTENT),
                  "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [1.0, 1.0, 1.0]}}},
        "light": {"type": "point", "position": [float(x) for x in light_pos],
                  "intensity": {"type": "rgb", "value": [intensity] * 3}},
    }


def load_material_scene(scene_src, checkpoint_root, rendering_root):
    """exp-005 neural-material scene via RoboCloth's loader (registers the BSDF plugins)."""
    import json, sys, tempfile, shutil
    from pathlib import Path
    import mitsuba as mi
    sys.path.insert(0, str(rendering_root))
    from render import load_materials  # noqa
    from brdf_plugin.cook_torrancebrdf import CookTorranceBRDF  # noqa
    from brdf_plugin.mlp import MLPBRDF  # noqa
    from brdf_plugin.utils.scene_loader import load_scene_with_bsdf_overrides  # noqa
    mi.register_bsdf("cooktorrancebrdf", lambda props: CookTorranceBRDF(props))
    mi.register_bsdf("mlpbrdf", lambda props: MLPBRDF(props))
    scene_src = Path(scene_src)
    tmp = Path(tempfile.mkdtemp(prefix="vm_scene_"))
    shutil.copy(scene_src / "scene.xml", tmp / "scene.xml")
    spec = json.loads((scene_src / "materials.json").read_text())
    spec["checkpoint_root"] = str(checkpoint_root)
    (tmp / "materials.json").write_text(json.dumps(spec, indent=2) + "\n")
    overrides, radiance = load_materials(str(tmp), default_two_sided=False)
    scene = load_scene_with_bsdf_overrides(str(tmp / "scene.xml"), overrides, radiance=radiance)
    return scene, spec, tmp
