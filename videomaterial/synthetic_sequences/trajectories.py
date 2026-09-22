"""Camera / light trajectory sampler for the RoboCloth synthetic material sequences (exp-015).

Conventions (sample frame, identical to exp-005): origin at the sample centre, +Y is the sample
normal, +X is the u axis, +Z the v axis; ``theta`` is the polar angle from +Y, ``phi`` the
azimuth from +X towards +Z. Directions are unit vectors ``(sin t cos p, cos t, sin t sin p)``.

Sampling lives on the equal-area (Lambert azimuthal) disk, ``r = sqrt(2) * sin(theta / 2)``:
uniform density on the disk is uniform solid angle on the hemisphere, and interpolation on the
disk has no zenith singularity and no azimuth wrap-around.

Public API
----------
``sample_trajectory(cls, seed, spec) -> Trajectory``  deterministic given ``seed``.
``greedy_select(directions_list, spec, n, quotas=None)``  coverage-maximising subset selection.
``TrajectorySpec.b1_light() / b2_camera()``  the agreed B1 / B2 settings.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

SQRT2 = math.sqrt(2.0)
TRAIN_CLASSES = ("spline", "highlight_sweep")
TEST_CLASSES = ("ring", "spiral", "lissajous")
REFERENCE_CLASSES = ("spiral_reference",)
ALL_CLASSES = TRAIN_CLASSES + TEST_CLASSES + REFERENCE_CLASSES
SPEED_PROFILES = ("const", "ease", "slow_mirror")


# ----------------------------------------------------------------------------- geometry helpers
def ang_to_vec(theta_deg, phi_deg):
    t, p = np.radians(theta_deg), np.radians(phi_deg)
    return np.stack([np.sin(t) * np.cos(p), np.cos(t), np.sin(t) * np.sin(p)], axis=-1)


def vec_to_ang(v):
    v = np.asarray(v, dtype=np.float64)
    theta = np.degrees(np.arccos(np.clip(v[..., 1], -1.0, 1.0)))
    phi = np.degrees(np.arctan2(v[..., 2], v[..., 0])) % 360.0
    return theta, phi


def ang_to_disk(theta_deg, phi_deg):
    t, p = np.radians(theta_deg), np.radians(phi_deg)
    r = SQRT2 * np.sin(t / 2.0)
    return np.stack([r * np.cos(p), r * np.sin(p)], axis=-1)


def disk_to_ang(pts):
    pts = np.asarray(pts, dtype=np.float64)
    r = np.linalg.norm(pts, axis=-1)
    theta = np.degrees(2.0 * np.arcsin(np.clip(r / SQRT2, 0.0, 1.0)))
    phi = np.degrees(np.arctan2(pts[..., 1], pts[..., 0])) % 360.0
    return theta, phi


def disk_radius(theta_deg):
    return SQRT2 * math.sin(math.radians(theta_deg) / 2.0)


def mirror_direction(theta_deg, phi_deg):
    """Mirror (specular) direction of a direction about the sample normal +Y."""
    return theta_deg, (phi_deg + 180.0) % 360.0


def angular_steps_deg(v):
    d = np.clip((v[1:] * v[:-1]).sum(-1), -1.0, 1.0)
    return np.degrees(np.arccos(d))


# ----------------------------------------------------------------------------- spec
@dataclass
class TrajectorySpec:
    moving_element: str                 # "light" | "camera"
    polar_min_deg: float
    polar_max_deg: float
    step_cap_deg: float                 # max angular step between consecutive frames
    radius: float = 3.0
    frames: int = 81
    fixed_theta_deg: float = 45.0       # the element that does NOT move
    fixed_phi_deg: float = 90.0
    min_arc_deg: float = 40.0           # shortest clip we accept
    highlight_tol_deg: float = 3.0      # highlight_sweep passes this close to the mirror direction
    extra: dict = field(default_factory=dict)

    @property
    def mirror_theta_phi(self):
        return mirror_direction(self.fixed_theta_deg, self.fixed_phi_deg)

    @property
    def fixed_direction(self):
        return ang_to_vec(self.fixed_theta_deg, self.fixed_phi_deg)

    @staticmethod
    def b1_light(**kw):
        """B1: fixed camera at θ 45° φ 90°, moving point light, polar 5–80°, cap 6°/frame."""
        return TrajectorySpec("light", 5.0, 80.0, 6.0, **kw)

    @staticmethod
    def b2_camera(**kw):
        """B2: fixed light at θ 45° φ 90°, moving camera, polar 0.5–60°, cap 3°/frame."""
        return TrajectorySpec("camera", 0.5, 60.0, 3.0, **kw)


@dataclass
class Trajectory:
    cls: str
    seed: int
    directions: np.ndarray              # (F, 3) unit vectors in the sample frame
    params: dict                        # everything needed to reproduce the curve

    @property
    def positions(self):
        return self.directions * self.params["radius"]

    @property
    def theta_phi_deg(self):
        return vec_to_ang(self.directions)

    def to_metadata(self) -> dict:
        theta, phi = self.theta_phi_deg
        return {
            "class": self.cls,
            "seed": self.seed,
            **self.params,
            "theta_deg_range_actual": [float(theta.min()), float(theta.max())],
            "phi_coverage_deg": float(np.ptp(np.unwrap(np.radians(phi))) * 180.0 / math.pi),
        }


# ----------------------------------------------------------------------------- curve families
def _rand_disk(rng, n, tmin, tmax):
    a, b = disk_radius(tmin) ** 2, disk_radius(tmax) ** 2
    r = np.sqrt(rng.uniform(a, b, n))
    ph = rng.uniform(0.0, 2.0 * math.pi, n)
    return np.stack([r * np.cos(ph), r * np.sin(ph)], -1)


def _catmull_rom(P, samples=800):
    """Centripetal Catmull-Rom through all control points (no overshoot spikes)."""
    P = np.asarray(P, dtype=np.float64)
    if len(P) == 2:
        t = np.linspace(0, 1, samples)[:, None]
        return P[0] * (1 - t) + P[1] * t
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    n_seg = len(P) - 1
    per = max(2, samples // n_seg)
    for i in range(n_seg):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        d = lambda a, b: max(np.linalg.norm(b - a) ** 0.5, 1e-6)  # centripetal alpha = 0.5
        t0, t1, t2, t3 = 0.0, d(p0, p1), d(p0, p1) + d(p1, p2), d(p0, p1) + d(p1, p2) + d(p2, p3)
        t = np.linspace(t1, t2, per, endpoint=(i == n_seg - 1))[:, None]
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    return np.vstack(out)


def _curve(cls, rng, spec, params):
    """Return a dense polyline on the disk (N, 2) and fill ``params`` with what generated it."""
    tmin, tmax = spec.polar_min_deg, spec.polar_max_deg
    u = np.linspace(0.0, 1.0, 800)
    if cls == "spline":
        k = int(rng.integers(3, 6))
        P = _rand_disk(rng, k, tmin, tmax)
        params.update(control_points_disk=P.tolist(), interpolation="centripetal_catmull_rom")
        return _catmull_rom(P)
    if cls == "highlight_sweep":
        mth, mph = spec.mirror_theta_phi
        m = ang_to_disk(mth, mph) + rng.normal(0.0, disk_radius(spec.highlight_tol_deg) * 0.6, 2)
        ang = rng.uniform(0.0, math.pi)
        dvec = np.array([math.cos(ang), math.sin(ang)])
        nvec = np.array([-dvec[1], dvec[0]])
        half = rng.uniform(0.35, 0.7)          # half-length on the disk
        bend = rng.normal(0.0, 0.25)
        x = (u - 0.5) * 2.0 * half
        P = m[None] + x[:, None] * dvec[None] + bend * (x ** 2)[:, None] * nvec[None]
        params.update(mirror_disk=m.tolist(), sweep_direction_deg=math.degrees(ang),
                      half_length_disk=half, bend=bend)
        return P
    if cls == "ring":
        th = rng.uniform(max(tmin, 10.0), tmax)
        p0 = rng.uniform(0.0, 360.0)
        sweep = rng.uniform(90.0, 360.0) * rng.choice([-1.0, 1.0])
        params.update(ring_theta_deg=th, phi_start_deg=p0, phi_sweep_deg=sweep)
        return ang_to_disk(np.full_like(u, th), p0 + sweep * u)
    if cls == "spiral":
        a, b = rng.uniform(tmin, tmax, 2)
        revs = rng.uniform(0.5, 2.5) * rng.choice([-1.0, 1.0])
        p0 = rng.uniform(0.0, 360.0)
        params.update(theta_start_deg=a, theta_end_deg=b, revolutions=revs, phi_start_deg=p0)
        return ang_to_disk(a + (b - a) * u, p0 + 360.0 * revs * u)
    if cls == "lissajous":
        R = disk_radius(tmax) * 0.85
        a, b = rng.choice([1, 2, 3], 2, replace=False)
        ph = rng.uniform(0.0, 2.0 * math.pi)
        params.update(lissajous_a=int(a), lissajous_b=int(b), phase_deg=math.degrees(ph), amplitude_disk=R)
        return np.stack([R * np.sin(2 * math.pi * a * u + ph), R * np.sin(2 * math.pi * b * u)], -1)
    raise ValueError(f"unknown trajectory class {cls!r}")


# ----------------------------------------------------------------------------- resampling
def _resample(poly, spec, rng, profile, cls, params):
    theta, phi = disk_to_ang(poly)
    overshoot = float(((theta < spec.polar_min_deg) | (theta > spec.polar_max_deg)).mean())
    theta = np.clip(theta, spec.polar_min_deg, spec.polar_max_deg)
    v = ang_to_vec(theta, phi)
    seg = np.radians(angular_steps_deg(v))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    F = spec.frames
    Lmax = min(total, math.radians(spec.step_cap_deg) * (F - 1))
    if Lmax < math.radians(spec.min_arc_deg):
        return None
    L = rng.uniform(max(math.radians(spec.min_arc_deg), 0.5 * Lmax), Lmax)
    s0 = rng.uniform(0.0, total - L)
    u = np.linspace(0.0, 1.0, F)
    if profile == "ease":
        u = 0.5 * (u * u * (3 - 2 * u)) + 0.5 * u
    elif profile == "slow_mirror":
        mth, mph = spec.mirror_theta_phi
        mvec = ang_to_vec(mth, mph)
        ss = np.linspace(s0, s0 + L, 400)
        vv = np.stack([np.interp(ss, s, v[:, k]) for k in range(3)], -1)
        vv /= np.linalg.norm(vv, axis=-1, keepdims=True)
        dist = np.degrees(np.arccos(np.clip(vv @ mvec, -1, 1)))
        w = 1.0 / (0.15 + dist / 25.0)
        c = np.cumsum(w); c = (c - c[0]) / (c[-1] - c[0])
        u = np.interp(u, c, np.linspace(0.0, 1.0, 400))
    sq = s0 + u * L
    out = np.stack([np.interp(sq, s, v[:, k]) for k in range(3)], -1)
    out /= np.linalg.norm(out, axis=-1, keepdims=True)
    steps = angular_steps_deg(out)
    if steps.max() > spec.step_cap_deg * 1.05:
        return None
    params.update(speed_profile=profile, arc_length_deg=math.degrees(L),
                  arc_start_deg=math.degrees(s0), curve_length_deg=math.degrees(total),
                  max_step_deg=float(steps.max()), median_step_deg=float(np.median(steps)),
                  overshoot_clipped_fraction=overshoot)
    return out


def sample_trajectory(cls: str, seed: int, spec: TrajectorySpec, profile: Optional[str] = None,
                      max_attempts: int = 200) -> Trajectory:
    """Deterministic: the same (cls, seed, spec) always yields the same trajectory."""
    if cls not in ALL_CLASSES:
        raise ValueError(f"unknown class {cls!r}; choose from {ALL_CLASSES}")
    rng = np.random.default_rng([int(seed), zlib.crc32(cls.encode()), int(spec.polar_max_deg * 100), int(spec.polar_min_deg * 100)])
    for attempt in range(max_attempts):
        params = {"radius": spec.radius, "frames": spec.frames, "moving_element": spec.moving_element,
                  "polar_range_deg": [spec.polar_min_deg, spec.polar_max_deg], "step_cap_deg": spec.step_cap_deg,
                  "fixed_element_deg": {"theta": spec.fixed_theta_deg, "phi": spec.fixed_phi_deg},
                  "mirror_direction_deg": dict(zip(("theta", "phi"), spec.mirror_theta_phi)),
                  "attempt": attempt}
        if cls == "spiral_reference":   # exp-005 schedule: uniform in t, θ 80→5, φ = 720·t, no cap
            t = np.linspace(0.0, 1.0, spec.frames)
            v = ang_to_vec(80.0 + (5.0 - 80.0) * t, 720.0 * t)
            steps = angular_steps_deg(v)
            params.update(theta_start_deg=80.0, theta_end_deg=5.0, revolutions=2.0, phi_start_deg=0.0,
                          speed_profile="uniform_in_t", arc_length_deg=float(steps.sum()),
                          max_step_deg=float(steps.max()), median_step_deg=float(np.median(steps)),
                          overshoot_clipped_fraction=0.0, note="exp-005 light schedule, exempt from the step cap")
            return Trajectory(cls, seed, v, params)
        poly = _curve(cls, rng, spec, params)
        prof = profile or ("slow_mirror" if cls == "highlight_sweep" else str(rng.choice(["const", "ease"])))
        v = _resample(poly, spec, rng, prof, cls, params)
        if v is not None:
            return Trajectory(cls, seed, v, params)
    raise RuntimeError(f"could not sample a valid {cls} trajectory for seed {seed}")


# ----------------------------------------------------------------------------- coverage
def cell_ids(directions, spec, n_rings=8, n_sectors=24):
    theta, phi = vec_to_ang(directions)
    a, b = math.sin(math.radians(spec.polar_min_deg) / 2) ** 2, math.sin(math.radians(spec.polar_max_deg) / 2) ** 2
    ri = np.clip(((np.sin(np.radians(theta) / 2) ** 2 - a) / (b - a) * n_rings).astype(int), 0, n_rings - 1)
    si = np.clip((phi / 360.0 * n_sectors).astype(int), 0, n_sectors - 1)
    return ri * n_sectors + si


def coverage_stats(list_of_directions, spec, n_rings=8, n_sectors=24):
    cnt = np.zeros(n_rings * n_sectors)
    for v in list_of_directions:
        cnt += np.bincount(cell_ids(v, spec, n_rings, n_sectors), minlength=n_rings * n_sectors)
    return {"cells": n_rings * n_sectors, "empty": int((cnt == 0).sum()), "min": int(cnt.min()),
            "median": float(np.median(cnt)), "max": int(cnt.max()), "cv": float(cnt.std() / max(cnt.mean(), 1e-9))}


def greedy_select(pool, spec, n, quotas=None, n_rings=8, n_sectors=24):
    """pool: list of (cls, directions). Greedily maximise Σ log(1+count) over equal-area cells,
    under optional per-class quotas {cls: count}. Returns the selected indices."""
    C = [np.bincount(cell_ids(v, spec, n_rings, n_sectors), minlength=n_rings * n_sectors) for _, v in pool]
    cnt = np.zeros(n_rings * n_sectors)
    left = dict(quotas) if quotas else None
    sel, used = [], set()
    for _ in range(n):
        best, bg = None, -np.inf
        for i, (cls, _) in enumerate(pool):
            if i in used or (left is not None and left.get(cls, 0) <= 0):
                continue
            g = (np.log1p(cnt + C[i]) - np.log1p(cnt)).sum()
            if g > bg:
                bg, best = g, i
        if best is None:
            break
        used.add(best); sel.append(best); cnt += C[best]
        if left is not None:
            left[pool[best][0]] -= 1
    return sel
