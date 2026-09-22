"""Unit tests for videomaterial.synthetic_sequences.trajectories (run: python -m pytest -q)."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from synthetic_sequences import trajectories as T  # noqa: E402

SPECS = {"B1": T.TrajectorySpec.b1_light(), "B2": T.TrajectorySpec.b2_camera()}
CLASSES = T.TRAIN_CLASSES + T.TEST_CLASSES


def test_equal_area_roundtrip():
    th = np.array([0.5, 5.0, 30.0, 45.0, 80.0, 89.0]); ph = np.array([0.0, 10.0, 90.0, 180.0, 270.0, 359.0])
    th2, ph2 = T.disk_to_ang(T.ang_to_disk(th, ph))
    assert np.allclose(th, th2, atol=1e-9) and np.allclose(ph, ph2, atol=1e-9)
    v = T.ang_to_vec(th, ph); th3, ph3 = T.vec_to_ang(v)
    assert np.allclose(th, th3, atol=1e-9) and np.allclose(ph, ph3, atol=1e-9)
    assert np.allclose(np.linalg.norm(v, axis=-1), 1.0)


def test_equal_area_is_uniform_in_solid_angle():
    rng = np.random.default_rng(0)
    P = T._rand_disk(rng, 200000, 5.0, 80.0)
    th, _ = T.disk_to_ang(P)
    # cos(theta) must be uniform between cos(80) and cos(5) for uniform solid angle
    c = np.cos(np.radians(th)); lo, hi = math.cos(math.radians(80)), math.cos(math.radians(5))
    hist, _ = np.histogram(c, bins=10, range=(lo, hi))
    assert hist.max() / hist.min() < 1.15


def test_mirror_direction():
    assert T.mirror_direction(45.0, 90.0) == (45.0, 270.0)
    assert T.mirror_direction(30.0, 350.0) == (30.0, 170.0)


def test_determinism_and_constraints():
    for name, spec in SPECS.items():
        for cls in CLASSES:
            for seed in (1, 2, 3):
                a = T.sample_trajectory(cls, seed, spec)
                b = T.sample_trajectory(cls, seed, spec)
                assert np.array_equal(a.directions, b.directions), (name, cls, seed)
                assert a.params == b.params
                assert a.directions.shape == (spec.frames, 3)
                assert np.allclose(np.linalg.norm(a.directions, axis=-1), 1.0)
                th, _ = a.theta_phi_deg
                assert th.min() >= spec.polar_min_deg - 1e-6 and th.max() <= spec.polar_max_deg + 1e-6
                steps = T.angular_steps_deg(a.directions)
                assert steps.max() <= spec.step_cap_deg * 1.05 + 1e-9, (name, cls, seed, steps.max())
                assert math.degrees(0) <= a.params["arc_length_deg"] >= spec.min_arc_deg - 1e-6
                c = T.sample_trajectory(cls, seed + 100, spec)
                assert not np.array_equal(a.directions, c.directions)


def test_arc_length_resampling_is_uniform_for_const_profile():
    spec = SPECS["B1"]
    for seed in range(5):
        tr = T.sample_trajectory("ring", seed, spec, profile="const")
        steps = T.angular_steps_deg(tr.directions)
        assert steps.std() / steps.mean() < 0.05, steps.std() / steps.mean()


def test_highlight_sweep_passes_near_mirror():
    for spec in SPECS.values():
        mth, mph = spec.mirror_theta_phi; m = T.ang_to_vec(mth, mph)
        for seed in range(6):
            tr = T.sample_trajectory("highlight_sweep", seed, spec)
            dist = np.degrees(np.arccos(np.clip(tr.directions @ m, -1, 1)))
            assert dist.min() < spec.highlight_tol_deg + 2.0, dist.min()
            # slow_mirror profile: more frames near the mirror than a uniform sweep would give
            assert (dist < 10.0).mean() > 0.15


def test_spiral_reference_reproduces_exp005_poses():
    spec = SPECS["B1"]
    tr = T.sample_trajectory("spiral_reference", 0, spec)
    for i, (th, ph) in ((0, (80.0, 0.0)), (40, (42.5, 0.0)), (80, (5.0, 0.0))):
        v = T.ang_to_vec(th, ph)
        assert np.allclose(tr.directions[i], v, atol=2e-3), (i, tr.directions[i], v)
    pos = tr.positions
    assert np.allclose(pos[0], [2.954423259036624, 0.5209445330007912, 0.0], atol=5e-3)  # exp-005 frame 0


def test_coverage_greedy_beats_random():
    spec = SPECS["B1"]; rng = np.random.default_rng(1)
    pool = [("spline", T.sample_trajectory("spline", s, spec).directions) for s in range(200)]
    pool += [("highlight_sweep", T.sample_trajectory("highlight_sweep", s, spec).directions) for s in range(40)]
    sel = T.greedy_select(pool, spec, 32, quotas={"spline": 27, "highlight_sweep": 5})
    assert len(sel) == 32 and sum(pool[i][0] == "highlight_sweep" for i in sel) == 5
    g = T.coverage_stats([pool[i][1] for i in sel], spec)
    r = T.coverage_stats([pool[i][1] for i in rng.choice(len(pool), 32, replace=False)], spec)
    assert g["cv"] < r["cv"] and g["empty"] <= r["empty"]


def test_metadata_is_json_serialisable():
    import json
    tr = T.sample_trajectory("spline", 7, SPECS["B2"])
    s = json.dumps(tr.to_metadata()); assert "control_points_disk" in s and "centripetal" in s


def test_greedy_select_matches_reference():
    """The vectorised selector must be bit-identical to the readable loop version."""
    spec = SPECS["B1"]
    pool = [(cls, T.sample_trajectory(cls, 500 + i, spec).directions)
            for i, cls in enumerate(["spline"] * 12 + ["highlight_sweep"] * 6 + ["ring"] * 4)]
    assert T.greedy_select(pool, spec, 8) == T._greedy_select_reference(pool, spec, 8)
    q = {"spline": 5, "highlight_sweep": 2}
    a, b = T.greedy_select(pool, spec, 7, quotas=q), T._greedy_select_reference(pool, spec, 7, quotas=q)
    assert a == b and len(a) == 7
    assert sum(pool[i][0] == "spline" for i in a) == 5
    # quota exhaustion stops the selection rather than spilling into other classes
    assert len(T.greedy_select(pool, spec, 20, quotas={"ring": 4})) == 4


def test_greedy_select_beats_random_coverage():
    spec = SPECS["B1"]
    pool = [("spline", T.sample_trajectory("spline", 9000 + i, spec).directions) for i in range(60)]
    sel = T.greedy_select(pool, spec, 12)
    rng = np.random.default_rng(0)
    g = T.coverage_stats([pool[i][1] for i in sel], spec)
    r = T.coverage_stats([pool[i][1] for i in rng.choice(len(pool), 12, replace=False)], spec)
    assert g["empty"] <= r["empty"] and g["cv"] <= r["cv"]
