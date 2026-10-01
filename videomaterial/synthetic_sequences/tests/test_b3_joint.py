"""B3 (jointly moving camera AND light) manifest, poses, metadata and checks (run: python tests/run_tests.py).

The source manifests are small B1 / B2 manifests built by ``make_manifest.build`` itself, padded to the
real 900 / 100 split sizes where a test needs the real group sizes. Files go to ``tempfile`` (honours
$TMPDIR — on g00s that must point at the fileserver).
"""
import hashlib, json, re, sys, tempfile, types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from synthetic_sequences import make_manifest as M  # noqa: E402
from synthetic_sequences import seqlib as L  # noqa: E402
from synthetic_sequences import check_metadata as CM  # noqa: E402

SMALL_POOL = {"spline": 60, "highlight_sweep": 20}
SMALL_QUOTAS = {"spline": 17, "highlight_sweep": 3}
SMALL_TEST = {"spline": 3, "highlight_sweep": 1, "ring": 2, "spiral": 2, "lissajous": 1, "spiral_reference": 1}
_CACHE = {}


def _small_source(setup):
    if setup not in _CACHE:
        saved = (M.POOL, M.TRAIN_QUOTAS, M.TEST_COUNTS)
        M.POOL, M.TRAIN_QUOTAS, M.TEST_COUNTS = SMALL_POOL, SMALL_QUOTAS, SMALL_TEST
        try:
            _CACHE[setup] = M.build(master_seed=20260922 if setup == "B1" else 20260923, verbose=False, setup=setup)
        finally:
            M.POOL, M.TRAIN_QUOTAS, M.TEST_COUNTS = saved
    return _CACHE[setup]


def _full_size_source(setup):
    """A source manifest with the real 900 train / 100 test sizes (seeds are cheap to fake: the pairing
    only looks at split / dir; the poses of these entries are never generated)."""
    m = json.loads(json.dumps(_small_source(setup)))
    tr = [s for s in m["sequences"] if s["split"] == "train"]; te = [s for s in m["sequences"] if s["split"] == "test"]
    seqs = []
    for k in range(900):
        s = dict(tr[k % len(tr)]); s.update(index=k, id=f"{s['class']}_{k+1:04d}", dir=f"train/{s['class']}_{k+1:04d}")
        seqs.append(s)
    for k in range(100):
        s = dict(te[k % len(te)]); s.update(index=900 + k, id=f"{s['class']}_{k+1:04d}", dir=f"test/{s['class']}_{k+1:04d}")
        seqs.append(s)
    m["sequences"] = seqs
    return m


def _b3_full(seed=20261001):
    return M.build_b3(_full_size_source("B1"), _full_size_source("B2"), master_seed=seed, verbose=False, stats=False)


def _b3_small(**kw):
    kw.setdefault("test_groups", {"unseen_light": 4, "unseen_camera": 3, "unseen_pair": 3})
    return M.build_b3(_small_source("B1"), _small_source("B2"), verbose=False, **kw)


# ------------------------------------------------------------------------------------------ pairing
def test_b3_group_sizes_names_and_exr_policy():
    m = _b3_full()
    c = m["counts"]
    assert (c["total"], c["train"], c["test"], c["keep_exr"]) == (1000, 900, 100, 190)
    assert c["by_split_group"] == {"train": {"pair": 900},
                                   "test": {"unseen_light": 34, "unseen_camera": 33, "unseen_pair": 33}}
    dirs = [s["dir"] for s in m["sequences"]]
    assert len(set(dirs)) == 1000
    pat = re.compile(r"^(train/pair|test/unseen_light|test/unseen_camera|test/unseen_pair)_\d{4}$")
    assert all(pat.match(d) for d in dirs), "flat <split>/<group>_NNNN names only (no seeds, no dates)"
    assert [s["index"] for s in m["sequences"]] == list(range(1000))
    train = [s for s in m["sequences"] if s["split"] == "train"]
    assert [s["index"] for s in train if s["keep_exr"]] == list(range(0, 900, M.EXR_EVERY_N_TRAIN))
    assert all(s["keep_exr"] for s in m["sequences"] if s["split"] == "test")
    assert m["setup"] == "B3" and m["setup_dir"] == "B3_Joint" and m["twin"] == "pbr_patch"


def test_b3_train_pairing_is_one_to_one():
    m = _b3_full()
    train = [s for s in m["sequences"] if s["split"] == "train"]
    lights = [s["light"]["dir"] for s in train]; cams = [s["camera"]["dir"] for s in train]
    assert all(s["light"]["setup"] == "B1" and s["light"]["split"] == "train" for s in train)
    assert all(s["camera"]["setup"] == "B2" and s["camera"]["split"] == "train" for s in train)
    assert len(set(lights)) == 900 and len(set(cams)) == 900, "each train path used exactly once"
    src_l = {s["dir"] for s in _full_size_source("B1")["sequences"] if s["split"] == "train"}
    src_c = {s["dir"] for s in _full_size_source("B2")["sequences"] if s["split"] == "train"}
    assert set(lights) == src_l and set(cams) == src_c
    # a random pairing, not the identity / index-aligned one
    idx = [(s["light"]["manifest_index"], s["camera"]["manifest_index"]) for s in train]
    assert sum(a == b for a, b in idx) < 10


def test_b3_unseen_light_uses_b1_test_light_and_b2_train_camera():
    g = [s for s in _b3_full()["sequences"] if s["group"] == "unseen_light"]
    assert len(g) == 34
    assert all(s["light"]["setup"] == "B1" and s["light"]["split"] == "test" for s in g)
    assert all(s["camera"]["setup"] == "B2" and s["camera"]["split"] == "train" for s in g)
    assert len({s["light"]["dir"] for s in g}) == 34 and len({s["camera"]["dir"] for s in g}) == 34


def test_b3_unseen_camera_uses_b2_test_camera_and_b1_train_light():
    g = [s for s in _b3_full()["sequences"] if s["group"] == "unseen_camera"]
    assert len(g) == 33
    assert all(s["camera"]["setup"] == "B2" and s["camera"]["split"] == "test" for s in g)
    assert all(s["light"]["setup"] == "B1" and s["light"]["split"] == "train" for s in g)
    assert len({s["light"]["dir"] for s in g}) == 33 and len({s["camera"]["dir"] for s in g}) == 33


def test_b3_unseen_pair_is_not_a_training_pair():
    for seed in (20261001, 1, 2, 3):
        m = _b3_full(seed)
        train = {(s["light"]["dir"], s["camera"]["dir"]) for s in m["sequences"] if s["split"] == "train"}
        g = [s for s in m["sequences"] if s["group"] == "unseen_pair"]
        assert len(g) == 33
        assert all(s["light"]["split"] == "train" and s["camera"]["split"] == "train" for s in g)
        assert not ({(s["light"]["dir"], s["camera"]["dir"]) for s in g} & train)
        assert len({s["light"]["dir"] for s in g}) == 33 and len({s["camera"]["dir"] for s in g}) == 33


def test_b3_unseen_pair_redraw_when_forced():
    """With a 3-path pool every camera draw of 3 hits the training pairing unless it is a derangement of
    it, so the redraw loop must run and still end with no training pair."""
    def tiny(setup):
        m = json.loads(json.dumps(_small_source(setup)))
        tr = [s for s in m["sequences"] if s["split"] == "train"][:3]
        te = [s for s in m["sequences"] if s["split"] == "test"][:3]
        m["sequences"] = tr + te
        return m
    redraws = []
    for seed in range(12):
        m = M.build_b3(tiny("B1"), tiny("B2"), master_seed=seed, verbose=False, stats=False,
                       test_groups={"unseen_light": 2, "unseen_camera": 2, "unseen_pair": 3})
        train = {(s["light"]["dir"], s["camera"]["dir"]) for s in m["sequences"] if s["split"] == "train"}
        assert not ({(s["light"]["dir"], s["camera"]["dir"]) for s in m["sequences"] if s["group"] == "unseen_pair"} & train)
        redraws.append(m["pairing"]["unseen_pair_camera_redraws"])
    assert max(redraws) > 0


def test_b3_deterministic_given_master_seed():
    a, b = _b3_full(7), _b3_full(7)
    assert a["sequences"] == b["sequences"]
    assert _b3_full(8)["sequences"] != a["sequences"]


def test_b3_rejects_wrong_sources():
    try:
        M.build_b3(_small_source("B2"), _small_source("B1"), verbose=False, stats=False)
    except ValueError:
        return
    raise AssertionError("B3 must refuse a B2 light manifest / B1 camera manifest")


def test_b3_statistics_cover_both_elements():
    m = _b3_small()
    st = m["pairing"]["statistics"]
    # the train light / camera sets ARE the source train sets, so coverage equals the sources' greedy stats
    assert st["light_coverage_192_cells"]["train"] == _small_source("B1")["selection"]["coverage_stats"]["greedy"]
    assert st["camera_coverage_192_cells"]["train"] == _small_source("B2")["selection"]["coverage_stats"]["greedy"]
    assert st["joint_coverage"]["train"]["cells"] == 1024
    d = st["distinct_paths"]
    assert d["unseen_light"]["light_test_paths"]["uses"] == 4 and "light_train_paths" not in d["unseen_light"]
    assert d["unseen_camera"]["camera_test_paths"]["uses"] == 3 and "camera_train_paths" not in d["unseen_camera"]
    json.dumps(m)  # serialisable


# -------------------------------------------------------------------------------------------- poses
def test_b3_poses_equal_the_sources():
    m = _b3_small()
    for s in m["sequences"][:3] + m["sequences"][-4:]:
        p = L.poses_for_sequence(s, "B3", 81, m["master_seed"])
        lp = L.build_poses("B1", s["light"]["class"], s["light"]["seed"], 81)
        cp = L.build_poses("B2", s["camera"]["class"], s["camera"]["seed"], 81)
        assert np.array_equal(p.light_pos, lp.light_pos) and np.array_equal(p.cam_pos, cp.cam_pos)
        # B3 moves both: neither element is constant
        assert np.ptp(p.light_pos, axis=0).max() > 0.1 and np.ptp(p.cam_pos, axis=0).max() > 0.1


def _fake_source_dataset(root, m):
    """Write source frames.jsonl files exactly as render_manifest would (frame_record), for the
    sources referenced by the first sequences of manifest ``m``."""
    cfg = L.RenderConfig()
    for s in m["sequences"][:3]:
        for role in ("light", "camera"):
            src = s[role]
            out = Path(root) / src["setup_dir"] / src["dir"]; out.mkdir(parents=True, exist_ok=True)
            p = L.build_poses(src["setup"], src["class"], src["seed"], 81)
            with (out / "frames.jsonl").open("w") as f:
                for k in range(81):
                    rec = {"index": k, "seed": 0, "seconds": 0.0, "stats": {}, "png": out / f"frames_png/frame_{k:03d}.png", "exr": None}
                    f.write(json.dumps(L.frame_record(rec, p, cfg, out), sort_keys=True) + "\n")


def test_b3_verify_joint_sources_exact_and_detects_tampering():
    import mitsuba as mi
    mi.set_variant("scalar_rgb")
    m = _b3_small()
    with tempfile.TemporaryDirectory(prefix="b3_src_") as root:
        _fake_source_dataset(root, m)
        s = m["sequences"][0]
        p = L.poses_for_sequence(s, "B3", 81, m["master_seed"])
        r = L.verify_joint_sources(p, root)
        assert r["exact"] and r["c2w_compared"] and r["camera_c2w_max_abs_diff"] == 0.0
        f = Path(root) / s["camera"]["setup_dir"] / s["camera"]["dir"] / "frames.jsonl"
        lines = f.read_text().splitlines()
        rec = json.loads(lines[40]); rec["camera"]["position"][0] += 1e-12
        lines[40] = json.dumps(rec, sort_keys=True); f.write_text("\n".join(lines) + "\n")
        try:
            L.verify_joint_sources(p, root)
        except RuntimeError:
            return
        raise AssertionError("a 1e-12 change in one camera position must be detected")


# ----------------------------------------------------------------------------------------- metadata
def _stub_modules():
    mi = types.SimpleNamespace(__version__="3.8.0"); dr = types.SimpleNamespace(__version__="1.3.1")
    torch = types.SimpleNamespace(__version__="stub", cuda=types.SimpleNamespace(is_available=lambda: False))
    return mi, dr, torch


def test_b3_metadata_and_frames_schema():
    import mitsuba as mi_real
    mi_real.set_variant("scalar_rgb")
    m = _b3_small()
    s = next(x for x in m["sequences"] if x["group"] == "unseen_light")
    p = L.poses_for_sequence(s, "B3", 81, m["master_seed"])
    mi, dr, torch = _stub_modules()
    with tempfile.TemporaryDirectory(prefix="b3_meta_") as root:
        out = Path(root) / "B3_Joint" / s["dir"]; out.mkdir(parents=True)
        meta = L.build_metadata(out, p, "material", L.RenderConfig(), {"kind": "stub"}, mi, dr, torch, 0.0, s["split"], "x")
        assert meta["sequence_id"] == f"B3_Joint/{s['dir']}" and meta["setup"] == "B3"
        assert meta["camera"]["mode"] == "trajectory" and meta["light"]["mode"] == "trajectory"
        assert meta["camera"]["fixed_direction_deg"] is None and meta["light"]["fixed_direction_deg"] is None
        tr = meta["trajectory"]
        assert tr["moving_element"] == "camera+light" and tr["group"] == "unseen_light" and tr["split"] == "test"
        assert tr["light_path"] == f"B1_Fixed_camera/{s['light']['dir']}" and tr["light_path"].split("/")[1] == "test"
        assert tr["camera_path"] == f"B2_Fixed_light/{s['camera']['dir']}"
        assert tr["light_class"] == s["light"]["class"] and tr["camera_class"] == s["camera"]["class"]
        assert tr["light"]["seed"] == s["light"]["seed"] and tr["camera"]["seed"] == s["camera"]["seed"]
        assert meta["seeds"] == {"light_trajectory": s["light"]["seed"], "camera_trajectory": s["camera"]["seed"],
                                 "pairing_master": m["master_seed"], "render_base": 20260922}
        # B1/B2 schema blocks are shared, not re-typed
        b1 = L.build_metadata(out, p.joint["light"], "material", L.RenderConfig(), {"kind": "stub"}, mi, dr, torch, 0.0, "train", "x")
        for k in ("sample", "frame_convention", "render", "frames"):
            assert meta[k] == b1[k]
        assert list(meta) == list(b1), "same top-level key order as B1/B2"
        json.dumps(meta)
        # frames.jsonl: per frame BOTH camera pose and light pose; derivable with check_metadata alone
        rec = {"index": 40, "seed": 1, "seconds": 0.0, "stats": {}, "png": out / "frames_png/frame_040.png", "exr": None}
        line = L.frame_record(rec, p, L.RenderConfig(), out)
        assert line["camera"]["position"] == p.cam_pos[40].tolist() and line["light"]["position"] == p.light_pos[40].tolist()
        dv = CM.derive(meta, line)
        assert dv["inside"].sum() > 1000
        # analytic E at a hit point == I cos(theta_i) / r^2 with THIS frame's light
        iy, ix = np.argwhere(dv["inside"])[len(np.argwhere(dv["inside"])) // 2]
        hit = dv["hit"][iy, ix]; Lp = np.array(line["light"]["position"])
        r2 = ((Lp - hit) ** 2).sum(); cos = (Lp - hit)[1] / np.sqrt(r2)
        assert abs(dv["E"][iy, ix] - 20.0 * cos / r2) < 1e-12


def test_b1_b2_metadata_unchanged_by_the_joint_branch():
    """B1/B2 metadata keeps its fixed/trajectory modes and single-seed block."""
    import mitsuba as mi_real
    mi_real.set_variant("scalar_rgb")
    mi, dr, torch = _stub_modules()
    with tempfile.TemporaryDirectory(prefix="b12_meta_") as root:
        for setup, cam_mode, light_mode in (("B1", "fixed", "trajectory"), ("B2", "trajectory", "fixed")):
            p = L.build_poses(setup, "spline", 11, 81)
            out = Path(root) / L.SETUPS[setup][0] / "train/spline_0001"; out.mkdir(parents=True)
            meta = L.build_metadata(out, p, "material", L.RenderConfig(), {"kind": "stub"}, mi, dr, torch, 0.0, "train", "x")
            assert (meta["camera"]["mode"], meta["light"]["mode"]) == (cam_mode, light_mode)
            assert meta["seeds"] == {"trajectory": 11, "render_base": 20260922}
            assert meta["trajectory"]["moving_element"] == ("light" if setup == "B1" else "camera")
            assert p.joint is None


def test_check_joint_sources_from_files():
    import mitsuba as mi
    mi.set_variant("scalar_rgb")
    m = _b3_small()
    mi_, dr, torch = _stub_modules()
    with tempfile.TemporaryDirectory(prefix="b3_chk_") as root:
        _fake_source_dataset(root, m)
        s = m["sequences"][1]
        p = L.poses_for_sequence(s, "B3", 81, m["master_seed"])
        out = Path(root) / "B3_Joint" / s["dir"]; out.mkdir(parents=True)
        meta = L.build_metadata(out, p, "material", L.RenderConfig(), {"kind": "stub"}, mi_, dr, torch, 0.0, s["split"], "x")
        (out / "metadata.json").write_text(json.dumps(meta))
        with (out / "frames.jsonl").open("w") as f:
            for k in range(81):
                rec = {"index": k, "seed": 0, "seconds": 0.0, "stats": {}, "png": out / f"frames_png/frame_{k:03d}.png", "exr": None}
                f.write(json.dumps(L.frame_record(rec, p, L.RenderConfig(), out), sort_keys=True) + "\n")
        r = CM.check_joint_sources(out, root)
        assert r["pass"] and r["frames"] == 81 and not r["mismatches"], r


# ------------------------------------------------------------------------------- B1/B2 untouched
# sha256 of json.dumps({"sequences", "selection"}, sort_keys=True) of the FULL B1 / B2 manifests as built
# by the code that produced the datasets on Leonardo (master seeds 20260922 / 20260923); the B3 branch
# must not move a single byte of them.
FINGERPRINTS = {"B1": "7716e8e60bb7e58f47c58e24cc484b00d46e64ac2e7926390274552ef7f4d864",   # Leonardo B1_Fixed_camera/manifest.json
                "B2": "8cee6b6d0a3c2bc62165bd1a8793ebb7fd07d00531e05cd9f8c7295d0f9e7f2b"}   # Leonardo B2_Fixed_light/manifest.json


def _fingerprint(m):
    return hashlib.sha256(json.dumps({"sequences": m["sequences"], "selection": m["selection"]},
                                     sort_keys=True).encode()).hexdigest()


def test_b1_b2_full_manifests_rebuild_identically():
    for setup, seed in (("B1", 20260922), ("B2", 20260923)):
        fp = _fingerprint(M.build(master_seed=seed, verbose=False, setup=setup))
        assert fp == FINGERPRINTS[setup], f"{setup} manifest changed: {fp}"


def test_outline_distance_is_distance_to_projected_quad():
    """check_metadata.outline_distance_px: 0 on the projected edges, = Euclidean distance to a corner
    outside it, and the pixel-centre inside mask agrees with the projected quadrilateral."""
    import mitsuba as mi
    mi.set_variant("scalar_rgb")
    m = _b3_small()
    s = m["sequences"][0]
    p = L.poses_for_sequence(s, "B3", 81, m["master_seed"])
    mi_, dr, torch = _stub_modules()
    with tempfile.TemporaryDirectory(prefix="b3_outline_") as root:
        out = Path(root) / "B3_Joint" / s["dir"]; out.mkdir(parents=True)
        meta = L.build_metadata(out, p, "material", L.RenderConfig(), {"kind": "stub"}, mi_, dr, torch, 0.0, s["split"], "x")
    for idx in (0, 40, 80):
        rec = {"camera": {"c2w": L.S.transform_to_list(L.S.look_at_matrix(p.cam_pos[idx]))},
               "light": {"position": p.light_pos[idx].tolist()}}
        cs = CM.corners_px(meta, rec)
        a, b = np.array(cs[0]), np.array(cs[1])
        mid = (a + b) / 2 - 0.5                        # a pixel whose centre is the edge midpoint
        assert CM.outline_distance_px(meta, rec, [mid[1]], [mid[0]])[0] < 1e-6
        out_pt = a + 3.0 * (a - (np.array(cs[2]) + a) / 2) / np.linalg.norm(a - (np.array(cs[2]) + a) / 2)
        d = CM.outline_distance_px(meta, rec, [out_pt[1] - 0.5], [out_pt[0] - 0.5])[0]
        assert abs(d - 3.0) < 1e-6, d                   # beyond corner a along the diagonal
        dv = CM.derive(meta, rec)
        ys, xs = np.nonzero(dv["inside"])
        far_inside = CM.outline_distance_px(meta, rec, ys, xs) > 0.01
        assert far_inside.mean() > 0.95                 # mask pixels lie inside the quad
