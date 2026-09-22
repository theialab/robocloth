"""Structure/determinism tests for make_manifest.py (run: python tests/run_tests.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from synthetic_sequences import make_manifest as M  # noqa: E402


# Reduced pool/quotas so the test costs a second, not ten. The real values are exercised by the
# production run itself (make_manifest.py prints the same counts and coverage table).
SMALL_POOL = {"spline": 60, "highlight_sweep": 20}
SMALL_QUOTAS = {"spline": 17, "highlight_sweep": 3}
SMALL_TEST = {"spline": 3, "highlight_sweep": 1, "ring": 2, "spiral": 2, "lissajous": 1, "spiral_reference": 1}


def _small(**kw):
    saved = (M.POOL, M.TRAIN_QUOTAS, M.TEST_COUNTS)
    M.POOL, M.TRAIN_QUOTAS, M.TEST_COUNTS = SMALL_POOL, SMALL_QUOTAS, SMALL_TEST
    try:
        return M.build(verbose=False, **kw)
    finally:
        M.POOL, M.TRAIN_QUOTAS, M.TEST_COUNTS = saved


def test_deterministic_given_master_seed():
    a, b = _small(master_seed=7), _small(master_seed=7)
    assert a["sequences"] == b["sequences"]
    assert _small(master_seed=8)["sequences"] != a["sequences"]


def test_test_seeds_are_unseen_in_training():
    m = _small()
    train = {s["seed"] for s in m["sequences"] if s["split"] == "train"}
    # the whole training POOL, not just the selected 900, must be disjoint from the test seeds
    pool = set()
    for cls, n in SMALL_POOL.items():
        pool |= {M.BLOCK[f"train_{cls}"] + m["master_seed"] + i for i in range(n)}
    test = {s["seed"] for s in m["sequences"] if s["split"] == "test"}
    assert train <= pool and not (test & pool) and len(test) == sum(SMALL_TEST.values())


def test_ids_unique_and_quotas_respected():
    m = _small()
    dirs = [s["dir"] for s in m["sequences"]]
    assert len(set(dirs)) == len(dirs)
    assert all(s["dir"] == f"{s['split']}/{s['id']}" for s in m["sequences"])
    for cls, k in SMALL_QUOTAS.items():
        assert sum(s["split"] == "train" and s["class"] == cls for s in m["sequences"]) == k
    for cls, k in SMALL_TEST.items():
        assert sum(s["split"] == "test" and s["class"] == cls for s in m["sequences"]) == k


def test_exr_policy():
    m = _small()
    train = [s for s in m["sequences"] if s["split"] == "train"]
    assert all(s["keep_exr"] for s in m["sequences"] if s["split"] == "test")
    assert [s["index"] for s in train if s["keep_exr"]] == list(range(0, len(train), M.EXR_EVERY_N_TRAIN))


def test_greedy_covers_better_than_random():
    cov = _small()["selection"]["coverage_stats"]
    assert cov["greedy"]["cv"] < cov["random_baseline"]["cv"]
