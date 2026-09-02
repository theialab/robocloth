"""
B6 — observation RGB sampling is NEAREST pixel (np.round + clip), not bilinear.

The released observations_structured.npz files were produced with this
lookup, so `sample_pixels_nearest` in reconstruction/reconstruct.py is the
source of truth and docs/capture_pipeline.md (step 5) describes it. These
tests pin the ACTUAL numpy semantics the code has: np.round is IEEE
round-half-to-even (1.5 -> 2, 2.5 -> 2, 0.5 -> 0), then clip to the image.

Run:  <python> -m unittest discover -s tests/reconstruction -v
(needs numpy + hydra/omegaconf for the reconstruct.py import; no data, no GPU)
"""
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

RECON_DIR = Path(__file__).resolve().parents[2] / "reconstruction"
if str(RECON_DIR) not in sys.path:
    sys.path.insert(0, str(RECON_DIR))
import reconstruct  # noqa: E402  (import only — hydra main() is not invoked)


class NearestPixelSamplingTest(unittest.TestCase):
    def setUp(self):
        # 3x3 RGB image with a unique triple per pixel: (10r+c, 100+10r+c, 200+10r+c)
        self.img = np.zeros((3, 3, 3), dtype=np.float32)
        for r in range(3):
            for c in range(3):
                self.img[r, c] = [10 * r + c, 100 + 10 * r + c, 200 + 10 * r + c]

    def test_numpy_round_is_half_to_even(self):
        # Documented semantics the sampler inherits from np.round.
        self.assertEqual(np.round(0.5), 0.0)
        self.assertEqual(np.round(1.5), 2.0)
        self.assertEqual(np.round(2.5), 2.0)
        self.assertEqual(np.round(1.49), 1.0)
        self.assertEqual(np.round(1.51), 2.0)

    def test_sub_pixel_coordinates_pick_rounded_pixel(self):
        # ((px, py), (expected col, expected row)) under np.round + clip to [0, 2]
        cases = [
            ((1.49, 0.49), (1, 0)),
            ((1.51, 0.51), (2, 1)),
            ((1.5, 0.5), (2, 0)),    # halves go to the nearest EVEN integer
            ((2.5, 1.5), (2, 2)),    # 2.5 -> 2, 1.5 -> 2
            ((0.5, 2.5), (0, 2)),
            ((-0.7, 0.0), (0, 0)),   # rounds to -1, clipped to 0
            ((5.2, 3.9), (2, 2)),    # rounds to 5 / 4, clipped to 2
        ]
        px = np.array([c[0][0] for c in cases])
        py = np.array([c[0][1] for c in cases])
        got = reconstruct.sample_pixels_nearest(self.img, px, py)
        self.assertEqual(got.shape, (len(cases), 3))
        for (coord, (col, row)), value in zip(cases, got):
            np.testing.assert_array_equal(value, self.img[row, col], err_msg=f"(px, py)={coord}")

    def test_not_bilinear(self):
        # Halfway between pixel (0,0) and (0,1) bilinear would return their
        # mean (0.5, 100.5, 200.5); nearest-pixel returns pixel (0,0) exactly.
        got = reconstruct.sample_pixels_nearest(self.img, np.array([0.5]), np.array([0.0]))
        np.testing.assert_array_equal(got[0], self.img[0, 0])
        # dtype is the image's dtype (an interpolating sampler would promote uint16 to float)
        img16 = (self.img * 100).astype(np.uint16)
        got16 = reconstruct.sample_pixels_nearest(img16, np.array([1.5, 0.49]), np.array([1.5, 2.51]))
        self.assertEqual(got16.dtype, np.uint16)
        np.testing.assert_array_equal(got16, img16[[2, 2], [2, 0]])

    def test_matches_original_inline_expression(self):
        # The helper was factored out of two call sites in reconstruct.py that
        # inlined exactly this expression; results must be identical for both
        # the float32 (extract_observations) and float64 (reprojection) inputs.
        rng = np.random.default_rng(0)
        H, W = 37, 53
        img = rng.integers(0, 65535, size=(H, W, 3)).astype(np.uint16)
        for dtype in (np.float32, np.float64):
            px = rng.uniform(-3, W + 3, 5000).astype(dtype)
            py = rng.uniform(-3, H + 3, 5000).astype(dtype)
            x = np.clip(np.round(px).astype(np.int32), 0, W - 1)
            y = np.clip(np.round(py).astype(np.int32), 0, H - 1)
            np.testing.assert_array_equal(reconstruct.sample_pixels_nearest(img, px, py), img[y, x])

    def test_both_call_sites_use_the_helper(self):
        src = inspect.getsource(reconstruct)
        # definition + two call sites
        self.assertGreaterEqual(src.count("sample_pixels_nearest("), 3)
        # no stray inline rounding left behind at the old sites
        self.assertNotIn("np.round(pixels[:, 0])", src)
        self.assertNotIn("np.round(pixels_x[valid_indices])", src)


if __name__ == "__main__":
    unittest.main()
