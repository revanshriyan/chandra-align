"""Phase 11 tests — robustness battery.

- Perturbations are deterministic (same seed -> identical output) and
  seed-sensitive.
- Rotation GT composition is geometrically correct (recovered transform
  agrees with composed truth).
- Battery runs end-to-end on synthetic pairs; JSON schema has required fields.
- Re-runs reproduce results exactly.
- Too-few-points withholds independent accuracy instead of faking it.
"""

import json

import numpy as np
import pytest

from chandra_align.eval.robustness import (
    PERTURBATIONS,
    compose_gt_with_warp,
    independent_accuracy,
    measure_pair,
    median_inlier_error,
    overlap_coverage,
)
from chandra_align.testing import make_pair_degenerate, make_pair_shift

REQUIRED_FIELDS = {
    "base_pair", "perturbation", "params", "seed",
    "n_det_a", "n_det_b", "n_lowe", "abstain", "verdict",
    "rmse_px", "median_resid_px", "inliers", "entropy",
    "overlap_coverage", "independent_accuracy_px", "accuracy_note",
}

TINY = (256, 256)


def test_perturbation_determinism():
    rng_img = np.random.default_rng(0).uniform(0, 255, (64, 64))
    for name, fn in PERTURBATIONS.items():
        kwargs = {}
        if name == "gaussian_noise":
            kwargs = {"sigma": 5.0}
        elif name == "brightness_shift":
            kwargs = {"delta": 10.0}
        elif name == "contrast_gain":
            kwargs = {"gain": 1.2}
        elif name == "gamma_curve":
            kwargs = {"gamma": 1.3}
        elif name == "small_rotation":
            kwargs = {"angle_deg": 2.0}
        elif name == "jpeg_artifacts":
            kwargs = {"quality": 50}
        a1, _ = fn(rng_img, seed=7, **kwargs)
        a2, _ = fn(rng_img, seed=7, **kwargs)
        assert np.array_equal(a1, a2), f"{name} not deterministic"
    # only the stochastic perturbation must be seed-sensitive; the rest are
    # parameter-determined transforms that legitimately ignore the seed
    g1, _ = PERTURBATIONS["gaussian_noise"](rng_img, sigma=5.0, seed=7)
    g2, _ = PERTURBATIONS["gaussian_noise"](rng_img, sigma=5.0, seed=8)
    assert not np.array_equal(g1, g2)


def test_rotation_gt_composition_geometric():
    # Rotating the moving image must compose the truth as M_gt @ dM.
    ref, mov, M_gt = make_pair_shift(shape=TINY, dx=5.0, dy=-2.0, seed=7)
    mov_r, info = PERTURBATIONS["small_rotation"](mov, angle_deg=2.0, seed=0)
    M_new = compose_gt_with_warp(M_gt, np.asarray(info["dM"]))
    rec = measure_pair(ref, mov_r, M_new, image_shape=TINY, sift_nfeatures=2000)
    assert rec["verdict"] not in ("ABSTAIN",), f"pipeline failed: {rec['abstain']}"
    assert rec["independent_accuracy_px"] is not None
    assert rec["independent_accuracy_px"] < 2.0


def test_sun_flip_is_inversion():
    img = np.random.default_rng(1).uniform(0, 255, (16, 16))
    out, _ = PERTURBATIONS["sun_flip"](img, seed=0)
    assert np.allclose(out, 255.0 - img)


def test_measure_pair_schema_and_clean_baseline():
    ref, mov, M_gt = make_pair_shift(shape=TINY, seed=7)
    rec = measure_pair(ref, mov, M_gt, image_shape=TINY, sift_nfeatures=2000)
    # runner adds base_pair/perturbation/params/seed; measure_pair provides the rest
    assert (REQUIRED_FIELDS - {"base_pair", "perturbation", "params", "seed"}
            ).issubset(rec.keys())
    assert rec["verdict"] == "SUCCESS_SUBPIXEL"
    assert rec["median_resid_px"] is not None
    assert 0.0 <= rec["overlap_coverage"] <= 1.0
    assert rec["overlap_coverage"] > 0.8
    assert rec["independent_accuracy_px"] is not None
    assert rec["independent_accuracy_px"] < 1.0


def test_withheld_accuracy_on_degenerate():
    a, b, M_none = make_pair_degenerate(shape=TINY, seed=7)
    rec = measure_pair(a, b, np.eye(2, 3), image_shape=TINY, sift_nfeatures=2000)
    assert rec["independent_accuracy_px"] is None
    assert "no independent accuracy available" in rec["accuracy_note"]


def test_median_error_with_empty():
    assert median_inlier_error(np.array([])) is None
    assert median_inlier_error([0.1, 0.2, 10.0]) == pytest.approx(0.2)


def test_overlap_coverage_bounds():
    ref, mov, M_gt = make_pair_shift(shape=TINY, seed=7)
    rec = measure_pair(ref, mov, M_gt, image_shape=TINY, sift_nfeatures=2000)
    assert 0.0 <= rec["overlap_coverage"] <= 1.0
    # empty inliers -> zero coverage, not an error
    assert overlap_coverage(np.zeros((0, 2)), M_gt, TINY) == 0.0


def test_battery_rerun_reproducible():
    # Two end-to-end runs on a small battery must be byte-identical.
    from chandra_align.eval.robustness import BATTERY_VERSION
    ref, mov, M_gt = make_pair_shift(shape=TINY, seed=7)

    def run_once():
        rows = []
        for i, (pname, params) in enumerate(
                [("identity", {}), ("gaussian_noise", {"sigma": 5.0}),
                 ("sun_flip", {})]):
            fn = PERTURBATIONS[pname]
            mov_p, info = fn(mov, seed=500 + i, **params)
            M_run = compose_gt_with_warp(M_gt, np.asarray(info["dM"])) \
                if "dM" in info else M_gt
            rec = measure_pair(ref, mov_p, M_run, image_shape=TINY,
                               sift_nfeatures=2000)
            rec.update({"base_pair": "shift", "perturbation": pname,
                        "params": params, "seed": 500 + i})
            rows.append(rec)
        return json.dumps({"battery_version": BATTERY_VERSION, "runs": rows},
                          indent=2, sort_keys=True)

    assert run_once() == run_once()
