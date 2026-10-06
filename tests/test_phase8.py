"""Phase 8 tests — sub-pixel refinement contenders, Gate 3, ABSTAIN codes.

All synthetic; no GPU/torch required.
"""

import numpy as np
import pytest

from chandra_align.refine.subpixel import refine_lk, refine_phasecorr
from chandra_align.refine import verify_guarded
from chandra_align.metrics.conditioning import (
    check_transform_conditioning,
    classify_prefit_abstain,
    dedup_correspondences,
)
from chandra_align.metrics import validate_registration_gate
from chandra_align.testing import make_pair_shift


@pytest.fixture()
def shifted():
    ref, mov, _ = make_pair_shift(shape=(256, 256), dx=5.0, dy=2.0, seed=3)
    rng = np.random.default_rng(1)
    pa = rng.uniform(30, 220, size=(30, 2))
    pb = pa + np.array([5.0, 2.0]) + rng.normal(0, 0.3, size=(30, 2))
    return ref, mov, pa, pb


CFG = {"ransac_reproj_threshold": 3.0, "max_iters": 2000, "confidence": 0.99}


def test_verify_guarded_accepts_clean_fit(shifted):
    ref, mov, pa, pb = shifted
    res = verify_guarded(pa, pb, CFG, image_shape=ref.shape)
    assert res["ok"] is True
    assert res["abstain_code"] is None
    assert res["n_unique"] <= res["n_raw"]
    assert res["gate3"]["ok"] is True


def test_abstain_codes():
    z = np.zeros((0, 2))
    assert classify_prefit_abstain(z, z) == "ZERO_CANDIDATES"
    pa = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    assert classify_prefit_abstain(pa, pa) == "INSUFFICIENT_UNIQUE"
    line = np.column_stack([np.linspace(0, 100, 20), np.linspace(0, 100, 20)])
    assert classify_prefit_abstain(line, line + 1.0) == "COLLINEAR"


def test_abstain_short_circuits_guarded():
    res = verify_guarded(np.zeros((0, 2)), np.zeros((0, 2)), CFG)
    assert res["ok"] is False
    assert res["abstain_code"] == "ZERO_CANDIDATES"
    assert res["model"] is None


def test_gate3_rejects_collapse_signature():
    # The 1.24e-20 scale collapse seen in the IIRS SIFT fallback.
    M = np.array([[1.24e-20, 0.0, 5.0], [0.0, 1.24e-20, 3.0]])
    ok, rep = check_transform_conditioning(M)
    assert ok is False
    assert rep["checks"]["det_gt_1e-4"] is False


def test_gate3_rejects_extreme_anisotropy():
    M = np.array([[100.0, 0.0, 0.0], [0.0, 0.01, 0.0]])
    ok, rep = check_transform_conditioning(M)
    assert ok is False
    assert rep["checks"]["scale_ratio_lt_20"] is False


def test_gate3_accepts_identity_like():
    M = np.array([[1.0, 0.01, 5.0], [-0.01, 1.0, -3.0]])
    ok, rep = check_transform_conditioning(M, rmse_px=0.4)
    assert ok is True


def test_gate_model_backstop_rejects_degenerate():
    # A degenerate model fails the gate even when residual tiers would pass.
    M = np.array([[1e-12, 0.0, 0.0], [0.0, 1e-12, 0.0]])
    msg, code = validate_registration_gate(
        0.3, 20, 8, 1.5, {"Q1": 5, "Q2": 5, "Q3": 5, "Q4": 5}, model=M)
    assert code == "DEGENERATE_FAILURE"
    assert "Gate 3" in msg


def test_gate_without_model_unchanged():
    msg, code = validate_registration_gate(
        0.3, 20, 8, 1.5, {"Q1": 5, "Q2": 5, "Q3": 5, "Q4": 5})
    assert code == "SUCCESS_SUBPIXEL"


def test_lk_keeps_and_does_not_hurt(shifted):
    ref, mov, pa, pb = shifted
    res = verify_guarded(pa, pb, CFG, image_shape=ref.shape)
    assert res["ok"]
    ka, rb, st = refine_lk(ref, mov, res["inliers_a"], res["inliers_b"],
                           model=res["model"])
    assert st["refined_pairs"] > 0
    # refined residual must not exceed input residual (keep iff no worse)
    assert st["residual_mean_out_px"] <= st["residual_mean_in_px"] + 1e-6


def test_phasecorr_returns_points(shifted):
    ref, mov, pa, pb = shifted
    res = verify_guarded(pa, pb, CFG, image_shape=ref.shape)
    assert res["ok"]
    ka, rb, st = refine_phasecorr(ref, mov, res["inliers_a"][:10],
                                 res["inliers_b"][:10])
    assert st["refined_pairs"] > 0
    assert ka.shape == rb.shape
    assert st["peak_height_mean"] > 0


def test_dedup_merges_close_points():
    pa = np.array([[0.0, 0.0], [1.0, 1.0], [50.0, 50.0]])
    pb = pa + 1.0
    ka, kb, idx, n_raw, n_unique = dedup_correspondences(pa, pb, radius=2.0)
    assert n_raw == 3 and n_unique == 2


def test_span_guard_relaxes_threshold():
    # Inliers clustered in a thin Y band trigger the span guard refit.
    rng = np.random.default_rng(5)
    pa = np.column_stack([rng.uniform(30, 220, 30), rng.uniform(100, 130, 30)])
    pb = pa + np.array([5.0, 2.0]) + rng.normal(0, 0.5, size=(30, 2))
    res = verify_guarded(pa, pb, CFG, image_shape=(512, 512))
    assert res["span_guard"] is True
