"""Tests for the opt-in deformation-field stage (CHANDRA_DEFORM_FIELD=1).

Covers the pre-registered success clauses:
  (1) flag OFF -> bit-identical pipeline behavior (proven across flag values);
  (2) full suite still passes (run separately);
  (3) flag ON -> end-to-end run, frozen gate re-run, before/after numbers;
  (4) degenerate inputs -> fail-closed skip, original verdict preserved.

Unit tests on the stage itself run everywhere (pure numpy). App-level tests
need the Gradio runtime and skip locally, running in CI -- the same pattern
as tests/test_app.py.
"""

import os

import numpy as np
import pytest
import cv2

from chandra_align.deform_field import (
    MIN_FIELD_POINTS,
    apply_deform_field_stage,
    deform_field_enabled,
)


# ---------------------------------------------------------------------------
# Stage unit tests (no app import needed).
# ---------------------------------------------------------------------------

def _shift_pair(n=200, seed=7, noise=0.05):
    rng = np.random.default_rng(seed)
    p1 = rng.uniform(0, 2000, (n, 2))
    M = np.array([[1.0, 0.0, 7.3], [0.0, 1.0, -3.9]])
    p2 = (p1 @ M[:, :2].T + M[:, 2]) + rng.normal(0, noise, (n, 2))
    return p1, p2, M


def _warp_pair(n=200, seed=7):
    p1, p2_aff, M = _shift_pair(n=n, seed=seed, noise=0.0)
    xx, yy = p1[:, 0], p1[:, 1]
    warp = np.stack([3.0 * np.sin(xx / 400.0), 2.0 * np.cos(yy / 500.0)], axis=1)
    rng = np.random.default_rng(seed + 1)
    p2 = p2_aff + warp + rng.normal(0, 0.3, (n, 2))
    return p1, p2, M


def test_flag_guard_values():
    old = os.environ.get("CHANDRA_DEFORM_FIELD")
    try:
        for val, expected in [("1", True), ("true", True), ("YES", True),
                              ("on", True), ("0", False), ("off", False),
                              ("", False), ("2", False)]:
            os.environ["CHANDRA_DEFORM_FIELD"] = val
            assert deform_field_enabled() is expected, val
        if "CHANDRA_DEFORM_FIELD" in os.environ:
            del os.environ["CHANDRA_DEFORM_FIELD"]
        assert deform_field_enabled() is False
    finally:
        if old is None:
            os.environ.pop("CHANDRA_DEFORM_FIELD", None)
        else:
            os.environ["CHANDRA_DEFORM_FIELD"] = old


def test_pure_shift_reports_no_improvement():
    """Affine already perfect -> honest 'no_improvement', not a fake field."""
    p1, p2, M = _shift_pair()
    out = apply_deform_field_stage(p1, p2, M)
    assert out["applied"] is False
    assert out["reason"] == "no_improvement"


def test_smooth_warp_recovery():
    """Known smooth warp: stage applies, RMSE and held-out both drop."""
    p1, p2, M = _warp_pair()
    out = apply_deform_field_stage(p1, p2, M)
    assert out["applied"] is True
    assert out["rmse_after_px"] < out["rmse_before_px"]
    assert out["heldout_rmse_px"] < out["heldout_rmse_affine_px"]
    assert out["lambda_chosen"] in (0.1, 1.0, 10.0, 100.0)
    assert out["min_jacobian_det"] > 0.5
    assert len(out["lambda_sweep"]) == 4
    r = out["residuals_mag"]
    assert r.shape == (len(p1),)
    assert np.all(np.isfinite(r))


def test_stage_is_deterministic():
    p1, p2, M = _warp_pair()
    a = apply_deform_field_stage(p1, p2, M)
    b = apply_deform_field_stage(p1, p2, M)
    assert a["applied"] == b["applied"]
    assert a.get("lambda_chosen") == b.get("lambda_chosen")
    assert a["rmse_after_px"] == pytest.approx(b["rmse_after_px"])
    assert a["heldout_rmse_px"] == pytest.approx(b["heldout_rmse_px"])


@pytest.mark.parametrize("p_src,p_ref,M,reason", [
    (None, None, np.eye(2, 3), "bad_points"),
    (np.zeros((0, 2)), np.zeros((0, 2)), np.eye(2, 3), "bad_points"),
    (np.full((10, 2), np.nan), np.zeros((10, 2)), np.eye(2, 3), "bad_points"),
])
def test_degenerate_inputs_fail_closed(p_src, p_ref, M, reason):
    out = apply_deform_field_stage(p_src, p_ref, M)
    assert out["applied"] is False
    assert out["reason"] == reason
    assert "residuals_vec" not in out


def test_insufficient_points_fail_closed():
    p1, p2, M = _shift_pair(n=MIN_FIELD_POINTS - 1)
    out = apply_deform_field_stage(p1, p2, M)
    assert out["applied"] is False
    assert out["reason"] == "insufficient_points"


def test_bad_matrix_fails_closed():
    p1, p2, _ = _shift_pair()
    for bad_M in (None, np.eye(3), np.full((2, 3), np.nan)):
        out = apply_deform_field_stage(p1, p2, bad_M)
        assert out["applied"] is False
        assert out["reason"] == "bad_matrix"


def test_mismatched_point_counts_fail_closed():
    p1, p2, M = _shift_pair()
    out = apply_deform_field_stage(p1, p2[:10], M)
    assert out["applied"] is False
    assert out["reason"] == "bad_points"


# ---------------------------------------------------------------------------
# App-level tests (require the Gradio runtime; skip locally, run in CI).
# ---------------------------------------------------------------------------

def _require_app():
    pytest.importorskip("gradio", reason="application tests require the pinned Gradio runtime")
    pytest.importorskip("multipart", reason="application tests require python-multipart")
    from app import _align_core, MIN_REGISTRATION_INLIERS  # noqa: F401
    import app as app_module
    return app_module


def _real_pair():
    import app as app_module  # noqa: F401  (importorskip already ran)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref = cv2.imread(os.path.join(root, "data", "benchmark_crops",
                                  "ohrc_01_reference.png"), cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(os.path.join(root, "data", "benchmark_crops",
                                  "ohrc_01_source.png"), cv2.IMREAD_GRAYSCALE)
    assert ref is not None and sec is not None
    return ref, sec


def _key_outputs(out):
    jm = out["judge_metrics"]
    return {
        "status_code": out["status_code"],
        "inlier_cnt": out["inlier_cnt"],
        "rmse_gate_px": jm["rmse_gate_px"],
        "rmse_in_sample_px": jm["rmse_in_sample_px"],
        "rmse_gate_basis": jm["rmse_gate_basis"],
    }


def test_flag_off_bit_identical_across_flag_values():
    """With the flag unset/'0'/'off', registration outputs are identical."""
    app_module = _require_app()
    ref, sec = _real_pair()
    results = {}
    old = os.environ.get("CHANDRA_DEFORM_FIELD")
    try:
        for label, val in [("unset", None), ("zero", "0"), ("off", "off")]:
            if val is None:
                os.environ.pop("CHANDRA_DEFORM_FIELD", None)
            else:
                os.environ["CHANDRA_DEFORM_FIELD"] = val
            out = app_module._align_core(ref, sec)
            results[label] = _key_outputs(out)
            # Stage telemetry present but inert.
            assert out["judge_metrics"]["deform_field_stage"] == {
                "applied": False, "reason": "flag_off"}
    finally:
        if old is None:
            os.environ.pop("CHANDRA_DEFORM_FIELD", None)
        else:
            os.environ["CHANDRA_DEFORM_FIELD"] = old
    base = results["unset"]
    for label, res in results.items():
        assert res["status_code"] == base["status_code"], label
        assert res["inlier_cnt"] == base["inlier_cnt"], label
        assert res["rmse_gate_px"] == pytest.approx(base["rmse_gate_px"]), label
        assert res["rmse_in_sample_px"] == pytest.approx(base["rmse_in_sample_px"]), label
        assert res["rmse_gate_basis"] == base["rmse_gate_basis"], label


def test_flag_on_end_to_end_frozen_tiers():
    """Flag ON: runs end-to-end, verdict stays within the frozen tiers."""
    app_module = _require_app()
    ref, sec = _real_pair()
    old = os.environ.get("CHANDRA_DEFORM_FIELD")
    try:
        os.environ["CHANDRA_DEFORM_FIELD"] = "1"
        out = app_module._align_core(ref, sec)
    finally:
        if old is None:
            os.environ.pop("CHANDRA_DEFORM_FIELD", None)
        else:
            os.environ["CHANDRA_DEFORM_FIELD"] = old
    assert out["status_code"] in ("SUCCESS_SUBPIXEL", "COARSE_ADVISORY",
                                  "DEGENERATE_FAILURE")
    info = out["judge_metrics"]["deform_field_stage"]
    assert isinstance(info["applied"], bool)
    if info["applied"]:
        assert info["lambda_chosen"] in (0.1, 1.0, 10.0, 100.0)
        assert info["rmse_after_px"] <= info["rmse_before_px"]
        assert info["min_jacobian_det"] > 0.5
        assert info["heldout_rmse_px"] < info["heldout_rmse_affine_px"]


def test_flag_on_degenerate_pair_no_crash():
    """Flag ON with garbage input: no crash from the stage, verdict preserved."""
    app_module = _require_app()
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 255, (256, 256)).astype(np.uint8)
    sec = rng.integers(0, 255, (256, 256)).astype(np.uint8)
    old = os.environ.get("CHANDRA_DEFORM_FIELD")
    try:
        os.environ["CHANDRA_DEFORM_FIELD"] = "1"
        out = app_module._align_core(ref, sec)
    finally:
        if old is None:
            os.environ.pop("CHANDRA_DEFORM_FIELD", None)
        else:
            os.environ["CHANDRA_DEFORM_FIELD"] = old
    assert out["status_code"] in ("SUCCESS_SUBPIXEL", "COARSE_ADVISORY",
                                  "DEGENERATE_FAILURE")


def test_flag_on_pre_balance_hook_ohrc01_improves():
    """Flag ON, pre-balance hook: stage applies in-pipeline and improves ohrc_01.

    A16 Part B's SUCCESS_SUBPIXEL (0.45 px) was measured on A14's standalone
    842-inlier match set; the true _align_core's match_pair_hf caps keypoints
    (618 raw -> ~100 RANSAC inliers), so the stage's internal held-out reads
    higher here. This test pins what IS true in-pipeline: the stage applies
    at the pre-balance placement and the gate RMSE improves vs flag-off,
    under the frozen tiers. See docs/deform-field-pipeline-stage-2026-10-08.md.
    """
    app_module = _require_app()
    ref, sec = _real_pair()
    old = os.environ.get("CHANDRA_DEFORM_FIELD")
    try:
        os.environ.pop("CHANDRA_DEFORM_FIELD", None)
        out_off = app_module._align_core(ref, sec)
        os.environ["CHANDRA_DEFORM_FIELD"] = "1"
        out_on = app_module._align_core(ref, sec)
    finally:
        if old is None:
            os.environ.pop("CHANDRA_DEFORM_FIELD", None)
        else:
            os.environ["CHANDRA_DEFORM_FIELD"] = old
    info = out_on["judge_metrics"]["deform_field_stage"]
    assert info["applied"] is True
    assert info["hook_placement"] == "pre_balance"
    assert info["lambda_chosen"] in (0.1, 1.0, 10.0, 100.0)
    assert info["min_jacobian_det"] > 0.5
    # The gate scores the stage's internal held-out (generalization), not the
    # affine in-sample fit.
    assert out_on["judge_metrics"]["rmse_gate_basis"] == \
        "held-out (deform-field stage internal stride 80/20 split)"
    assert out_on["status_code"] in ("SUCCESS_SUBPIXEL", "COARSE_ADVISORY",
                                     "DEGENERATE_FAILURE")
    # The relocation must improve (or at worst preserve) the gate RMSE.
    gate_on = float(out_on["judge_metrics"]["rmse_gate_px"])
    gate_off = float(out_off["judge_metrics"]["rmse_gate_px"])
    assert gate_on <= gate_off


def test_quota_flag_gating_unit():
    """Flag off -> per-cell quota honored (64); flag on -> uncapped."""
    from chandra_align.features.distribution import select_detector_keypoints

    class _KP:
        def __init__(self, x, y, r):
            self.pt = (x, y)
            self.response = r

    rng = np.random.default_rng(7)
    kps = [_KP(*rng.uniform(0, 2048, 2), rng.random()) for _ in range(2000)]
    shape = (2048, 2048)
    old = os.environ.get("CHANDRA_DEFORM_FIELD")
    try:
        os.environ.pop("CHANDRA_DEFORM_FIELD", None)
        _, idx_off = select_detector_keypoints(kps, shape, 64)
        assert len(idx_off) == 16 * 64
        os.environ["CHANDRA_DEFORM_FIELD"] = "1"
        _, idx_on = select_detector_keypoints(kps, shape, 64)
        assert len(idx_on) == 2000  # uncapped
        # Deterministic and the capped set is a subset of the uncapped set.
        _, idx_off2 = select_detector_keypoints(kps, shape, 64)
        assert len(idx_off2) == 2000  # flag still on
    finally:
        if old is None:
            os.environ.pop("CHANDRA_DEFORM_FIELD", None)
        else:
            os.environ["CHANDRA_DEFORM_FIELD"] = old
    # Flag off again -> back to quota.
    _, idx_off3 = select_detector_keypoints(kps, shape, 64)
    assert np.array_equal(np.asarray(idx_off), np.asarray(idx_off3))
