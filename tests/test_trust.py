"""Trust-flag tests (M6) with the circularity guard."""

import numpy as np

from chandra_align.trust import (evaluate, split_fit_holdout, trust_flag,
                                 calibration_status)


def test_degenerate_pair_flagged_not_trusted(cfg):
    """The featureless pair must get an explicit Not Trusted flag.

    Zero matches -> 0.0 inlier ratio, 0.0 uniformity, RMSE UNMEASURED.
    """
    n_raw, n_inliers = 0, 0
    inlier_ratio = 0.0
    uniformity = 0.0
    flag = trust_flag("UNMEASURED", inlier_ratio, uniformity, cfg["trust"])
    assert flag == "Not Trusted"


def test_trusted_when_all_thresholds_met(cfg):
    """A good pair (enough inliers, spread, low held-out RMSE) is Trusted."""
    flag = trust_flag(0.22, inlier_ratio=0.61, uniformity=0.55, cfg_trust=cfg["trust"])
    assert flag == "Trusted"


def test_trust_flag_fails_on_low_rmse_only():
    """A low RMSE alone is NOT enough — uniformity + inlier ratio also gate."""
    cfg_t = {"inlier_ratio_threshold": 0.15, "uniformity_threshold": 0.125,
             "rmse_px_threshold": 0.5}
    assert trust_flag(0.1, 0.60, 0.05, cfg_t) == "Not Trusted"   # uniformity fails
    assert trust_flag(0.1, 0.05, 0.60, cfg_t) == "Not Trusted"   # ratio fails
    assert trust_flag(1.2, 0.60, 0.60, cfg_t) == "Not Trusted"   # rmse fails


def test_calibration_status_label(cfg):
    assert calibration_status(cfg["trust"]) == "UNCALIBRATED"


def test_circularity_guard_split():
    rng = np.random.default_rng(3)
    pa = rng.uniform(0, 512, (60, 2))
    pb = pa + rng.normal(0, 0.1, (60, 2))
    fa, fb, ha, hb = split_fit_holdout(pa, pb, holdout_fraction=0.2)
    assert len(ha) + len(fa) == len(pa)
    assert len(ha) >= 1
    # held-out points must not intersect fit points
    fa_set = {tuple(r) for r in fa}
    ha_set = {tuple(r) for r in ha}
    assert fa_set.intersection(ha_set) == set()


def test_circularity_guard_evaluate():
    rng = np.random.default_rng(4)
    pa = rng.uniform(0, 512, (80, 2))
    M_true = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -7.0]])
    pb = np.column_stack([pa[:, 0] + M_true[0, 2], pa[:, 1] + M_true[1, 2]])
    M, rm = evaluate(M_true, pa, pb)
    assert rm["held_out"]
    assert rm["rmse_px"] != "UNMEASURED"
    assert rm["rmse_px"] < 0.5
    assert rm["n_check_points"] >= 1


def test_circularity_guard_never_reports_fit_set_rmse():
    """With <5 points, held-out is empty -> RMSE must be UNMEASURED, not the fit RMSE."""
    pa = np.array([[0, 0], [10, 10], [20, 0]], dtype=float)
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    M_ref, rm = evaluate(M, pa, pa)
    assert rm["rmse_px"] == "UNMEASURED"


def test_heldout_rmse_is_reported_separately_from_fit_residuals():
    """The evaluator computes held-out error from check points disjoint from fit points."""
    from chandra_align.metrics import rmse_heldout

    fit_a = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
    fit_b = fit_a + np.array([2.0, -3.0])
    hold_a = np.array([[20, 20], [30, 40]], dtype=float)
    hold_b = hold_a + np.array([2.25, -2.0])
    matrix = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -3.0]])
    held = rmse_heldout(matrix, hold_a, hold_b)

    assert held["held_out"] is True
    assert held["n_check_points"] == len(hold_a)
    assert held["rmse_px"] == np.sqrt(0.25**2 + 1.0**2)
    assert held["mae_px"] == np.sqrt(0.25**2 + 1.0**2)
    assert held["mae_m"] == held["mae_px"]
    assert not any(np.array_equal(point, fit) for point in hold_a for fit in fit_a)


def test_heldout_rmse_is_explicit_in_gate_telemetry_and_dossier():
    from app import format_telemetry_report
    from chandra_align.visualization import create_combined_visualization

    telemetry = format_telemetry_report(
        status="SUCCESS_SUBPIXEL", inliers=20, total_pts=25,
        inlier_ratio=80.0, spatial_entropy=1.4, quad_counts=[5, 5, 5, 5],
        rmse=0.42, rmse_in_sample=0.31, rmse_heldout=0.42,
        mae_in_sample=0.28, mae_heldout=0.36,
        heldout_count=4, rmse_gate_basis="held-out",
    )
    assert "In-sample RMSE (fit residuals): 0.3100 px" in telemetry
    assert "Held-out RMSE (4 check points): 0.4200 px" in telemetry
    assert "In-sample MAE (fit residuals): 0.2800 px" in telemetry
    assert "Held-out MAE (4 check points): 0.3600 px" in telemetry
    assert "Gate RMSE: 0.4200 px (held-out)" in telemetry
    assert "Measured: 0.4200 px" in telemetry

    image = np.zeros((32, 32), dtype=np.uint8)
    fig = create_combined_visualization(
        image, image, image, [], rmse_px=0.42,
        rmse_in_sample_px=0.31, rmse_heldout_px=0.42,
        mae_in_sample_px=0.28, mae_heldout_px=0.36,
        heldout_count=4, rmse_gate_basis="held-out",
    )
    try:
        title = fig._suptitle.get_text()
        assert "In-sample RMSE: 0.3100 px" in title
        assert "Held-out RMSE (4 checks): 0.4200 px" in title
        assert "In-sample MAE: 0.2800 px" in title
        assert "Held-out MAE: 0.3600 px" in title
        assert "Gate RMSE: 0.4200 px (held-out)" in title
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)

    fallback_telemetry = format_telemetry_report(
        status="SUCCESS_SUBPIXEL", inliers=4, total_pts=4,
        inlier_ratio=100.0, spatial_entropy=1.2, quad_counts=[1, 1, 1, 1],
        rmse=0.31, rmse_in_sample=0.31, rmse_heldout=None,
        heldout_count=0,
        rmse_gate_basis="in-sample fallback; held-out unavailable",
        mae_in_sample=0.31, mae_heldout=None,
    )
    assert "Held-out RMSE: UNAVAILABLE" in fallback_telemetry
    assert "Held-out MAE: UNAVAILABLE" in fallback_telemetry
    assert "Gate RMSE: 0.3100 px (in-sample fallback; held-out unavailable)" in fallback_telemetry
