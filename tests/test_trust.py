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
