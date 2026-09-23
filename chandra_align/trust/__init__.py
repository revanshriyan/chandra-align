"""Trust & circularity guard (M6).

- Circularity guard: fit transform on a subset, report RMSE on held-out check points.
  Structurally impossible to bypass: rmse_heldout() refuses empty held-out sets, and
  evaluate() splits the verified inliers itself unless an explicit held-out set is
  passed.
- Trust flag: thresholds from measured inlier ratio + uniformity + held-out RMSE.
  calibration_status stays "UNCALIBRATED" until recalibrated on real data.
"""

import numpy as np

from ..metrics import rmse_heldout


def split_fit_holdout(pts_a, pts_b, holdout_fraction=0.2, seed=0):
    """Deterministic split of verified inliers into fit-set and held-out check points.

    Evenly spaced deterministic selection (seed accepted for interface stability but
    the split itself is stride-based, so runs are reproducible).
    Returns (fit_a, fit_b, hold_a, hold_b).
    """
    pa = np.asarray(pts_a, np.float64).reshape(-1, 2)
    pb = np.asarray(pts_b, np.float64).reshape(-1, 2)
    n = len(pa)
    if n < 5:
        # too few points to hold out: everything goes to fit, held-out RMSE will be
        # UNMEASURED (rmse_heldout refuses an empty set) — the guard still fires
        return pa, pb, np.zeros((0, 2)), np.zeros((0, 2))
    n_hold = max(1, int(round(n * holdout_fraction)))
    stride_idx = np.linspace(0, n - 1, n_hold).astype(int)
    hold_mask = np.zeros(n, dtype=bool)
    hold_mask[stride_idx] = True
    return pa[~hold_mask], pb[~hold_mask], pa[hold_mask], pb[hold_mask]


def evaluate(M, pts_a, pts_b, holdout_fraction=0.2):
    """Circularity-guarded RMSE: refit on the fit subset, report on held-out points.

    Returns (M_refit, rmse_dict). rmse_dict["rmse_px"] is UNMEASURED ("UNMEASURED")
    when too few points exist to hold any out — structurally impossible to get a
    fit-set RMSE out of this function.
    """
    fit_a, fit_b, hold_a, hold_b = split_fit_holdout(pts_a, pts_b, holdout_fraction)
    M_refit = M
    if len(fit_a) >= 3:
        import cv2
        M_refit, _ = cv2.estimateAffinePartial2D(
            fit_a.astype(np.float32), fit_b.astype(np.float32),
            method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M_refit is None:
            M_refit = M
    if len(hold_a) == 0:
        return M_refit, {"held_out": False, "n_check_points": 0,
                         "rmse_x_px": "UNMEASURED", "rmse_y_px": "UNMEASURED",
                         "rmse_px": "UNMEASURED", "rmse_m": "UNMEASURED"}
    return M_refit, rmse_heldout(M_refit, hold_a, hold_b)


def trust_flag(rmse_px, inlier_ratio: float, uniformity: float, cfg_trust: dict):
    """Calibrated trust flag from the config thresholds.

    "Trusted" requires ALL of: held-out RMSE at or under the threshold, inlier ratio
    at or over the floor, uniformity at or over the floor. Any failure -> explicit
    "Not Trusted". An UNMEASURED RMSE is never Trusted.
    """
    rmse_thresh = float(cfg_trust.get("rmse_px_threshold", 0.5))
    ratio_floor = float(cfg_trust.get("inlier_ratio_threshold", 0.15))
    uni_floor = float(cfg_trust.get("uniformity_threshold", 0.125))
    if rmse_px == "UNMEASURED" or rmse_px is None:
        return "Not Trusted"
    if float(rmse_px) <= rmse_thresh and inlier_ratio >= ratio_floor and uniformity >= uni_floor:
        return "Trusted"
    return "Not Trusted"


def calibration_status(cfg_trust: dict) -> str:
    return str(cfg_trust.get("calibration_status", "UNCALIBRATED"))
