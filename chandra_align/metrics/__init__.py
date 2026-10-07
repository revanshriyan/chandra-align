"""
Metrics — RMSE (x, y, total; px and m), inlier stats, uniformity, residual map.

RMSE on held-out check points is the primary metric (ISRO-named). Inlier ratio and
uniformity are reported alongside. RCM / success-rate are NEVER reported without
manual labels (no manual set exists — so they are always UNMEASURED here).
"""

import numpy as np
from .quadrant import (
    build_judge_metrics_summary,
    compute_quadrant_metrics,
    compute_spatial_uniformity_metrics,
    format_quadrant_html,
    validate_registration_gate as _validate_registration_gate,
)


def validate_registration_gate(rmse, inliers, min_inliers, spatial_entropy, quad_counts,
                               model=None):
    """Public package entry point for the three-tier photogrammetric gate.

    When ``model`` (the fitted 2x3/3x3 transform) is supplied, Gate 3
    transform-conditioning runs as a final backstop: a transform that fails
    conditioning is REJECTED even if the residual tiers would accept it.
    """
    status_message, status_code = _validate_registration_gate(
        rmse, inliers, min_inliers, spatial_entropy, quad_counts
    )
    if model is not None and status_code != "DEGENERATE_FAILURE":
        from .conditioning import check_transform_conditioning
        try:
            rmse_v = float(rmse)
        except (TypeError, ValueError):
            rmse_v = float("inf")
        ok, report = check_transform_conditioning(model, rmse_px=rmse_v)
        if not ok:
            failed = [k for k, v in report.get("checks", {}).items() if not v]
            status_message = (
                f"REJECTED (Gate 3 transform conditioning: {', '.join(failed)})"
            )
            status_code = "DEGENERATE_FAILURE"
    return status_message, status_code


def apply_transform(M, pts):
    """Apply a 2x3 affine to (N, 2) points."""
    M = np.asarray(M, np.float64)
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    out = pts @ M[:, :2].T + M[:, 2]
    return out


def rmse_heldout(M, pts_hold_a, pts_hold_b, gsd_m=1.0):
    """RMSE of fitted transform M on HELD-OUT check points (circularity guard).

    Returns dict: rmse_x_px, rmse_y_px, rmse_px, rmse_m, held_out=True, n.
    Raises ValueError if fewer than 1 held-out pair is given — RMSE on the fit set is
    never a substitute (the caller must not pass fit points here).
    """
    pa = np.asarray(pts_hold_a, np.float64).reshape(-1, 2)
    pb = np.asarray(pts_hold_b, np.float64).reshape(-1, 2)
    if len(pa) == 0:
        raise ValueError(
            "no held-out check points: RMSE is UNMEASURED — pass held-out points, "
            "never the fit set (circularity rule)")
    pred = apply_transform(M, pa)
    res = pred - pb
    rx = float(np.sqrt((res[:, 0] ** 2).mean()))
    ry = float(np.sqrt((res[:, 1] ** 2).mean()))
    magnitude = np.linalg.norm(res, axis=1)
    rt = float(np.sqrt((res ** 2).sum(axis=1).mean()))
    return {
        "held_out": True,
        "n_check_points": int(len(pa)),
        "rmse_x_px": rx,
        "rmse_y_px": ry,
        "rmse_px": rt,
        "rmse_m": rt * float(gsd_m),
        "mae_px": float(magnitude.mean()),
        "mae_m": float(magnitude.mean()) * float(gsd_m),
    }


def inlier_stats(n_raw_matches: int, n_inliers: int, inlier_ratio: float):
    return {
        "raw_matches": int(n_raw_matches),
        "inliers": int(n_inliers),
        "inlier_ratio": float(inlier_ratio),
    }


def residual_map(pts_a, pts_b, M, grid=(4, 4), shape=None):
    """Reprojection-residual magnitudes bucketed on a grid (per §7 metrics)."""
    pa = np.asarray(pts_a, np.float64).reshape(-1, 2)
    pb = np.asarray(pts_b, np.float64).reshape(-1, 2)
    res = apply_transform(M, pa) - pb
    mag = np.hypot(res[:, 0], res[:, 1])
    out = {"residuals_px": mag.tolist(), "mean_px": float(mag.mean()) if len(mag) else 0.0,
           "max_px": float(mag.max()) if len(mag) else 0.0}
    if shape is not None:
        from ..refine import grid_bucket
        buckets = grid_bucket(pa, shape, grid)
        cell_mean = {f"{r},{c}": float(mag[idxs].mean()) for (r, c), idxs in buckets.items()}
        out["per_cell_mean_px"] = cell_mean
    return out


def metrics_bundle(rmse_dict, inliers_dict, uniformity: float, runtime_s,
                   trust_flag: str, matcher_name: str, approximation_flag: bool,
                   extra: dict = None):
    """Assemble the §7 metrics bundle. Anything unmeasured must arrive as None and is
    serialised as "UNMEASURED" — never a fabricated number."""
    bundle = {
        "rmse": rmse_dict if rmse_dict is not None else "UNMEASURED",
        "inliers": inliers_dict,
        "uniformity_score": float(uniformity),
        "runtime_s": float(runtime_s) if runtime_s is not None else "UNMEASURED",
        "trust_flag": trust_flag,
        "matcher": matcher_name,
        "approximation_flag": bool(approximation_flag),
        "success_rate": "UNMEASURED (no manual held-out labels)",
        "correct_match_ratio": "UNMEASURED (no manual held-out labels)",
    }
    if extra:
        bundle.update(extra)
    return bundle


# Import enhanced metrics from the separate module
from chandra_align.metrics.enhanced import (
    metrics_bundle_with_ground,
    compute_ground_metrics,
    compute_deformation_field,
    grid_deformation_analysis,
    get_sensor_pixel_scale,
    ResidualVector,
    GroundMetrics,
    SENSOR_PIXEL_SCALES,
)

__all__ = [
    "apply_transform",
    "rmse_heldout",
    "inlier_stats",
    "residual_map",
    "compute_quadrant_metrics",
    "format_quadrant_html",
    "build_judge_metrics_summary",
    "validate_registration_gate",
    "metrics_bundle",
    "metrics_bundle_with_ground",
    "compute_ground_metrics",
    "compute_deformation_field",
    "grid_deformation_analysis",
    "get_sensor_pixel_scale",
    "ResidualVector",
    "GroundMetrics",
    "SENSOR_PIXEL_SCALES",
]
