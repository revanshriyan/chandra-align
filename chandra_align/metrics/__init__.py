"""
Metrics — RMSE (x, y, total; px and m), inlier stats, uniformity, residual map.

RMSE on held-out check points is the primary metric (ISRO-named). Inlier ratio and
uniformity are reported alongside. RCM / success-rate are NEVER reported without
manual labels (no manual set exists — so they are always UNMEASURED here).
"""

import html
import numpy as np


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
    rt = float(np.sqrt((res ** 2).sum(axis=1).mean()))
    return {
        "held_out": True,
        "n_check_points": int(len(pa)),
        "rmse_x_px": rx,
        "rmse_y_px": ry,
        "rmse_px": rt,
        "rmse_m": rt * float(gsd_m),
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


def compute_quadrant_metrics(inliers_src, residuals, img_shape):
    """Return quadrant inlier counts/RMSE and four-quadrant spatial entropy.

    ``inliers_src`` must contain (x, y) image coordinates. ``residuals`` may
    contain one scalar residual per point or (dx, dy) residual vectors.
    Points outside the raster and non-finite coordinates are excluded.
    """
    points = np.asarray(inliers_src if inliers_src is not None else [], dtype=np.float64)
    points = points.reshape(-1, 2)
    residual_values = np.asarray(residuals if residuals is not None else [], dtype=np.float64)
    if residual_values.size == 0:
        residual_values = np.zeros((0, 2), dtype=np.float64)
    elif residual_values.ndim == 1:
        residual_values = residual_values.reshape(-1, 1)
    elif residual_values.ndim != 2 or residual_values.shape[1] not in (1, 2):
        raise ValueError("Residuals must be a vector or an (N, 2) array")
    if len(points) != len(residual_values):
        raise ValueError("Inlier coordinate and residual counts must match")
    if len(img_shape) < 2:
        raise ValueError("img_shape must provide height and width")

    height, width = int(img_shape[0]), int(img_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("Image height and width must be positive")

    x, y = points[:, 0], points[:, 1]
    valid_points = (
        np.isfinite(points).all(axis=1)
        & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    )
    right = x >= (width / 2.0)
    bottom = y >= (height / 2.0)
    quadrant_masks = {
        "Q1": valid_points & ~right & ~bottom,
        "Q2": valid_points & right & ~bottom,
        "Q3": valid_points & ~right & bottom,
        "Q4": valid_points & right & bottom,
    }
    names = {
        "Q1": "Top-Left",
        "Q2": "Top-Right",
        "Q3": "Bottom-Left",
        "Q4": "Bottom-Right",
    }
    residual_finite = np.isfinite(residual_values).all(axis=1)
    if residual_values.shape[1] == 2:
        residual_sq = np.sum(np.square(np.nan_to_num(residual_values)), axis=1)
    else:
        residual_sq = np.square(np.nan_to_num(residual_values[:, 0]))

    quadrant_dict = {}
    for key, mask in quadrant_masks.items():
        count = int(mask.sum())
        metric_mask = mask & residual_finite
        rmse = float(np.sqrt(np.mean(residual_sq[metric_mask]))) if metric_mask.any() else 0.0
        quadrant_dict[key] = {
            "name": names[key],
            "inlier_count": count,
            "rmse_px": rmse,
        }

    total = sum(item["inlier_count"] for item in quadrant_dict.values())
    if total:
        probabilities = [item["inlier_count"] / total for item in quadrant_dict.values()]
        entropy = float(-sum(p * np.log2(p) for p in probabilities if p > 0.0))
        if entropy == 0.0:
            entropy = 0.0  # Normalize IEEE negative zero for clean UI formatting.
    else:
        entropy = 0.0
    return quadrant_dict, entropy


def format_quadrant_html(quadrant_dict, spatial_entropy):
    """Render a dark-slate quadrant metric card and target-status badge."""
    try:
        entropy = float(spatial_entropy)
    except (TypeError, ValueError, OverflowError):
        entropy = 0.0
    if not np.isfinite(entropy):
        entropy = 0.0
    target_passed = entropy > 0.850
    badge_class = "pass-badge" if target_passed else "fail-badge"
    badge_text = "TARGET MET" if target_passed else "BELOW TARGET"
    rows = []
    for key, default_name in (
        ("Q1", "Top-Left"), ("Q2", "Top-Right"),
        ("Q3", "Bottom-Left"), ("Q4", "Bottom-Right"),
    ):
        item = quadrant_dict.get(key, {}) if isinstance(quadrant_dict, dict) else {}
        name = html.escape(str(item.get("name", default_name)))
        try:
            count = max(0, int(item.get("inlier_count", 0)))
        except (TypeError, ValueError, OverflowError):
            count = 0
        try:
            rmse = float(item.get("rmse_px", 0.0))
        except (TypeError, ValueError, OverflowError):
            rmse = 0.0
        if not np.isfinite(rmse):
            rmse = 0.0
        rows.append(
            "<tr>"
            f"<td>{name}</td><td>{count}</td><td>{rmse:.4f} px</td>"
            "</tr>"
        )
    return (
        '<div class="metric-card quadrant-metric-card" '
        'style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:14px;margin-top:8px;color:#f8fafc">'
        '<strong>Quadrant-Wise Spatial Metrics</strong>'
        '<div style="margin:10px 0">'
        f'<span class="{badge_class}">Spatial Entropy {entropy:.3f} / 2.000 · {badge_text} (target &gt; 0.850)</span>'
        '</div>'
        '<table style="width:100%;border-collapse:collapse;color:#e2e8f0">'
        '<thead><tr><th align="left">Quadrant</th><th align="right">Inliers</th>'
        '<th align="right">Sub-pixel RMSE</th></tr></thead>'
        '<tbody>' + "".join(rows) + '</tbody></table></div>'
    )


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
