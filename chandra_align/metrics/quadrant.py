"""Spatial quadrant metrics and judge-facing registration summary schema."""

import html

import numpy as np


def _finite_float(value):
    """Return a strict-JSON-safe Python float."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def compute_quadrant_metrics(inliers_src, residuals, img_shape):
    """Return per-quadrant inlier counts/RMSE and four-quadrant entropy.

    Coordinates are (x, y) in the raster frame. Residuals may be scalar
    magnitudes or (dx, dy) vectors. Invalid/out-of-frame coordinates are
    excluded from the spatial counts.
    """
    points = np.asarray(inliers_src if inliers_src is not None else [], dtype=np.float64).reshape(-1, 2)
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
    valid = np.isfinite(points).all(axis=1) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    right, bottom = x >= width / 2.0, y >= height / 2.0
    masks = {
        "Q1": valid & ~right & ~bottom,
        "Q2": valid & right & ~bottom,
        "Q3": valid & ~right & bottom,
        "Q4": valid & right & bottom,
    }
    names = {
        "Q1": "Top-Left", "Q2": "Top-Right",
        "Q3": "Bottom-Left", "Q4": "Bottom-Right",
    }

    residual_finite = np.isfinite(residual_values).all(axis=1)
    if residual_values.shape[1] == 2:
        residual_sq = np.sum(np.square(np.nan_to_num(residual_values)), axis=1)
    else:
        residual_sq = np.square(np.nan_to_num(residual_values[:, 0]))

    quadrants = {}
    for key, mask in masks.items():
        count = int(mask.sum())
        valid_rmse = mask & residual_finite
        rmse = float(np.sqrt(np.mean(residual_sq[valid_rmse]))) if valid_rmse.any() else 0.0
        quadrants[key] = {"name": names[key], "inlier_count": count, "rmse_px": _finite_float(rmse)}

    total = sum(item["inlier_count"] for item in quadrants.values())
    if total:
        probabilities = [item["inlier_count"] / total for item in quadrants.values()]
        entropy = _finite_float(-sum(p * np.log2(p) for p in probabilities if p > 0.0))
        if entropy == 0.0:
            entropy = 0.0  # Canonicalize IEEE negative zero for JSON/UI output.
    else:
        entropy = 0.0
    return quadrants, entropy


def format_quadrant_html(quadrant_dict, spatial_entropy):
    """Render the dark-slate quadrant metrics card and entropy target badge."""
    entropy = _finite_float(spatial_entropy)
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
        rmse = _finite_float(item.get("rmse_px", 0.0))
        rows.append(f"<tr><td>{name}</td><td>{count}</td><td>{rmse:.4f} px</td></tr>")
    return (
        '<div class="metric-card quadrant-metric-card" '
        'style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:14px;margin-top:8px;color:#f8fafc">'
        '<strong>Quadrant-Wise Spatial Metrics</strong><div style="margin:10px 0">'
        f'<span class="{badge_class}">Spatial Entropy {entropy:.3f} / 2.000 · {badge_text} (target &gt; 0.850)</span>'
        '</div><table style="width:100%;border-collapse:collapse;color:#e2e8f0">'
        '<thead><tr><th align="left">Quadrant</th><th align="right">Inliers</th>'
        '<th align="right">Sub-pixel RMSE</th></tr></thead><tbody>'
        + "".join(rows) + '</tbody></table></div>'
    )


def build_judge_metrics_summary(
    rmse_pixels,
    inlier_count,
    total_correspondences,
    spatial_entropy_score,
    quadrant_metrics,
):
    """Build the strict, fixed-key JSON summary used by the evaluation UI."""
    try:
        inliers = max(0, int(inlier_count))
    except (TypeError, ValueError, OverflowError):
        inliers = 0
    try:
        total = max(0, int(total_correspondences))
    except (TypeError, ValueError, OverflowError):
        total = 0
    ratio_pct = (100.0 * inliers / total) if total > 0 else 0.0
    quadrants = quadrant_metrics if isinstance(quadrant_metrics, dict) else {}

    return {
        "registration_status": "SUCCESS" if inliers >= 8 else "FAILED",
        "global_metrics": {
            "rmse_pixels": _finite_float(rmse_pixels),
            "inlier_count": inliers,
            "inlier_ratio_pct": _finite_float(ratio_pct),
            "spatial_entropy_score": _finite_float(spatial_entropy_score),
        },
        "quadrant_breakdown": {
            "Q1_top_left_rmse": _finite_float(quadrants.get("Q1", {}).get("rmse_px", 0.0)),
            "Q2_top_right_rmse": _finite_float(quadrants.get("Q2", {}).get("rmse_px", 0.0)),
            "Q3_bottom_left_rmse": _finite_float(quadrants.get("Q3", {}).get("rmse_px", 0.0)),
            "Q4_bottom_right_rmse": _finite_float(quadrants.get("Q4", {}).get("rmse_px", 0.0)),
        },
        "transformation_type": "4-DOF Partial Affine + Phase Congruency",
    }


__all__ = [
    "compute_quadrant_metrics",
    "format_quadrant_html",
    "build_judge_metrics_summary",
]
