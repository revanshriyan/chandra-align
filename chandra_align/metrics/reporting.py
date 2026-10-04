"""Post-gate confidence and execution-path details for scientific reports."""

from __future__ import annotations

import math


def build_confidence_assessment(
    *, status_code, status_message, rmse_px, inlier_count, correspondence_count,
    entropy, quadrant_counts, min_inliers, execution_diagnostics,
):
    """Build report-only evidence confidence, decision reason, and fallback path.

    The score is a 0-100 heuristic evidence index, not a probability and never
    participates in registration gate decisions.
    """
    try:
        inliers = max(0, int(inlier_count))
    except (TypeError, ValueError, OverflowError):
        inliers = 0
    try:
        correspondences = max(0, int(correspondence_count))
    except (TypeError, ValueError, OverflowError):
        correspondences = 0
    try:
        entropy_value = float(entropy)
        if not math.isfinite(entropy_value):
            entropy_value = 0.0
    except (TypeError, ValueError, OverflowError):
        entropy_value = 0.0
    try:
        rmse = float(rmse_px)
        if not math.isfinite(rmse) or rmse < 0:
            rmse = None
    except (TypeError, ValueError, OverflowError):
        rmse = None

    if isinstance(quadrant_counts, dict):
        counts = [quadrant_counts.get(f"Q{i}", 0) for i in range(1, 5)]
    else:
        counts = list(quadrant_counts or ())[:4]
    counts.extend([0] * (4 - len(counts)))
    active_quadrants = sum(
        1 for value in counts
        if (isinstance(value, dict) and _safe_int(value.get("inlier_count", 0)) > 0)
        or (not isinstance(value, dict) and _safe_int(value) > 0)
    )

    support = min(inliers / max(int(min_inliers or 1), 1), 1.0)
    inlier_ratio = inliers / correspondences if correspondences else 0.0
    spread = min(max(entropy_value / 2.0, 0.0), 1.0)
    residual_quality = max(0.0, 1.0 - rmse / 2.5) if rmse is not None else 0.0
    score = round(100.0 * (
        0.30 * support + 0.25 * inlier_ratio + 0.20 * spread
        + 0.25 * residual_quality
    ), 2)

    code = str(status_code or "UNKNOWN").upper()
    if code == "SUCCESS_SUBPIXEL":
        decision_reason = (
            f"Passed sub-pixel gate: gate RMSE {rmse:.4f} px <= 0.50 px; "
            f"{inliers} inliers >= {int(min_inliers)}; entropy {entropy_value:.4f} >= 0.75; "
            f"{active_quadrants}/4 active quadrants >= 3."
            if rmse is not None else
            f"Passed sub-pixel gate with {inliers} inliers and {active_quadrants}/4 active quadrants."
        )
    elif code == "COARSE_ADVISORY":
        decision_reason = (
            f"Coarse advisory: gate returned a regional-fit advisory at RMSE {rmse:.4f} px; "
            f"{inliers} inliers; entropy {entropy_value:.4f}; "
            f"{active_quadrants}/4 active quadrants."
            if rmse is not None else
            "Coarse advisory: gate criteria permit regional alignment, but RMSE is unavailable."
        )
    else:
        decision_reason = str(status_message or "Registration gate rejected the candidate fit.")

    diagnostics = execution_diagnostics if isinstance(execution_diagnostics, dict) else {}
    primary = diagnostics.get("primary_engine") or "Unknown"
    registration = diagnostics.get("registration_engine") or diagnostics.get("engine") or primary
    steps = [str(primary)]
    if str(registration) != str(primary):
        steps.append(str(registration))
    fallback = bool(diagnostics.get("fallback_triggered", False))
    fallback_path = {
        "steps": steps,
        "primary_engine": str(primary),
        "registration_engine": str(registration),
        "fallback_triggered": fallback,
        "fallback_reason": diagnostics.get("fallback_reason"),
        "fallback_error": diagnostics.get("fallback_error"),
    }
    return {
        "confidence_score": score,
        "confidence_basis": (
            "Heuristic evidence index (0-100; not a calibrated probability): "
            "30% inlier support, 25% inlier ratio, 20% normalized entropy, "
            "25% gate-RMSE quality relative to 2.5 px."
        ),
        "decision_reason": decision_reason,
        "fallback_path": fallback_path,
    }


def _safe_int(value):
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


__all__ = ["build_confidence_assessment"]
