"""Evaluation harnesses (Phase 11+). Re-exports the robustness battery."""

from .robustness import (
    BATTERY_VERSION,
    PERTURBATIONS,
    compose_gt_with_warp,
    independent_accuracy,
    median_inlier_error,
    overlap_coverage,
)

__all__ = [
    "BATTERY_VERSION",
    "PERTURBATIONS",
    "compose_gt_with_warp",
    "independent_accuracy",
    "median_inlier_error",
    "overlap_coverage",
]
