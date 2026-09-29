"""Feature detection utilities used by the registration pipeline."""

from .distribution import (
    bucket_ids,
    select_distributed_matches,
    spatial_distribution_metrics,
    select_quadrant_keypoints,
    select_quadrant_balanced_matches,
)

__all__ = ["bucket_ids", "select_distributed_matches", "spatial_distribution_metrics", "select_quadrant_keypoints", "select_quadrant_balanced_matches"]
