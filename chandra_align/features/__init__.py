"""Feature detection utilities used by the registration pipeline."""

from .distribution import (
    bucket_ids,
    select_distributed_matches,
    spatial_distribution_metrics,
)

__all__ = ["bucket_ids", "select_distributed_matches", "spatial_distribution_metrics"]
