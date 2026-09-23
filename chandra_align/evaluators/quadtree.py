import math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class QuadtreeMetrics:
    uniformity_score: float  # Score in [0.0, 1.0]
    occupied_leaf_cells: int
    total_leaf_cells: int
    occupancy_ratio: float
    shannon_entropy: float
    depth: int


def evaluate_quadtree_uniformity(
    keypoints: np.ndarray,
    image_shape: Tuple[int, int],
    depth: int = 4
) -> QuadtreeMetrics:
    """
    Evaluates spatial keypoint distribution uniformity using recursive quadtree partitioning.

    Parameters:
    -----------
    keypoints : np.ndarray
        Array of shape (N, 2) containing (x, y) pixel coordinates.
    image_shape : Tuple[int, int]
        (height, width) of the image bounding region.
    depth : int
        Quadtree depth level. Default = 4 (yielding 4^4 = 256 leaf cells).

    Returns:
    --------
    QuadtreeMetrics
        Calculated spatial distribution metrics.
    """
    height, width = image_shape
    total_leaf_cells = 4 ** depth

    if keypoints is None or len(keypoints) == 0 or height <= 0 or width <= 0:
        return QuadtreeMetrics(
            uniformity_score=0.0,
            occupied_leaf_cells=0,
            total_leaf_cells=total_leaf_cells,
            occupancy_ratio=0.0,
            shannon_entropy=0.0,
            depth=depth,
        )

    pts = np.asarray(keypoints, dtype=np.float32)
    grid_side = 2 ** depth

    # Normalize coordinates to cell indices [0, grid_side - 1]
    cell_x = np.clip((pts[:, 0] / width * grid_side).astype(int), 0, grid_side - 1)
    cell_y = np.clip((pts[:, 1] / height * grid_side).astype(int), 0, grid_side - 1)

    # Encode 2D cell coordinate into 1D leaf cell index
    cell_indices = cell_y * grid_side + cell_x

    # Bin keypoints into leaf cells
    counts = np.bincount(cell_indices, minlength=total_leaf_cells)
    occupied_cells = int(np.count_nonzero(counts))
    occupancy_ratio = occupied_cells / total_leaf_cells

    # Calculate normalized Shannon entropy over occupied cells
    non_zero_counts = counts[counts > 0]
    total_pts = len(pts)
    probabilities = non_zero_counts / total_pts

    # Entropy H = - sum(p * log2(p))
    entropy = -np.sum(probabilities * np.log2(probabilities))

    # Max possible entropy for the number of occupied cells
    max_possible_entropy = math.log2(occupied_cells) if occupied_cells > 1 else 1.0
    normalized_entropy = (entropy / max_possible_entropy) if max_possible_entropy > 0 else 1.0

    # Combined Uniformity Score: Occupancy Ratio weighted by entropy distribution
    uniformity_score = float(np.clip(occupancy_ratio * normalized_entropy, 0.0, 1.0))

    return QuadtreeMetrics(
        uniformity_score=round(uniformity_score, 4),
        occupied_leaf_cells=occupied_cells,
        total_leaf_cells=total_leaf_cells,
        occupancy_ratio=round(occupancy_ratio, 4),
        shannon_entropy=round(entropy, 4),
        depth=depth,
    )