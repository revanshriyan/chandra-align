"""
Footprint Guard - Spatial Overlap IoU Check

Prevents compute-heavy feature matching on images with insufficient spatial overlap.
"""

import numpy as np
from typing import Tuple, Optional
import warnings


def compute_bbox_iou(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float],
    image_shape1: Optional[Tuple[int, int]] = None,
    image_shape2: Optional[Tuple[int, int]] = None
) -> float:
    """
    Compute Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        bbox1: (x_min, y_min, x_max, y_max) for first image
        bbox2: (x_min, y_min, x_max, y_max) for second image
        image_shape1: (height, width) of first image (for normalization)
        image_shape2: (height, width) of second image (for normalization)
        
    Returns:
        IoU score [0, 1]
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # Check if there's no intersection
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return min(1.0, max(0.0, iou))


def estimate_footprint_from_image(
    image: np.ndarray,
    threshold: float = 10.0
) -> Tuple[float, float, float, float]:
    """
    Estimate valid image footprint from pixel values.
    
    Args:
        image: 2D or 3D numpy array
        threshold: Pixel value threshold to consider as valid data
        
    Returns:
        Bounding box (x_min, y_min, x_max, y_max) of valid pixels
    """
    if image.ndim == 3:
        # Convert to grayscale if RGB
        gray = np.mean(image, axis=2)
    else:
        gray = image
    
    # Find valid pixels (above threshold and not NaN)
    valid = np.isfinite(gray) & (gray > threshold)
    
    if not np.any(valid):
        h, w = gray.shape
        return 0.0, 0.0, float(w), float(h)
    
    rows, cols = np.where(valid)
    y_min, y_max = float(rows.min()), float(rows.max())
    x_min, x_max = float(cols.min()), float(cols.max())
    
    return x_min, y_min, x_max, y_max


def compute_spatial_overlap(
    image1: np.ndarray,
    image2: np.ndarray,
    threshold: float = 10.0
) -> dict:
    """
    Compute spatial overlap between two images using bounding box IoU.
    
    Args:
        image1: First image
        image2: Second image
        threshold: Pixel threshold for valid data
        
    Returns:
        Dict with IoU, bbox1, bbox2, and spatial_overlap_pct
    """
    bbox1 = estimate_footprint_from_image(image1, threshold)
    bbox2 = estimate_footprint_from_image(image2, threshold)
    
    iou = compute_bbox_iou(bbox1, bbox2)
    
    return {
        'iou': iou,
        'spatial_overlap_pct': iou * 100.0,
        'bbox1': bbox1,
        'bbox2': bbox2,
        'threshold': threshold
    }


def footprint_guard(
    image1: np.ndarray,
    image2: np.ndarray,
    min_overlap_pct: float = 15.0,
    threshold: float = 10.0
) -> dict:
    """
    Footprint guard - checks if two images have sufficient spatial overlap.
    
    Args:
        image1: First image (reference)
        image2: Second image (moving)
        min_overlap_pct: Minimum spatial overlap percentage (default 15%)
        threshold: Pixel threshold for valid data
        
    Returns:
        Dict with:
            - pass: bool - whether overlap is sufficient
            - spatial_overlap_pct: percentage overlap
            - iou: Intersection over Union
            - status: 'PASS' or 'FAIL'
            - reason: description of failure reason if any
    """
    overlap_result = compute_spatial_overlap(image1, image2, threshold)
    iou = overlap_result['iou']
    overlap_pct = overlap_result['spatial_overlap_pct']
    
    pass_guard = overlap_pct >= min_overlap_pct
    
    result = {
        'pass': pass_guard,
        'iou': iou,
        'spatial_overlap_pct': overlap_pct,
        'min_overlap_pct': min_overlap_pct,
        'status': 'PASS' if pass_guard else 'FAIL',
        'reason': None
    }
    
    if not pass_guard:
        result['reason'] = (
            f"INSUFFICIENT_FOOTPRINT_OVERLAP: "
            f"Spatial overlap {overlap_pct:.1f}% < minimum {min_overlap_pct}%"
        )
        warnings.warn(result['reason'], UserWarning)
    
    return result


def compute_quadtree_entropy(
    keypoints: np.ndarray,
    image_shape: Tuple[int, int],
    grid_size: int = 4
) -> dict:
    """
    Compute Quadtree spatial entropy score for keypoint distribution.
    
    Args:
        keypoints: Nx2 array of (x, y) keypoint coordinates
        image_shape: (height, width) of image
        grid_size: Grid dimension for quadtree (default 4x4)
        
    Returns:
        Dict with entropy score, cell counts, and distribution quality
    """
    h, w = image_shape
    cell_h, cell_w = h / grid_size, w / grid_size
    
    cell_counts = np.zeros((grid_size, grid_size))
    
    for x, y in keypoints:
        col = int(min(x / cell_w, grid_size - 1))
        row = int(min(y / cell_h, grid_size - 1))
        cell_counts[row, col] += 1
    
    total = np.sum(cell_counts)
    if total == 0:
        return {
            'entropy': 0.0,
            'cell_counts': cell_counts.tolist(),
            'occupied_cells': 0,
            'total_keypoints': 0,
            'uniformity_score': 0.0
        }
    
    # Normalized entropy
    probs = cell_counts / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(grid_size * grid_size)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    occupied = np.sum(cell_counts > 0)
    
    return {
        'entropy': float(entropy),
        'normalized_entropy': float(normalized_entropy),
        'cell_counts': cell_counts.tolist(),
        'occupied_cells': int(occupied),
        'total_cells': grid_size * grid_size,
        'total_keypoints': int(total),
        'uniformity_score': float(normalized_entropy)
    }