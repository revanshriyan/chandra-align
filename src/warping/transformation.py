"""
Transformation Fallback Ladder - Piecewise-Affine -> Global Affine -> Similarity

Provides robust spatial warping with automatic fallback on failure.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import warnings
from src.guards.matrix_guard import validate_transformation_matrix


class TransformMode(Enum):
    PIECEWISE_AFFINE = "piecewise_affine"
    GLOBAL_AFFINE = "global_affine"
    SIMILARITY = "similarity"
    IDENTITY = "identity"


class TransformationError(Exception):
    """Custom exception for transformation failures."""
    pass


def compute_global_affine(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    method: int = cv2.LMEDS
) -> Optional[np.ndarray]:
    """
    Compute global affine transformation from matched point pairs.
    
    Args:
        src_points: Nx2 source points
        dst_points: Nx2 destination points
        method: Robust estimation method (LMEDS, RANSAC, etc.)
        
    Returns:
        2x3 affine matrix or None if failed
    """
    if len(src_points) < 3 or len(dst_points) < 3:
        return None
    
    try:
        M, inliers = cv2.estimateAffinePartial2D(
            src_points.astype(np.float32),
            dst_points.astype(np.float32),
            method=method,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99
        )
        return M
    except cv2.error:
        return None


def compute_similarity_transform(
    src_points: np.ndarray,
    dst_points: np.ndarray
) -> Optional[np.ndarray]:
    """
    Compute similarity transform (rotation + translation + uniform scale).
    
    Args:
        src_points: Nx2 source points
        dst_points: Nx2 destination points
        
    Returns:
        2x3 similarity matrix or None if failed
    """
    if len(src_points) < 2 or len(dst_points) < 2:
        return None
    
    # Center the points
    src_centroid = np.mean(src_points, axis=0)
    dst_centroid = np.mean(dst_points, axis=0)
    
    src_centered = src_points - src_centroid
    dst_centered = dst_points - dst_centroid
    
    # Solve for rotation + scale using SVD
    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute scale
    src_var = np.sum(src_centered ** 2)
    dst_var = np.sum(dst_centered ** 2)
    scale = np.sqrt(dst_var / src_var) if src_var > 0 else 1.0
    
    # Construct similarity matrix
    M = np.eye(3)
    M[:2, :2] = scale * R
    M[:2, 2] = dst_centroid - scale * (R @ src_centroid)
    
    return M[:2, :]  # Return 2x3 affine


def create_piecewise_affine_mesh(
    image_shape: Tuple[int, int],
    grid_size: Tuple[int, int] = (8, 8),
    overlap: float = 0.1
) -> List[Dict]:
    """
    Create a piecewise affine mesh grid for warping.
    
    Args:
        image_shape: (height, width)
        grid_size: (rows, cols) for grid
        overlap: Overlap fraction between tiles
        
    Returns:
        List of tile dictionaries with corners and bounds
    """
    h, w = image_shape
    rows, cols = grid_size
    
    tile_h = int(h / rows * (1 + overlap))
    tile_w = int(w / cols * (1 + overlap))
    
    stride_h = int(h / rows)
    stride_w = int(w / cols)
    
    tiles = []
    for r in range(rows):
        for c in range(cols):
            y = r * stride_h
            x = c * stride_w
            
            # Source corners in destination image
            src_corners = np.array([
                [x, y],
                [min(x + tile_w, w), y],
                [min(x + tile_w, w), min(y + tile_h, h)],
                [x, min(y + tile_h, h)]
            ], dtype=np.float32)
            
            # Destination corners (same as source for identity)
            dst_corners = src_corners.copy()
            
            tiles.append({
                'bounds': (x, y, min(x + tile_w, w), min(y + tile_h, h)),
                'src_corners': src_corners,
                'dst_corners': dst_corners,
                'row': r,
                'col': c
            })
    
    return tiles


def warp_piecewise_affine(
    image: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    grid_size: Tuple[int, int] = (8, 8),
    overlap: float = 0.1,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REFLECT_101
) -> Tuple[np.ndarray, Dict]:
    """
    Warp image using piecewise affine transformation with quadtree mesh.
    
    Args:
        image: Source image
        src_points: Source control points
        dst_points: Destination control points
        grid_size: Grid resolution for piecewise affine
        overlap: Tile overlap fraction
        interpolation: OpenCV interpolation method
        border_mode: OpenCV border mode
        
    Returns:
        (warped_image, mesh_info)
    """
    h, w = image.shape[:2]
    
    # Create mesh
    tiles = create_piecewise_affine_mesh(image.shape[:2], grid_size)
    
    # Fit affine for each tile using nearby points
    tile_transforms = []
    for tile in tiles:
        # Find points within tile bounds (with some margin)
        bounds = tile['bounds']
        margin = 50
        mask = (
            (src_points[:, 0] >= bounds[0] - margin) &
            (src_points[:, 0] <= bounds[2] + margin) &
            (src_points[:, 1] >= bounds[1] - margin) &
            (src_points[:, 1] <= bounds[3] + margin)
        )
        
        if np.sum(mask) >= 3:
            M = compute_global_affine(
                src_points[mask], dst_points[mask]
            )
            if M is not None:
                # Validate matrix
                from src.guards.matrix_guard import validate_transformation_matrix
                if validate_transformation_matrix(M)['valid']:
                    tile_transforms.append({
                        'bounds': bounds,
                        'matrix': M
                    })
                    continue
        
        # No valid points or invalid matrix - mark for fallback
        tile_transforms.append({
            'bounds': bounds,
            'matrix': None
        })
    
    # Apply piecewise warping
    warped = np.zeros_like(image)
    weight_map = np.zeros(image.shape[:2], dtype=np.float32)
    
    for i, (tile, trans) in enumerate(zip(tiles, tile_transforms)):
        if trans['matrix'] is None:
            continue
            
        x, y, x2, y2 = trans['bounds']
        M = trans['matrix']
        
        # Extract tile from source
        tile_img = image[y:y2, x:x2]
        
        # Warp tile
        warped_tile = cv2.warpAffine(
            tile_img, M, (x2 - x, y2 - y),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT
        )
        
        # Create feather mask for blending
        mask = np.ones_like(warped_tile[:, :, 0], dtype=np.float32)
        # Feather edges
        feather = 10
        if warped_tile.shape[0] > 2 * feather and warped_tile.shape[1] > 2 * feather:
            mask[:feather, :] *= np.linspace(0, 1, feather)[:, None]
            mask[-feather:, :] *= np.linspace(1, 0, feather)[:, None]
            mask[:, :feather] *= np.linspace(0, 1, feather)[None, :]
            mask[:, -feather:] *= np.linspace(1, 0, feather)[None, :]
        
        # Accumulate
        if warped_tile.ndim == 3:
            for c in range(warped_tile.shape[2]):
                warped[y:y2, x:x2, c] += warped_tile[:, :, c] * mask
        else:
            warped[y:y2, x:x2] += warped_tile * mask
        weight_map[y:y2, x:x2] += mask
    
    # Normalize
    weight_map[weight_map == 0] = 1
    if warped.ndim == 3:
        for c in range(warped.shape[2]):
            warped[:, :, c] /= weight_map
    else:
        warped /= weight_map
    
    return warped.astype(image.dtype), {
        'tiles_used': sum(1 for t in tile_transforms if t['matrix'] is not None),
        'total_tiles': len(tiles),
        'grid_size': grid_size
    }


def transformation_ladder(
    image: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    image_shape: Tuple[int, int],
    min_inliers_piecewise: int = 10,
    min_inliers_global: int = 6,
    min_inliers_similarity: int = 2
) -> Tuple[np.ndarray, Dict]:
    """
    Transformation fallback ladder:
    1. Try Piecewise-Affine (quadtree mesh)
    2. Fallback to Global Affine
    3. Fallback to Similarity Transform
    4. Identity (no transform)
    
    Args:
        image: Source image
        src_points: Source keypoints
        dst_points: Destination keypoints
        image_shape: (height, width) of image
        min_inliers_*: Minimum inliers required for each level
        
    Returns:
        (warped_image, transform_info)
    """
    transform_info = {
        'mode': None,
        'matrix': None,
        'inliers': 0,
        'fallback_reason': None,
        'validation': None
    }
    
    # Check if we have enough points for piecewise
    if len(src_points) >= min_inliers_piecewise:
        try:
            warped, mesh_info = warp_piecewise_affine(
                image, src_points, dst_points
            )
            if mesh_info['tiles_used'] >= 4:  # At least 4 valid tiles
                transform_info['mode'] = TransformMode.PIECEWISE_AFFINE.value
                transform_info['mesh_info'] = mesh_info
                return warped, transform_info
            else:
                warnings.warn(f"Piecewise affine only used {mesh_info['tiles_used']} tiles, falling back")
        except Exception as e:
            warnings.warn(f"Piecewise affine failed: {e}, falling back to global affine")
    
    # Fallback 1: Global Affine
    if len(src_points) >= min_inliers_global:
        M = compute_global_affine(src_points, dst_points)
        if M is not None:
            validation = validate_transformation_matrix(M)
            if validation['valid']:
                try:
                    warped = cv2.warpAffine(
                        image, M, (image.shape[1], image.shape[0]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101
                    )
                    transform_info['mode'] = TransformMode.GLOBAL_AFFINE.value
                    transform_info['matrix'] = M
                    transform_info['fallback_reason'] = 'Piecewise affine failed or insufficient tiles'
                    return warped, transform_info
                except Exception as e:
                    warnings.warn(f"Global affine warp failed: {e}")
            else:
                warnings.warn(f"Global affine validation failed: {validation['issues']}")
    
    # Fallback 2: Similarity Transform
    if len(src_points) >= min_inliers_similarity:
        M = compute_similarity_transform(src_points, dst_points)
        if M is not None:
            validation = validate_transformation_matrix(M)
            if validation['valid']:
                try:
                    warped = cv2.warpAffine(
                        image, M, (image.shape[1], image.shape[0]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101
                    )
                    transform_info['mode'] = TransformMode.SIMILARITY.value
                    transform_info['matrix'] = M
                    transform_info['fallback_reason'] = 'Global affine failed or insufficient inliers'
                    return warped, transform_info
                except Exception as e:
                    warnings.warn(f"Similarity warp failed: {e}")
            else:
                warnings.warn(f"Similarity validation failed: {validation['issues']}")
    
    # Final fallback: Identity (no transform)
    warnings.warn("All transforms failed, returning original image (identity)")
    transform_info['mode'] = TransformMode.IDENTITY.value
    transform_info['matrix'] = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    transform_info['fallback_reason'] = 'All transformations failed - identity fallback'
    return image.copy(), transform_info


def apply_transformation(
    image: np.ndarray,
    matrix: np.ndarray,
    output_shape: Optional[Tuple[int, int]] = None,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REFLECT_101
) -> np.ndarray:
    """
    Apply affine transformation to image.
    
    Args:
        image: Source image
        matrix: 2x3 affine matrix
        output_shape: (width, height) for output
        interpolation: OpenCV interpolation
        border_mode: OpenCV border mode
        
    Returns:
        Warped image
    """
    if output_shape is None:
        output_shape = (image.shape[1], image.shape[0])
    
    return cv2.warpAffine(
        image, matrix, output_shape,
        flags=interpolation,
        borderMode=border_mode
    )