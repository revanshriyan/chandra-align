"""
CHANDRA-ALIGN: Lunar Cross-Sensor Registration Engine
Illumination-Invariant Preprocessing Module

Modular image enhancement routines for lunar imagery:
- CLAHE Enhancement
- Shadow Masking
- Wallis Filter (toggleable)
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    normalize_range: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) 
    to normalize low sun-elevation shadows across lunar orbital passes.
    
    Args:
        img: Input image (float64, float32, or uint8, single channel)
        clip_limit: Threshold for contrast limiting (default: 3.0 per ISRO spec)
        tile_grid_size: Size of grid for histogram equalization (default: 8x8)
        normalize_range: Output normalization range (default: [0, 1])
        
    Returns:
        CLAHE-enhanced image as float64 in normalize_range
    """
    img = np.asarray(img, dtype=np.float64)
    
    # Handle edge cases
    if img.ndim != 2:
        raise ValueError("CLAHE requires single-channel 2D image")
    
    # Normalize to uint8 for CLAHE
    lo, hi = np.nanpercentile(img, [1, 99])
    if hi <= lo:
        lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.float64)
    
    img_u8 = np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_clahe = clahe.apply(img_u8)
    
    # Return as float64 normalized to specified range
    lo_out, hi_out = normalize_range
    return (img_clahe.astype(np.float64) / 255.0) * (hi_out - lo_out) + lo_out


def detect_shadows(
    img: np.ndarray,
    method: str = "otsu",
    adaptive_block: int = 101,
    adaptive_c: float = -10.0
) -> np.ndarray:
    """
    Identify zero-reflectance shadow regions using adaptive/Otsu thresholding.
    Suppresses keypoint detection inside deep unlit craters.
    
    Args:
        img: Input image (float64, single channel, normalized [0, 1] or [0, 255])
        method: "otsu" or "adaptive"
        adaptive_block: Block size for adaptive threshold (must be odd)
        adaptive_c: Constant subtracted from mean for adaptive threshold
        
    Returns:
        Binary shadow mask (1.0 = shadow, 0.0 = illuminated)
    """
    img = np.asarray(img, dtype=np.float64)
    
    # Ensure image is in [0, 255] range
    if img.max() <= 1.0:
        img_u8 = (img * 255).astype(np.uint8)
    else:
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    
    if method == "otsu":
        _, shadow_mask = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "adaptive":
        if adaptive_block % 2 == 0:
            adaptive_block += 1
        shadow_mask = cv2.adaptiveThreshold(
            img_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, adaptive_block, adaptive_c
        )
    else:
        raise ValueError(f"Unknown shadow detection method: {method}")
    
    # Return as float mask (1.0 = shadow, 0.0 = illuminated)
    return (shadow_mask / 255.0).astype(np.float64)


def apply_wallis_filter(
    img: np.ndarray,
    target_mean: float = 128.0,
    target_std: float = 50.0,
    window_size: int = 31
) -> np.ndarray:
    """
    Wallis filter: Standardize local mean and variance across images 
    with stark lighting contrasts.
    
    Args:
        img: Input image (float64, single channel)
        target_mean: Target local mean (default: 128 for 8-bit range)
        target_std: Target local standard deviation (default: 50)
        window_size: Local window size (must be odd, default: 31)
        
    Returns:
        Wallis-filtered image as float64
    """
    img = np.asarray(img, dtype=np.float64)
    
    if window_size % 2 == 0:
        window_size += 1
    
    # Compute local mean using box filter
    kernel = np.ones((window_size, window_size), dtype=np.float64) / (window_size ** 2)
    local_mean = cv2.filter2D(img, -1, kernel)
    local_sqr_mean = cv2.filter2D(img ** 2, -1, kernel)
    local_var = np.maximum(local_sqr_mean - local_mean ** 2, 0.0)
    local_std = np.sqrt(local_var + 1e-8)
    
    # Wallis normalization: (img - local_mean) * (target_std / local_std) + target_mean
    # Handle division by zero
    scale = np.where(local_std > 1e-6, target_std / local_std, 1.0)
    filtered = (img - local_mean) * scale + target_mean
    
    return filtered


def apply_wallis_adaptive(
    img: np.ndarray,
    target_mean: float = 128.0,
    target_std: float = 50.0,
    window_size: int = 31
) -> np.ndarray:
    """
    Wallis filter with adaptive parameter estimation.
    Automatically estimates target statistics from global image.
    """
    img = np.asarray(img, dtype=np.float64)
    
    if window_size % 2 == 0:
        window_size += 1
    
    # Estimate global statistics
    global_mean = float(np.nanmean(img))
    global_std = float(np.nanstd(img))
    
    # Use provided targets or fall back to global
    target_mean = target_mean if target_mean > 0 else global_mean
    target_std = target_std if target_std > 0 else global_std
    
    return apply_wallis_filter(img, target_mean, target_std, window_size)


def preprocess_pipeline(
    img: np.ndarray,
    enable_clahe: bool = True,
    clahe_clip_limit: float = 3.0,
    clahe_tile_grid: Tuple[int, int] = (8, 8),
    enable_shadow_mask: bool = True,
    shadow_method: str = "otsu",
    enable_wallis: bool = False,
    wallis_target_mean: float = 128.0,
    wallis_target_std: float = 50.0,
    wallis_window: int = 31
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Full preprocessing pipeline combining CLAHE, shadow detection, and Wallis filter.
    
    Args:
        img: Input image
        enable_clahe: Apply CLAHE enhancement
        clahe_clip_limit: CLAHE clip limit
        clahe_tile_grid: CLAHE tile grid size
        enable_shadow_mask: Generate shadow mask
        shadow_method: Shadow detection method ("otsu" or "adaptive")
        enable_wallis: Apply Wallis filter
        wallis_target_mean: Wallis target mean
        wallis_target_std: Wallis target std
        wallis_window: Wallis window size
        
    Returns:
        Tuple of (processed_image, shadow_mask_or_none)
    """
    img = np.asarray(img, dtype=np.float64).copy()
    shadow_mask = None
    
    # Step 1: CLAHE
    if enable_clahe:
        img = apply_clahe(img, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid, normalize_range=(0.0, 255.0))
    
    # Step 2: Shadow detection (before Wallis to avoid distorting mask)
    if enable_shadow_mask:
        shadow_mask = detect_shadows(img, method=shadow_method)
    
    # Step 3: Wallis filter (optional)
    if enable_wallis:
        img = apply_wallis_filter(img, target_mean=wallis_target_mean, target_std=wallis_target_std, window_size=wallis_window)
    
    return img, shadow_mask


def suppress_keypoints_in_shadows(
    keypoints: list,
    shadow_mask: np.ndarray,
    threshold: float = 0.5
) -> list:
    """
    Filter out keypoints that fall within shadow regions.
    
    Args:
        keypoints: List of cv2.KeyPoint objects
        shadow_mask: Binary shadow mask (1.0 = shadow, 0.0 = illuminated)
        threshold: Shadow probability threshold (default: 0.5)
        
    Returns:
        Filtered list of keypoints outside shadow regions
    """
    if shadow_mask is None:
        return keypoints
    
    h, w = shadow_mask.shape[:2]
    filtered = []
    
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= x < w and 0 <= y < h:
            if shadow_mask[y, x] < threshold:
                filtered.append(kp)
        else:
            filtered.append(kp)  # Keep keypoints outside image bounds
    
    return filtered


__all__ = [
    "apply_clahe",
    "detect_shadows",
    "apply_wallis_filter",
    "apply_wallis_adaptive",
    "preprocess_pipeline",
    "suppress_keypoints_in_shadows"
]