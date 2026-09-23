import numpy as np
import cv2
from typing import Tuple

def refined_lee_filter(img: np.ndarray, win_size: int = 5, num_looks: float = 1.0) -> np.ndarray:
    """
    Applies the Refined Lee speckle filter to reduce SAR noise while preserving edges.
    
    Args:
        img: Input SAR intensity image (2D float array).
        win_size: Local window size (must be odd integer >= 3).
        num_looks: Equivalent number of looks (ENL) of the SAR system.
    """
    if win_size % 2 == 0 or win_size < 3:
        raise ValueError("win_size must be an odd integer >= 3")

    img_float = img.astype(np.float32)
    # Epsilon padding to prevent zero variance division
    eps = 1e-8

    # Mean and variance in moving window
    kernel = np.ones((win_size, win_size), dtype=np.float32) / (win_size ** 2)
    local_mean = cv2.filter2D(img_float, -1, kernel)
    local_sqr_mean = cv2.filter2D(img_float ** 2, -1, kernel)
    local_var = np.maximum(local_sqr_mean - local_mean ** 2, 0.0)

    # Theoretical noise variance for given ENL
    noise_var = (1.0 / num_looks) * (local_mean ** 2)

    # Weighting factor K
    k_weight = local_var / (local_var + noise_var + eps)
    k_weight = np.clip(k_weight, 0.0, 1.0)

    # Filtered output
    filtered = local_mean + k_weight * (img_float - local_mean)
    return np.clip(filtered, 0.0, None)

def frost_filter(img: np.ndarray, win_size: int = 5, damp_factor: float = 1.0) -> np.ndarray:
    """
    Applies the Frost exponential damping filter for SAR speckle reduction.
    """
    if win_size % 2 == 0 or win_size < 3:
        raise ValueError("win_size must be an odd integer >= 3")

    img_float = img.astype(np.float32)
    pad_size = win_size // 2
    padded = np.pad(img_float, pad_size, mode='reflect')
    out = np.zeros_like(img_float)

    # Precalculate spatial distances in window
    y_grid, x_grid = np.mgrid[-pad_size:pad_size+1, -pad_size:pad_size+1]
    distances = np.sqrt(x_grid ** 2 + y_grid ** 2)

    rows, cols = img_float.shape
    for r in range(rows):
        for c in range(cols):
            window = padded[r:r+win_size, c:c+win_size]
            w_mean = np.mean(window)
            if w_mean <= 1e-6:
                out[r, c] = img_float[r, c]
                continue
            
            w_std = np.std(window)
            c_v = w_std / w_mean  # Coefficient of variation
            
            # Exponential weights based on distance and local variation
            weights = np.exp(-damp_factor * c_v * distances)
            weights /= (np.sum(weights) + 1e-8)
            
            out[r, c] = np.sum(window * weights)

    return out

def prepare_sar_intensity(img: np.ndarray) -> np.ndarray:
    """Normalizes raw SAR image into uint8 displayable format with log compression."""
    clipped = np.clip(img.astype(np.float32), 1e-6, None)
    log_img = np.log10(clipped)
    norm = cv2.normalize(log_img, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)