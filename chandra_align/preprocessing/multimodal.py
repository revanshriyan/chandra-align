"""Structural preprocessing for optical, SWIR, and thermal image pairs."""

import cv2
import numpy as np


def _gray_float(image):
    image = np.asarray(image)
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[..., 0]
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            raise ValueError("multimodal preprocessing expects 1, 3, or 4 channels")
    if image.ndim != 2 or image.size == 0:
        raise ValueError("multimodal preprocessing expects non-empty grayscale images")
    gray = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(gray, (1, 99))
    if hi > lo:
        gray = np.clip((gray - lo) * (255.0 / (hi - lo)), 0, 255)
    else:
        gray = np.zeros(gray.shape, dtype=np.float32)
    return np.ascontiguousarray(gray, dtype=np.float32)


def gradient_structure(image):
    """Create an inversion-tolerant gradient-magnitude image for SIFT matching."""
    gray = _gray_float(image)
    smooth = cv2.GaussianBlur(gray, (0, 0), 0.8)
    gx = cv2.Scharr(smooth, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(smooth, cv2.CV_32F, 0, 1)
    magnitude = np.hypot(gx, gy)
    lo, hi = np.percentile(magnitude, (1, 99))
    if hi <= lo:
        return np.zeros(gray.shape, dtype=np.uint8)
    return np.clip((magnitude - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def preprocess_multimodal_pair(reference, secondary):
    """Convert both modalities to structural representations robust to inversion."""
    return gradient_structure(reference), gradient_structure(secondary)


def gaussian_scale_pyramid(image, min_dimension=64, max_levels=8):
    """Build a Gaussian pyramid, returning (image, original_pixel_scale) levels."""
    current = np.asarray(image)
    if current.ndim not in (2, 3) or current.size == 0:
        raise ValueError("pyramid input must be a non-empty 2D or 3D image")
    levels = [(current, 1.0)]
    scale = 1.0
    for _ in range(max(0, int(max_levels) - 1)):
        if min(current.shape[:2]) < 2 * min_dimension:
            break
        current = cv2.pyrDown(current)
        scale *= 0.5
        levels.append((current, scale))
    return levels


def resize_to_common_ground_sample(image, image_gsd_m, target_gsd_m, minimum_long_side=128):
    """Downsample only the finer-resolution image toward the coarser GSD.

    Returns the resized image and its linear pixel scale relative to the original;
    callers map detected points back by dividing by that scale.
    """
    gsd = float(image_gsd_m)
    target = float(target_gsd_m)
    if not np.isfinite(gsd) or not np.isfinite(target) or gsd <= 0 or target <= 0:
        raise ValueError("GSD values must be finite and positive")
    # Sensor metadata can imply aggressive scale factors (e.g. OHRC/IIRS). Keep
    # enough raster support for feature extraction when the inputs are cropped
    # to similar pixel dimensions and their true footprints are not supplied.
    h, w = image.shape[:2]
    minimum_long_side = max(1, int(minimum_long_side))
    minimum_scale = min(1.0, float(minimum_long_side) / max(h, w))
    scale = max(min(1.0, gsd / target), minimum_scale)
    if scale >= 0.999:
        return image, 1.0
    pyramid = gaussian_scale_pyramid(image, min_dimension=32)
    level, level_scale = max((entry for entry in pyramid if entry[1] >= scale),
                             key=lambda entry: entry[1], default=pyramid[-1])
    target_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    if level_scale > scale + 1e-6:
        level = cv2.resize(level, target_size, interpolation=cv2.INTER_AREA)
    actual_scale = level.shape[1] / float(w)
    return level, actual_scale
