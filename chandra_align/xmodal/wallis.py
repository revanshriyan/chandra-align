"""Wallis local-contrast normalization (Phase 9 preprocessing branch).

Wallis filtering re-centers and re-scales every pixel against its local
neighbourhood statistics, which evens out illumination gradients and lifts
low-contrast texture before matching.

What it does NOT do: it does NOT fix polarity reversal. If a crater rim is
bright in one sensor and dark in the other, Wallis keeps that sign flip —
only the local mean/std change. Cross-modal pairs with inverted contrast
need the polarity-invariant path (goa_nmi), not this one.
"""

import cv2
import numpy as np


def wallis_normalize(img, window=51, target_mean=127.0, target_std=40.0):
    """Normalize local contrast with a Wallis filter.

    out = (img - local_mean) * (target_std / (local_std + eps)) + target_mean

    local_mean and local_std come from box filters of size `window`.
    The gain (target_std / local_std) is clipped to [0.25, 4.0] so flat,
    near-zero-variance regions are not amplified into noise.

    Parameters
    ----------
    img : (H, W) array-like
        Single-channel image. Any numeric dtype; converted internally.
    window : int
        Box-filter side length (odd preferred). Must be >= 3.
    target_mean, target_std : float
        Desired local mean and standard deviation of the output.

    Returns
    -------
    (H, W) float64 array in [0, 255].
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"wallis_normalize expects a 2-D image, got shape {img.shape}")
    if int(window) < 3:
        raise ValueError(f"window must be >= 3, got {window}")
    w = int(window) | 1  # odd window for a symmetric neighbourhood

    local_mean = cv2.boxFilter(img, -1, (w, w), normalize=True,
                               borderType=cv2.BORDER_REFLECT)
    local_sq_mean = cv2.boxFilter(img * img, -1, (w, w), normalize=True,
                                  borderType=cv2.BORDER_REFLECT)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)
    local_std = np.sqrt(local_var)

    eps = 1e-6
    gain = target_std / (local_std + eps)
    gain = np.clip(gain, 0.25, 4.0)  # do not invent texture in flat regions

    out = (img - local_mean) * gain + target_mean
    return np.clip(out, 0.0, 255.0)
