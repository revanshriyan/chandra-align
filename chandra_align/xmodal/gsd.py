"""Common-GSD resampling (Phase 9 cross-modal branch).

Cross-modal pairs (e.g. IIRS vs TMC-2) live at different ground sampling
distances. Matching them at ~1:1 pixel correspondence silently compares
different ground scales, so one side is resampled to the other's working
GSD before any detector runs.

What it does NOT do: it does not invent resolution (upsampling adds no
information), it does not pick which GSD is "right" — the caller does —
and it NEVER silently assumes 1:1. Unknown GSD raises instead of guessing.
"""

import cv2
import numpy as np

_UNKNOWN_GSD_MSG = "GSD unknown — explicit routing required, never silent 1:1"


def to_common_gsd(img, src_gsd_m, dst_gsd_m):
    """Resample `img` from `src_gsd_m` to `dst_gsd_m` meters/px.

    scale_factor = src_gsd_m / dst_gsd_m (output pixels per input pixel
    along each axis). Downsampling (scale < 1) uses cv2.INTER_AREA,
    upsampling (scale >= 1) uses cv2.INTER_CUBIC.

    Parameters
    ----------
    img : (H, W) or (H, W, C) array-like
    src_gsd_m, dst_gsd_m : float
        Ground sampling distances in meters per pixel. Either may be None.

    Returns
    -------
    (resampled, scale_factor)

    Raises
    ------
    ValueError
        If either GSD is None or <= 0. This is deliberate: an unknown GSD
        must be routed explicitly by the caller, never defaulted to 1:1.
    """
    if src_gsd_m is None or dst_gsd_m is None:
        raise ValueError(_UNKNOWN_GSD_MSG)
    try:
        src_gsd = float(src_gsd_m)
        dst_gsd = float(dst_gsd_m)
    except (TypeError, ValueError):
        raise ValueError(_UNKNOWN_GSD_MSG)
    if not (src_gsd > 0) or not (dst_gsd > 0):
        raise ValueError(_UNKNOWN_GSD_MSG)

    img = np.asarray(img)
    scale = src_gsd / dst_gsd
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resampled = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return resampled, float(scale)
