"""Alignment model helpers and photogrammetric telemetry."""

import math
import numpy as np


def decompose_partial_affine(matrix):
    """Return translation, rotation, and uniform scale from a 2x3 affine."""
    affine = np.asarray(matrix, dtype=np.float64)
    if affine.shape != (2, 3) or not np.isfinite(affine).all():
        raise ValueError("Partial affine matrix must be a finite 2x3 array")
    a, b = float(affine[0, 0]), float(affine[1, 0])
    return {
        "delta_x_px": float(affine[0, 2]),
        "delta_y_px": float(affine[1, 2]),
        "rotation_deg": float(math.degrees(math.atan2(b, a))),
        "scale_s": float(math.sqrt(a * a + b * b)),
    }


__all__ = ["decompose_partial_affine"]
