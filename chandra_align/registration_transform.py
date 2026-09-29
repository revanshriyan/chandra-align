"""Robust similarity-transform estimation shared by app and local QA."""

from __future__ import annotations

import cv2
import numpy as np


MIN_REGISTRATION_INLIERS = 8


def refit_partial_affine_lstsq(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray | None:
    src = np.asarray(src_points, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_points, dtype=np.float64).reshape(-1, 2)
    N = len(src)
    if N < 2:
        return None
    A = np.zeros((2 * N, 4), dtype=np.float64)
    b = np.zeros(2 * N, dtype=np.float64)
    A[0::2, 0] = src[:, 0]
    A[0::2, 1] = -src[:, 1]
    A[0::2, 2] = 1.0
    A[1::2, 0] = src[:, 1]
    A[1::2, 1] = src[:, 0]
    A[1::2, 3] = 1.0
    b[0::2] = dst[:, 0]
    b[1::2] = dst[:, 1]
    try:
        p, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        if rank < 4 or not np.isfinite(p).all():
            return None
        a, b_param, tx, ty = p
        return np.array([[a, -b_param, tx], [b_param, a, ty]], dtype=np.float64)
    except (np.linalg.LinAlgError, ValueError):
        return None

def estimate_partial_affine(
    src_points: np.ndarray, dst_points: np.ndarray, ransac_threshold_px: float = 3.0
):
    """Estimate a finite 4-DOF similarity transform and its RANSAC mask."""
    src = np.asarray(src_points, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(dst_points, dtype=np.float32).reshape(-1, 2)
    if len(src) != len(dst) or len(src) < 3:
        return None, np.zeros(len(src), dtype=bool)
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src, dst = src[finite], dst[finite]
    try:
        threshold = float(ransac_threshold_px)
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("RANSAC threshold must be finite and positive")
        matrix, mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=threshold,
            maxIters=10000, confidence=0.999, refineIters=10,
        )
    except cv2.error:
        return None, np.zeros(len(src_points), dtype=bool)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return None, np.zeros(len(src_points), dtype=bool)
    accepted = mask.reshape(-1).astype(bool)
    if accepted.sum() >= 3:
        refitted = refit_partial_affine_lstsq(src[accepted], dst[accepted])
        if refitted is not None and np.isfinite(refitted).all():
            matrix = refitted
    if not finite.all():
        expanded = np.zeros(len(finite), dtype=bool)
        expanded[np.flatnonzero(finite)] = accepted
        accepted = expanded
    return matrix.astype(np.float64), accepted


def homogeneous_affine(matrix: np.ndarray) -> np.ndarray:
    """Represent a 2x3 partial affine in 3x3 form for existing exporters."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (2, 3) or not np.isfinite(value).all():
        raise ValueError("A finite 2x3 partial affine matrix is required.")
    result = np.eye(3, dtype=np.float64)
    result[:2, :] = value
    return result


def is_valid_registration(matrix: np.ndarray | None, inlier_count: int) -> bool:
    """Require a finite partial-affine matrix and at least eight RANSAC inliers."""
    if matrix is None or int(inlier_count) < MIN_REGISTRATION_INLIERS:
        return False
    value = np.asarray(matrix, dtype=np.float64)
    return value.shape == (2, 3) and bool(np.isfinite(value).all())
