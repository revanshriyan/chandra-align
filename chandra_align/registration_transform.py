"""Robust similarity-transform estimation shared by app and local QA."""

from __future__ import annotations

import cv2
import numpy as np


MIN_REGISTRATION_INLIERS = 8


def estimate_partial_affine(src_points: np.ndarray, dst_points: np.ndarray):
    """Estimate a finite 4-DOF similarity transform and its RANSAC mask."""
    src = np.asarray(src_points, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(dst_points, dtype=np.float32).reshape(-1, 2)
    if len(src) != len(dst) or len(src) < 3:
        return None, np.zeros(len(src), dtype=bool)
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src, dst = src[finite], dst[finite]
    try:
        matrix, mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
            maxIters=10000, confidence=0.999, refineIters=10,
        )
    except cv2.error:
        return None, np.zeros(len(src_points), dtype=bool)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return None, np.zeros(len(src_points), dtype=bool)
    accepted = mask.reshape(-1).astype(bool)
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
