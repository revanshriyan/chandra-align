"""Coarse-to-fine correspondence matching with affine propagation."""

from __future__ import annotations

import cv2
import numpy as np


def _as_homogeneous(matrix: np.ndarray) -> np.ndarray:
    out = np.eye(3, dtype=np.float64)
    out[:2] = np.asarray(matrix, dtype=np.float64).reshape(2, 3)
    return out


def _scale_transform(matrix: np.ndarray, sx: float, sy: float) -> np.ndarray:
    scale = np.diag([float(sx), float(sy), 1.0])
    return (scale @ _as_homogeneous(matrix) @ np.linalg.inv(scale))[:2]


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return points @ matrix[:, :2].T + matrix[:, 2]


def coarse_to_fine_match(
    reference: np.ndarray,
    source: np.ndarray,
    match_fn,
    *,
    levels=(0.25, 0.5, 1.0),
    ransac_threshold_px: float = 3.0,
):
    """Match progressively finer levels and carry the affine between levels.

    The matcher sees the reference and a source image warped by the current
    estimate. A robust residual affine is composed with that estimate. Only
    correspondences from the finest level are returned, mapped back to the
    original source coordinate frame for the unchanged downstream fit and gate.
    """
    ref = np.asarray(reference)
    src = np.asarray(source)
    if ref.ndim != 2 or src.ndim != 2 or ref.size == 0 or src.size == 0:
        raise ValueError("coarse-to-fine matching requires non-empty grayscale images")
    if not np.isfinite(ransac_threshold_px) or ransac_threshold_px <= 0:
        raise ValueError("RANSAC threshold must be finite and positive")
    scales = tuple(float(level) for level in levels)
    if len(scales) != 3 or any(not 0.0 < level <= 1.0 for level in scales):
        raise ValueError("exactly three pyramid levels in (0, 1] are required")
    if tuple(sorted(scales)) != scales or len(set(scales)) != 3 or scales[-1] != 1.0:
        raise ValueError("pyramid levels must increase coarse-to-fine and end at 1.0")

    ref_h, ref_w = ref.shape
    src_h, src_w = src.shape
    source_to_reference = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    last = None
    level_log = []

    for factor in scales:
        rw, rh = max(1, round(ref_w * factor)), max(1, round(ref_h * factor))
        sw, sh = max(1, round(src_w * factor)), max(1, round(src_h * factor))
        ref_level = cv2.resize(ref, (rw, rh), interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
        src_level = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
        sx_ref, sy_ref = rw / float(ref_w), rh / float(ref_h)
        sx_src, sy_src = sw / float(src_w), sh / float(src_h)

        # Current source-to-reference estimate represented at this level.
        source_level_to_ref = (
            np.diag([sx_ref, sy_ref, 1.0])
            @ _as_homogeneous(source_to_reference)
            @ np.diag([1.0 / sx_src, 1.0 / sy_src, 1.0])
        )[:2]
        warp_transform = source_level_to_ref.copy()
        warped = cv2.warpAffine(
            src_level, source_level_to_ref, (rw, rh),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )
        pts_ref, pts_warped, engine, diagnostics = match_fn(ref_level, warped, ransac_threshold_px)
        pts_ref = np.asarray(pts_ref, dtype=np.float32).reshape(-1, 2)
        pts_warped = np.asarray(pts_warped, dtype=np.float32).reshape(-1, 2)
        if len(pts_ref) != len(pts_warped):
            raise ValueError("matcher returned correspondence arrays with different lengths")
        stage = {"scale": factor, "correspondences": int(len(pts_ref)), "refined": False}
        if len(pts_ref) >= 3:
            delta, inliers = cv2.estimateAffinePartial2D(
                pts_warped, pts_ref, method=cv2.RANSAC,
                ransacReprojThreshold=float(ransac_threshold_px),
            )
            if delta is not None and inliers is not None and int(inliers.sum()) >= 3:
                source_level_to_ref = (_as_homogeneous(delta) @ _as_homogeneous(warp_transform))[:2]
                # Convert the accumulated level transform back to original pixel units.
                ref_scale = np.diag([sx_ref, sy_ref, 1.0])
                src_scale_inv = np.diag([1.0 / sx_src, 1.0 / sy_src, 1.0])
                source_to_reference = (
                    np.linalg.inv(ref_scale)
                    @ _as_homogeneous(source_level_to_ref)
                    @ np.linalg.inv(src_scale_inv)
                )[:2]
                stage["refined"] = True
                stage["ransac_inliers"] = int(inliers.sum())
        level_log.append(stage)
        last = (pts_ref, pts_warped, warp_transform, sx_ref, sy_ref, sx_src, sy_src, engine, diagnostics)

    pts_ref, pts_warped, source_level_to_ref, sx_ref, sy_ref, sx_src, sy_src, engine, diagnostics = last
    if len(pts_ref):
        inverse_level_transform = np.linalg.inv(_as_homogeneous(source_level_to_ref))[:2]
        source_scaled = _transform_points(inverse_level_transform, pts_warped)
        pts_source = np.column_stack((source_scaled[:, 0] / sx_src, source_scaled[:, 1] / sy_src))
        pts_ref_original = np.column_stack((pts_ref[:, 0] / sx_ref, pts_ref[:, 1] / sy_ref))
    else:
        pts_source = np.empty((0, 2), dtype=np.float32)
        pts_ref_original = np.empty((0, 2), dtype=np.float32)
    out_diagnostics = dict(diagnostics or {})
    out_diagnostics["coarse_to_fine"] = True
    out_diagnostics["pyramid_levels"] = level_log
    out_diagnostics["coarse_to_fine_matrix"] = source_to_reference.tolist()
    return pts_ref_original.astype(np.float32), pts_source.astype(np.float32), engine, out_diagnostics
