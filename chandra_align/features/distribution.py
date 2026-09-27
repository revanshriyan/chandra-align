"""Spatially balanced match selection and distribution metrics."""

from __future__ import annotations

import numpy as np


def _grid_shape(grid_shape):
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must be (rows, columns)")
    rows, cols = (int(v) for v in grid_shape)
    if rows <= 0 or cols <= 0:
        raise ValueError("grid_shape values must be positive")
    return rows, cols


def bucket_ids(points, image_shape, grid_shape=(8, 8)):
    """Return row-major bucket IDs for image coordinates, preserving point order."""
    rows, cols = _grid_shape(grid_shape)
    height, width = map(float, image_shape[:2])
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must have positive height and width")
    if not np.isfinite(pts).all():
        raise ValueError("point coordinates must be finite")
    x = np.clip(np.floor(pts[:, 0] * cols / width).astype(int), 0, cols - 1)
    y = np.clip(np.floor(pts[:, 1] * rows / height).astype(int), 0, rows - 1)
    return y * cols + x


def select_distributed_matches(
    points_ref,
    points_sec,
    image_shape,
    scores=None,
    grid_shape=(8, 8),
    max_per_bucket=8,
):
    """Select high-quality, spatially spread matches while preserving pairs.

    Within each bucket the strongest point is selected first, then each next point
    maximizes its minimum distance from already selected points (ANMS/SSC style).
    Returns the filtered reference points, secondary points, and original indices.
    """
    rows, cols = _grid_shape(grid_shape)
    quota = int(max_per_bucket)
    if quota <= 0:
        raise ValueError("max_per_bucket must be positive")
    ref = np.asarray(points_ref, dtype=np.float64).reshape(-1, 2)
    sec = np.asarray(points_sec, dtype=np.float64).reshape(-1, 2)
    if len(ref) != len(sec):
        raise ValueError("reference and secondary points must have equal length")
    if scores is None:
        quality = np.ones(len(ref), dtype=np.float64)
    else:
        quality = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(quality) != len(ref):
            raise ValueError("scores must have one value per match")
        quality = np.nan_to_num(quality, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf)
    if not np.isfinite(ref).all() or not np.isfinite(sec).all():
        raise ValueError("match coordinates must be finite")
    ids = bucket_ids(ref, image_shape, (rows, cols))
    chosen = []
    for bucket in range(rows * cols):
        candidates = np.flatnonzero(ids == bucket)
        if not len(candidates):
            continue
        order = candidates[np.argsort(-quality[candidates], kind="stable")]
        selected = [int(order[0])]
        remaining = list(map(int, order[1:]))
        while remaining and len(selected) < quota:
            distances = [np.min(np.linalg.norm(ref[c] - ref[selected], axis=1)) for c in remaining]
            best = int(np.argmax(distances))
            selected.append(remaining.pop(best))
        chosen.extend(selected)
    indices = np.asarray(sorted(chosen), dtype=int)
    return ref[indices], sec[indices], indices


def spatial_distribution_metrics(points, image_shape, grid_shape=(8, 8)):
    """Compute Shannon entropy (nats) and normalized spatial uniformity U."""
    rows, cols = _grid_shape(grid_shape)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return {"spatial_entropy": 0.0, "uniformity": 0.0, "occupied_buckets": 0,
                "total_buckets": rows * cols}
    ids = bucket_ids(pts, image_shape, (rows, cols))
    counts = np.bincount(ids, minlength=rows * cols).astype(np.float64)
    probabilities = counts / counts.sum()
    nonzero = probabilities[probabilities > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    total = rows * cols
    uniformity = entropy / np.log(total) if total > 1 else 1.0
    return {"spatial_entropy": entropy, "uniformity": float(np.clip(uniformity, 0.0, 1.0)),
            "occupied_buckets": int(np.count_nonzero(counts)), "total_buckets": total}
