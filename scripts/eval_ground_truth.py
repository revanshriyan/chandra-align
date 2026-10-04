#!/usr/bin/env python3
"""Evaluate a saved source-to-reference affine against independent point pairs.

FIT.npz must contain ``affine_matrix`` (2x3), ``pts_ref_inliers`` (Nx2), and
``pts_src_inliers`` (Nx2), all in the image pixel coordinates used by the run.
Ground-truth CSV columns are id,x_ref,y_ref,x_src,y_src.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = {"id", "x_ref", "y_ref", "x_src", "y_src"}
COINCIDENCE_TOLERANCE_PX = 1e-6


def _read_points(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain columns: {', '.join(sorted(REQUIRED_COLUMNS))}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no point rows")
    ids = [row["id"].strip() for row in rows]
    if any(not point_id for point_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Ground-truth ids must be non-empty and unique")
    try:
        ref = np.asarray([[float(row[k]) for k in ("x_ref", "y_ref")] for row in rows])
        src = np.asarray([[float(row[k]) for k in ("x_src", "y_src")] for row in rows])
    except (TypeError, ValueError) as exc:
        raise ValueError("Ground-truth coordinates must be numeric") from exc
    if not np.isfinite(ref).all() or not np.isfinite(src).all():
        raise ValueError("Ground-truth coordinates must be finite")
    return ids, ref, src


def evaluate_ground_truth(
    transform: np.ndarray,
    fit_ref: np.ndarray,
    fit_src: np.ndarray,
    gt_ref: np.ndarray,
    gt_src: np.ndarray,
    tolerance_px: float = COINCIDENCE_TOLERANCE_PX,
) -> float:
    """Return radial RMSE, refusing any checkpoint that overlaps a fit inlier."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (2, 3) or not np.isfinite(transform).all():
        raise ValueError("affine_matrix must be a finite 2x3 source-to-reference matrix")
    fit_ref = np.asarray(fit_ref, dtype=np.float64).reshape(-1, 2)
    fit_src = np.asarray(fit_src, dtype=np.float64).reshape(-1, 2)
    gt_ref = np.asarray(gt_ref, dtype=np.float64).reshape(-1, 2)
    gt_src = np.asarray(gt_src, dtype=np.float64).reshape(-1, 2)
    if len(fit_ref) != len(fit_src):
        raise ValueError("fit reference/source inlier arrays have different lengths")
    if len(gt_ref) != len(gt_src) or not len(gt_ref):
        raise ValueError("ground-truth reference/source arrays must have equal non-zero length")
    if not (np.isfinite(fit_ref).all() and np.isfinite(fit_src).all()
            and np.isfinite(gt_ref).all() and np.isfinite(gt_src).all()):
        raise ValueError("fit and ground-truth coordinates must be finite")
    for i, (ref_point, src_point) in enumerate(zip(gt_ref, gt_src)):
        if len(fit_ref) and (
            np.any(np.linalg.norm(fit_ref - ref_point, axis=1) <= tolerance_px)
            or np.any(np.linalg.norm(fit_src - src_point, axis=1) <= tolerance_px)
        ):
            raise ValueError(
                f"CIRCULARITY GUARD: ground-truth row {i + 1} coincides with a fit inlier "
                f"(tolerance {tolerance_px:g} px)"
            )
    predicted_ref = src_point_transform(transform, gt_src)
    residual = predicted_ref - gt_ref
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def src_point_transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ matrix[:, :2].T + matrix[:, 2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair", help="Pair identifier for the report")
    parser.add_argument("--ground-truth", required=True, type=Path, help="CSV: id,x_ref,y_ref,x_src,y_src")
    parser.add_argument("--fit-artifact", required=True, type=Path, help="NPZ with affine_matrix and fit inlier coordinates")
    args = parser.parse_args()
    try:
        ids, gt_ref, gt_src = _read_points(args.ground_truth)
        with np.load(args.fit_artifact, allow_pickle=False) as fit:
            needed = {"affine_matrix", "pts_ref_inliers", "pts_src_inliers"}
            if not needed.issubset(fit.files):
                raise ValueError(f"fit artifact must contain {', '.join(sorted(needed))}")
            rmse = evaluate_ground_truth(
                fit["affine_matrix"], fit["pts_ref_inliers"], fit["pts_src_inliers"],
                gt_ref, gt_src,
            )
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(f"pair: {args.pair}")
    print(f"ground_truth_points: {len(ids)}")
    print("ground_truth_rmse_px: {:.6f}".format(rmse))
    print("circularity_guard: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
