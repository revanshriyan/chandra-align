"""Native-resolution raster-window matching without assembling full strips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rasterio.windows import Window


@dataclass(frozen=True)
class TileWindow:
    x: int
    y: int
    width: int
    height: int
    # Core bounds are local to the tile; overlap pixels outside the core provide
    # context but do not contribute duplicate correspondences.
    core: tuple[int, int, int, int]


def _axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length < 1 or tile_size < 1 or overlap < 0 or overlap >= tile_size:
        raise ValueError("length/tile_size must be positive and 0 <= overlap < tile_size")
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def iter_tile_windows(width: int, height: int, tile_size: int = 2048, overlap: int = 256):
    """Yield a deterministic row-major tile grid with a non-overlap core per tile."""
    xs = _axis_starts(int(width), int(tile_size), int(overlap))
    ys = _axis_starts(int(height), int(tile_size), int(overlap))

    def core_bounds(starts, index, extent, actual_tile):
        start = starts[index]
        left = 0 if index == 0 else (starts[index - 1] + actual_tile + start) // 2
        right = extent if index == len(starts) - 1 else (start + actual_tile + starts[index + 1]) // 2
        return max(0, left - start), min(actual_tile, right - start)

    for yi, y in enumerate(ys):
        th = min(tile_size, height - y)
        cy0, cy1 = core_bounds(ys, yi, height, th)
        for xi, x in enumerate(xs):
            tw = min(tile_size, width - x)
            cx0, cx1 = core_bounds(xs, xi, width, tw)
            yield TileWindow(x, y, tw, th, (cx0, cy0, cx1, cy1))


def mosaic_tile_arrays(width: int, height: int, tiles):
    """Assemble non-overlapping tile cores; primarily useful for correctness checks."""
    mosaic = None
    written = np.zeros((height, width), dtype=bool)
    for tile, array in tiles:
        arr = np.asarray(array)
        if arr.shape[:2] != (tile.height, tile.width):
            raise ValueError("tile array dimensions disagree with its window")
        if mosaic is None:
            mosaic = np.zeros((height, width) + arr.shape[2:], dtype=arr.dtype)
        x0, y0, x1, y1 = tile.core
        dx0, dy0 = tile.x + x0, tile.y + y0
        dx1, dy1 = tile.x + x1, tile.y + y1
        if np.any(written[dy0:dy1, dx0:dx1]):
            raise ValueError("tile cores overlap")
        mosaic[dy0:dy1, dx0:dx1] = arr[y0:y1, x0:x1]
        written[dy0:dy1, dx0:dx1] = True
    if mosaic is None or not written.all():
        raise ValueError("tile cores did not cover the requested mosaic")
    return mosaic


def _affine_apply(matrix, xy):
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return pts @ np.asarray(matrix, dtype=np.float64)[:, :2].T + np.asarray(matrix)[:, 2]


def _mapped_origin(reference_to_source, x, y, width, height):
    center = np.array([[x + width / 2, y + height / 2]], dtype=np.float64)
    mapped = _affine_apply(reference_to_source, center)[0]
    return int(round(mapped[0] - width / 2)), int(round(mapped[1] - height / 2))


def _open_product(path):
    import rasterio
    path = Path(path)
    label = path.with_suffix(".xml") if path.suffix.lower() == ".img" else path
    return rasterio.open(label if label.exists() else path)


def tiled_registration(
    reference_path,
    source_path,
    reference_window: tuple[int, int, int, int],
    reference_to_source,
    matcher_fn,
    *,
    tile_size: int = 2048,
    overlap: int = 256,
    ransac_threshold_px: float = 3.0,
    sensor_name: str = "OHRC",
):
    """Read and match paired native-resolution tiles, then score with app gate functions.

    ``reference_to_source`` is a 2x3 approximate pixel mapping used only to
    choose corresponding source windows. Every point is translated into strip
    coordinates before one global robust fit and the existing gate functions.
    """
    import app
    from chandra_align.metrics import compute_quadrant_metrics, validate_registration_gate
    from chandra_align.trust import evaluate
    from chandra_align.preprocessing.general import apply_clahe

    rx, ry, rw, rh = map(int, reference_window)
    mapping = np.asarray(reference_to_source, dtype=np.float64).reshape(2, 3)
    collected_ref, collected_src = [], []
    tiles_read = 0
    with _open_product(reference_path) as ref_ds, _open_product(source_path) as src_ds:
        sx0, sy0 = _mapped_origin(mapping, rx, ry, rw, rh)
        if rx < 0 or ry < 0 or rx + rw > ref_ds.width or ry + rh > ref_ds.height:
            raise ValueError("reference window is outside its raster")
        if sx0 < 0 or sy0 < 0 or sx0 + rw > src_ds.width or sy0 + rh > src_ds.height:
            raise ValueError("mapped source window is outside its raster")
        for tile in iter_tile_windows(rw, rh, tile_size, overlap):
            gx, gy = rx + tile.x, ry + tile.y
            source_x, source_y = _mapped_origin(mapping, gx, gy, tile.width, tile.height)
            if source_x < 0 or source_y < 0 or source_x + tile.width > src_ds.width or source_y + tile.height > src_ds.height:
                continue
            ref_raw = ref_ds.read(1, window=Window(gx, gy, tile.width, tile.height), masked=True)
            src_raw = src_ds.read(1, window=Window(source_x, source_y, tile.width, tile.height), masked=True)
            ref_img = app.load_lunar_raster(np.asarray(ref_raw.astype(np.float32).filled(np.nan)), max_dimension=tile_size)
            src_img = app.load_lunar_raster(np.asarray(src_raw.astype(np.float32).filled(np.nan)), max_dimension=tile_size)
            if ref_img is None or src_img is None:
                continue
            ref_processed = apply_clahe(ref_img, clip_limit=3.0, normalize_range=(0.0, 255.0))
            src_processed = apply_clahe(src_img, clip_limit=3.0, normalize_range=(0.0, 255.0))
            pts_ref, pts_src, _engine, _diag = matcher_fn(ref_processed, src_processed, ransac_threshold_px)
            pts_ref = np.asarray(pts_ref, dtype=np.float32).reshape(-1, 2)
            pts_src = np.asarray(pts_src, dtype=np.float32).reshape(-1, 2)
            if len(pts_ref) != len(pts_src):
                raise ValueError("matcher returned correspondence arrays with different lengths")
            cx0, cy0, cx1, cy1 = tile.core
            keep = ((pts_ref[:, 0] >= cx0) & (pts_ref[:, 0] < cx1)
                    & (pts_ref[:, 1] >= cy0) & (pts_ref[:, 1] < cy1))
            if keep.any():
                collected_ref.append(pts_ref[keep] + np.array([tile.x, tile.y], np.float32))
                collected_src.append(pts_src[keep] + np.array([source_x, source_y], np.float32))
            tiles_read += 1

    all_ref_local = np.concatenate(collected_ref, axis=0) if collected_ref else np.empty((0, 2), np.float32)
    all_ref_local[:, 0] += rx
    all_ref_local[:, 1] += ry
    all_src = np.concatenate(collected_src, axis=0) if collected_src else np.empty((0, 2), np.float32)
    if len(all_ref_local) < 3:
        return {"verdict": "REJECTED", "gate": "insufficient correspondences",
                "tiles": tiles_read, "correspondences": len(all_ref_local), "inliers": 0,
                "rmse_insample": None, "rmse_heldout": None, "n_heldout_points": 0,
                "entropy": 0.0, "quadrants": "0/4 (0,0,0,0)"}

    # Match app._align_core's robust partial-affine fitting threshold and its
    # existing held-out split + gate APIs. Gate policy is not reimplemented.
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        all_src, all_ref_local, method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold_px),
    )
    if matrix is None or inlier_mask is None:
        return {"verdict": "REJECTED", "gate": "global affine fit failed",
                "tiles": tiles_read, "correspondences": len(all_ref_local), "inliers": 0,
                "rmse_insample": None, "rmse_heldout": None, "n_heldout_points": 0,
                "entropy": 0.0, "quadrants": "0/4 (0,0,0,0)"}
    inliers = inlier_mask.ravel().astype(bool)
    ref_inliers, src_inliers = all_ref_local[inliers], all_src[inliers]
    projected = cv2.transform(src_inliers.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    residual = ref_inliers - projected
    rmse_in = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))) if len(residual) else None
    quad_metrics, entropy = compute_quadrant_metrics(ref_inliers - np.array([rx, ry]), residual, (rh, rw))
    _, heldout = evaluate(matrix, src_inliers, ref_inliers)
    rmse_out = heldout.get("rmse_px") if heldout.get("held_out") else None
    rmse_gate = rmse_out if rmse_out is not None else (rmse_in if rmse_in is not None else float("inf"))
    status_message, status_code = validate_registration_gate(
        rmse_gate, len(ref_inliers), app.MIN_REGISTRATION_INLIERS, entropy, quad_metrics,
    )
    quad_counts = [int(quad_metrics.get(f"Q{i}", {}).get("inlier_count", 0)) for i in range(1, 5)]
    active = sum(count > 0 for count in quad_counts)
    return {
        "verdict": status_code, "gate": status_message, "tiles": tiles_read,
        "correspondences": len(all_ref_local), "inliers": int(len(ref_inliers)),
        "rmse_insample": rmse_in, "rmse_heldout": rmse_out,
        "n_heldout_points": int(heldout.get("n_check_points", 0)),
        "entropy": float(entropy), "quadrants": f"{active}/4 ({','.join(map(str, quad_counts))})",
        "affine": matrix.tolist(), "execution": "native-resolution windowed tiles",
        "ground_truth_rmse": None,
    }
