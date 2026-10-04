"""Propose review-only landmarks using Harris corners and NCC, not SIFT/LightGlue.

This is a candidate generator, not a ground-truth annotator. All generated
coordinates require visual review before they can be used for evaluation.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ground_truth" / "candidates"
LITE = ROOT / "data" / "qa" / "lite" / "chandra-align-images"
FULL = ROOT / "data" / "qa" / "fullres"
sys.path.insert(0, str(ROOT))


def load_pairs():
    import app

    synthetic_ref = cv2.imread(str(LITE / "lunar_reference.png"), cv2.IMREAD_GRAYSCALE)
    synthetic_sec = cv2.imread(str(LITE / "lunar_target_gentle.png"), cv2.IMREAD_GRAYSCALE)
    oh_ref = app.load_lunar_raster(str(FULL / "ch2_ohr_nrp_20240425T1209509264_d_img_d18.img"), max_dimension=4096)
    oh_sec = app.load_lunar_raster(str(FULL / "ch2_ohr_nrp_20240425T1406019344_d_img_d18.img"), max_dimension=4096)

    def read_tmc(path: Path, window: Window) -> np.ndarray:
        with rasterio.open(path.with_suffix(".xml")) as ds:
            values = ds.read(1, window=window, masked=True)
            values = np.asarray(values.astype(np.float32).filled(np.nan))
        return app.load_lunar_raster(values, max_dimension=4096)

    tmc_ref = read_tmc(FULL / "ch2_tmc_ncf_20231101T0125121344_d_img_d18.img", Window(361, 93328, 2048, 2048))
    tmc_sec = read_tmc(FULL / "ch2_tmc_ncn_20231101T0125121377_d_img_d18.img", Window(416, 101611, 2048, 2048))
    return {
        "OHRC_pair": (oh_ref, oh_sec),
        "TMC2_fore_nadir": (tmc_ref, tmc_sec),
    }


def make_candidates(ref: np.ndarray, src: np.ndarray):
    # Downsample only for proposal; coordinates are mapped back to canonical pixels.
    side = 640
    scale_r = side / max(ref.shape)
    scale_s = side / max(src.shape)
    r = cv2.resize(ref, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_AREA)
    s = cv2.resize(src, None, fx=scale_s, fy=scale_s, interpolation=cv2.INTER_AREA)
    if r.ndim != 2 or s.ndim != 2:
        raise ValueError("Expected grayscale images")
    r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    s = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    corners = cv2.goodFeaturesToTrack(
        r, maxCorners=180, qualityLevel=0.01, minDistance=16,
        blockSize=5, useHarrisDetector=True, k=0.04,
    )
    if corners is None:
        return [], r, s, scale_r, scale_s

    rad = 7
    matches = []
    for (x0, y0) in np.round(corners.reshape(-1, 2)).astype(int):
        if x0 < rad or y0 < rad or x0 >= r.shape[1] - rad or y0 >= r.shape[0] - rad:
            continue
        patch = r[y0-rad:y0+rad+1, x0-rad:x0+rad+1]
        scores = cv2.matchTemplate(s, patch, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(scores)
        if not np.isfinite(score) or score < 0.70:
            continue
        xs, ys = loc[0] + rad, loc[1] + rad
        matches.append((float(score), (float(x0), float(y0)), (float(xs), float(ys))))

    matches.sort(key=lambda row: row[0], reverse=True)
    unique = []
    for item in matches:
        if any(np.linalg.norm(np.subtract(item[2], old[2])) < 10 for old in unique):
            continue
        unique.append(item)
    if len(unique) < 4:
        return [], r, s, scale_r, scale_s

    p_ref = np.float32([m[1] for m in unique]).reshape(-1, 1, 2)
    p_src = np.float32([m[2] for m in unique]).reshape(-1, 1, 2)
    model, mask = cv2.findHomography(p_ref, p_src, cv2.RANSAC, 3.0)
    if model is None or mask is None:
        return [], r, s, scale_r, scale_s
    inliers = np.flatnonzero(mask.ravel())
    verified = []
    for index in inliers:
        score, pr, ps = unique[int(index)]
        pred = cv2.perspectiveTransform(np.float32([[pr]]), model)[0, 0]
        err = float(np.linalg.norm(pred - np.float32(ps)))
        verified.append({"score": score, "ref": pr, "src": ps, "model_error": err})
    verified.sort(key=lambda row: row["score"], reverse=True)
    return verified[:20], r, s, scale_r, scale_s


def write_package(pair: str, ref: np.ndarray, src: np.ndarray):
    points, r_small, s_small, scale_r, scale_s = make_candidates(ref, src)
    csv_path = OUT / f"{pair}_candidates.csv"
    fields = ["id", "x_ref", "y_ref", "x_src", "y_src", "ncc", "model_error_small_px", "review_status"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, item in enumerate(points, 1):
            writer.writerow({
                "id": f"{pair}-{i:02d}",
                "x_ref": round(item["ref"][0] / scale_r, 2),
                "y_ref": round(item["ref"][1] / scale_r, 2),
                "x_src": round(item["src"][0] / scale_s, 2),
                "y_src": round(item["src"][1] / scale_s, 2),
                "ncc": round(item["score"], 5),
                "model_error_small_px": round(item["model_error"], 3),
                "review_status": "PENDING_HUMAN",
            })

    # Each row is a paired 81x81 crop at the proposal center, with candidate id.
    cell_w, chip, label_h = 2 * 96 + 20, 81, 24
    cols = 4
    rows = max(1, (len(points) + cols - 1) // cols)
    sheet = np.full((rows * (chip + label_h + 12), cols * (cell_w + 10), 3), 245, np.uint8)
    for i, item in enumerate(points):
        row, col = divmod(i, cols)
        xbase, ybase = col * (cell_w + 10), row * (chip + label_h + 12)
        pr = tuple(int(v) for v in np.round(item["ref"]))
        ps = tuple(int(v) for v in np.round(item["src"]))
        rc = cv2.getRectSubPix(r_small, (chip, chip), pr)
        sc = cv2.getRectSubPix(s_small, (chip, chip), ps)
        y0 = ybase + label_h
        sheet[y0:y0+chip, xbase:xbase+chip] = cv2.cvtColor(rc, cv2.COLOR_GRAY2BGR)
        sheet[y0:y0+chip, xbase+chip+10:xbase+2*chip+10] = cv2.cvtColor(sc, cv2.COLOR_GRAY2BGR)
        cv2.putText(sheet, f"{pair}-{i+1:02d} NCC {item['score']:.3f}", (xbase, ybase+17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.rectangle(sheet, (xbase, y0), (xbase+chip-1, y0+chip-1), (30, 150, 30), 1)
        cv2.rectangle(sheet, (xbase+chip+10, y0), (xbase+2*chip+9, y0+chip-1), (30, 150, 30), 1)
    png_path = OUT / f"{pair}_candidate_chips.png"
    cv2.imwrite(str(png_path), sheet)
    print(f"{pair}: {len(points)} proposals; {csv_path}; {png_path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for pair, (ref, src) in load_pairs().items():
        if ref is None or src is None:
            raise RuntimeError(f"Failed to load canonical inputs for {pair}")
        write_package(pair, ref, src)


if __name__ == "__main__":
    main()
