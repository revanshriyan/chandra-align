"""scripts/run_scale_ratio_real_sensor.py — scale-invariance on REAL sensor texture.

Extension of scripts/run_scale_ratio.py: instead of synthetic crater blobs, the
source scene is a real lunar image crop at native resolution (the ohrc_01
benchmark crop, 2048x2048 uint8, real OHRC regolith texture). The moving image
is built with an exactly-known sub-pixel shift, then downsampled by ratio r
(INTER_AREA, simulating a coarser sensor) and upsampled back to the native grid
(INTER_CUBIC). The SIFT/RANSAC path with the FROZEN gate thresholds is run per
ratio. Nothing about the gates or matcher logic is changed.

Results are script-generated into results/table_scale_ratio_real_sensor.csv —
never hand-edited. Truth is used for scoring only, never for fitting
(no-peeking).

Phase 11 lesson: truth RMSE (transform correctness vs exact resampling truth)
is reported SEPARATELY from inlier self-consistency RMSE — never conflated.

Usage:
    python scripts/run_scale_ratio_real_sensor.py \
        [--out results/table_scale_ratio_real_sensor.csv]
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Frozen gate thresholds (do not change — see chandra_align/metrics/quadrant.py).
_GATE_ACCEPT_RMSE = 0.50
_GATE_COARSE_RMSE = 2.50
_GATE_ACCEPT_ENTROPY = 0.75
_GATE_COARSE_ENTROPY = 0.50
_GATE_ACCEPT_QUADS = 3
_GATE_COARSE_QUADS = 2
_MIN_INLIERS = 8

SEED = 7
RATIOS = [1, 2, 4, 8, 16]
DX, DY = 7.3, -3.9  # exact known shift, mirrors the synthetic pilot

TEXTURE_SOURCE = (
    "ohrc_01_reference.png: real OHRC benchmark crop (2048x2048 uint8, "
    "native-resolution regolith texture from data/benchmark_crops/)"
)


def load_texture():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "data", "benchmark_crops",
                        "ohrc_01_reference.png")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"texture crop not found: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def make_pair_real(ref, dx=DX, dy=DY):
    """Real-texture pair with exactly-known sub-pixel shift (scoring truth)."""
    M_gt = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
    h, w = ref.shape[:2]
    mov = cv2.warpAffine(ref, M_gt, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    return mov, M_gt


def sift_match(img_a, img_b, n_features=8000, lowes_ratio=0.75):
    sift = cv2.SIFT_create(nfeatures=n_features)
    a, b = img_a, img_b
    if a.dtype != np.uint8:
        a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if b.dtype != np.uint8:
        b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    k_a, des_a = sift.detectAndCompute(a, None)
    k_b, des_b = sift.detectAndCompute(b, None)
    if des_a is None or des_b is None or len(k_a) < 4 or len(k_b) < 4:
        return (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32))
    bf = cv2.BFMatcher()
    pairs = bf.knnMatch(des_a, des_b, k=2)
    good_a, good_b = [], []
    for m, n in pairs:
        if m.distance < lowes_ratio * n.distance:
            good_a.append(k_a[m.queryIdx].pt)
            good_b.append(k_b[m.trainIdx].pt)
    if len(good_a) < 4:
        return (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32))
    return (np.array(good_a, np.float32), np.array(good_b, np.float32))


def quadrant_entropy(pts, shape):
    h, w = shape[:2]
    x, y = pts[:, 0], pts[:, 1]
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    q = ((x[valid] >= w / 2).astype(int)
         + 2 * (y[valid] >= h / 2).astype(int))
    counts = np.bincount(q, minlength=4)
    total = counts.sum()
    if total == 0:
        return 0.0, 0
    p = counts[counts > 0] / total
    entropy = float(-np.sum(p * np.log2(p)))
    return entropy, int((counts > 0).sum())


def transform_rmse_vs_truth(M_est, M_gt, shape, n=200, seed=SEED):
    """RMSE of M_est vs M_gt over a deterministic grid (scoring only)."""
    rng = np.random.default_rng(seed)
    h, w = shape[:2]
    pts = np.column_stack([rng.uniform(0, w, n), rng.uniform(0, h, n)])
    ones = np.ones((n, 1))
    ph = np.hstack([pts, ones])
    pe = (M_est @ ph.T).T if M_est.shape[0] == 2 else (M_est[:2] @ ph.T).T
    pg = (M_gt @ ph.T).T
    return float(np.sqrt(np.mean(np.sum((pe - pg) ** 2, axis=1))))


def run_one_ratio(ref, mov, M_gt, ratio):
    cv2.setRNGSeed(SEED)
    h, w = ref.shape[:2]
    if ratio == 1:
        mov_r = mov
    else:
        small = cv2.resize(mov, (w // ratio, h // ratio),
                           interpolation=cv2.INTER_AREA)
        mov_r = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    pts_a, pts_b = sift_match(ref, mov_r)
    n_corr = len(pts_a)
    base = {"ratio": ratio, "n_corr": n_corr, "n_inliers": 0,
            "rmse_inlier_px": "UNMEASURED", "rmse_vs_truth_px": "UNMEASURED",
            "entropy": 0.0, "quadrants": 0, "verdict": "DEGENERATE_FAILURE"}
    if n_corr < 4:
        return base
    M_est, inl = cv2.estimateAffinePartial2D(
        pts_a, pts_b, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=2000, confidence=0.99)
    if M_est is None or inl is None:
        return base
    inl = inl.ravel().astype(bool)
    n_inl = int(inl.sum())
    if n_inl < 4:
        base["n_inliers"] = n_inl
        return base
    # Inlier RMSE via refit (no-peeking: truth never used for fitting).
    M_refit, _ = cv2.estimateAffinePartial2D(
        pts_a[inl], pts_b[inl], method=cv2.LMEDS)
    if M_refit is None:
        M_refit = M_est
    pred = (M_refit @ np.hstack(
        [pts_a[inl], np.ones((n_inl, 1))]).T).T
    rmse_inl = float(np.sqrt(np.mean(np.sum((pred - pts_b[inl]) ** 2, axis=1))))
    rmse_truth = transform_rmse_vs_truth(M_refit, M_gt, ref.shape)
    entropy, quads = quadrant_entropy(pts_a[inl], ref.shape)
    # Frozen gate (same thresholds as pilot).
    if (rmse_inl <= _GATE_ACCEPT_RMSE and n_inl >= _MIN_INLIERS
            and entropy >= _GATE_ACCEPT_ENTROPY and quads >= _GATE_ACCEPT_QUADS):
        verdict = "SUCCESS_SUBPIXEL"
    elif (rmse_inl <= _GATE_COARSE_RMSE and n_inl >= _MIN_INLIERS
            and entropy >= _GATE_COARSE_ENTROPY and quads >= _GATE_COARSE_QUADS):
        verdict = "COARSE_ADVISORY"
    else:
        verdict = "DEGENERATE_FAILURE"
    return {"ratio": ratio, "n_corr": n_corr, "n_inliers": n_inl,
            "rmse_inlier_px": round(rmse_inl, 4),
            "rmse_vs_truth_px": round(rmse_truth, 4),
            "entropy": round(entropy, 4), "quadrants": quads,
            "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/table_scale_ratio_real_sensor.csv")
    ap.add_argument("--ratios", default=",".join(map(str, RATIOS)))
    args = ap.parse_args()
    ratios = [int(r) for r in args.ratios.split(",")]

    ref = load_texture()
    print(f"texture: {ref.shape}, dtype {ref.dtype}", flush=True)
    mov, M_gt = make_pair_real(ref)

    rows = []
    for r in ratios:
        row = run_one_ratio(ref, mov, M_gt, r)
        row.update({"texture_source": TEXTURE_SOURCE, "seed": SEED,
                    "matcher": "sift_cpu",
                    "interp_down": "INTER_AREA" if r > 1 else "n/a",
                    "interp_up": "INTER_CUBIC" if r > 1 else "n/a",
                    "truth": "real_texture_exact_7.3px_shift"})
        rows.append(row)
        print(f"ratio {r:>2}: {row['n_corr']:>5} corr, "
              f"{row['n_inliers']:>4} inliers, "
              f"inlier-RMSE {row['rmse_inlier_px']} px, "
              f"truth-RMSE {row['rmse_vs_truth_px']} px -> {row['verdict']}",
              flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = ["ratio", "n_corr", "n_inliers", "rmse_inlier_px",
            "rmse_vs_truth_px", "entropy", "quadrants", "verdict",
            "texture_source", "seed", "matcher", "interp_down",
            "interp_up", "truth"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
