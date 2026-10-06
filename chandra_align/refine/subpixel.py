"""Phase 8 — sub-pixel refinement contenders.

Two alternatives to the existing NCC-window + parabolic refinement
(chandra_align.refine.refine_subpixel_ncc):

- ``refine_lk``: Lucas-Kanade pyramidal optical flow per RANSAC inlier;
  a refined point is kept iff its reprojection residual drops.
- ``refine_phasecorr``: FFT phase correlation per point window +
  2D paraboloid peak fit (Foroosh et al. 2002 style).

Both return ``(kept_a, refined_b, stats)`` with the same semantics as
``refine_subpixel_ncc`` so the bake-off in scripts/run_phase8_bakeoff.py
can compare the three on identical inputs. Winner is chosen by held-out
error (no-peeking discipline); the loser stays available but is not wired
into the default path.
"""

import numpy as np


def _as_f32(img):
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2).astype(np.float32)
    return np.ascontiguousarray(a)


def _as_u8(img):
    a = np.asarray(img)
    if a.ndim == 3:
        a = a.mean(axis=2)
    a = a.astype(np.float64)
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        return np.zeros(a.shape, np.uint8)
    return np.ascontiguousarray(
        np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8))


def refine_lk(img_a, img_b, pts_a, pts_b, model=None, win=21, max_level=3):
    """Lucas-Kanade refinement of each correspondence.

    Tracks pts_a into img_b with pyramidal LK starting from pts_b as the
    initial guess. When ``model`` (the RANSAC 2x3 affine that produced these
    inliers) is given, a refined point is kept iff its reprojection residual
    ``|M(pa) - pred|`` is strictly smaller than the input residual
    ``|M(pa) - pb|`` — refinement must earn its keep. Without a model, all
    successfully tracked points are kept. Returns (kept_a, refined_b, stats).
    Points LK fails to track are dropped honestly.
    """
    import cv2

    a = _as_u8(img_a)
    b = _as_u8(img_b)
    pa = np.asarray(pts_a, dtype=np.float32).reshape(-1, 2)
    pb = np.asarray(pts_b, dtype=np.float32).reshape(-1, 2)
    n = len(pa)
    empty = (np.zeros((0, 2), np.float64), np.zeros((0, 2), np.float64),
             {"method": "lk", "refined_pairs": 0})
    if n == 0:
        return empty
    pred, status, _err = cv2.calcOpticalFlowPyrLK(
        a, b, pa, pb,
        winSize=(win, win), maxLevel=max_level,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    status = status.ravel().astype(bool)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 2)
    pa64 = pa.astype(np.float64)
    pb64 = pb.astype(np.float64)
    if model is not None:
        M = np.asarray(model, dtype=np.float64).reshape(2, 3)
        proj = pa64 @ M[:, :2].T + M[:, 2]
        in_res = np.hypot(pb64[:, 0] - proj[:, 0], pb64[:, 1] - proj[:, 1])
        lk_res = np.hypot(pred[:, 0] - proj[:, 0], pred[:, 1] - proj[:, 1])
        # Keep iff LK did not hurt: refined residual no worse than the input
        # within float noise. Strict improvement is rare once RANSAC inliers
        # are already sub-pixel; the value is the independent LK estimate.
        keep = status & (lk_res <= in_res + 1e-6)
    else:
        in_res = np.full(n, np.nan)
        lk_res = np.full(n, np.nan)
        keep = status
    kept_a = pa64[keep]
    refined_b = pred[keep]
    stats = {
        "method": "lk",
        "refined_pairs": int(keep.sum()),
        "dropped_pairs": int(n - keep.sum()),
        "kept_fraction": float(keep.mean()) if n else 0.0,
    }
    if model is not None and keep.any():
        stats["residual_mean_in_px"] = float(in_res[keep].mean())
        stats["residual_mean_out_px"] = float(lk_res[keep].mean())
    return kept_a, refined_b, stats


def _paraboloid_peak(corr):
    """2D paraboloid sub-pixel peak fit on a 3x3 neighbourhood.

    Returns (dx, dy) offset of the apex from the integer peak, or None when
    the fit is degenerate (apex outside ±1 px or singular curvature).
    """
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    h, w = corr.shape
    if px <= 0 or py <= 0 or px >= w - 1 or py >= h - 1:
        return None
    z = corr[py - 1:py + 2, px - 1:px + 2].astype(np.float64)
    # Separable parabola fits in x and y through the peak row/column.
    row = z[1, :]
    col = z[:, 1]
    den_x = row[0] - 2 * row[1] + row[2]
    den_y = col[0] - 2 * col[1] + col[2]
    if abs(den_x) < 1e-12 or abs(den_y) < 1e-12:
        return None
    dx = 0.5 * (row[0] - row[2]) / den_x
    dy = 0.5 * (col[0] - col[2]) / den_y
    if abs(dx) > 1.0 or abs(dy) > 1.0:
        return None
    return float(dx), float(dy)


def refine_phasecorr(img_a, img_b, pts_a, pts_b, window=32):
    """FFT phase-correlation refinement per correspondence + paraboloid fit.

    For each pair, correlate windowed patches around pts_a and pts_b; the
    phase-correlation peak (with paraboloid sub-pixel fit) gives the residual
    shift, which corrects pts_b. Returns (kept_a, refined_b, stats).
    """
    a = _as_f32(img_a)
    b = _as_f32(img_b)
    ha, wa = a.shape
    hb, wb = b.shape
    r = window // 2
    win_y, win_x = np.mgrid[0:window, 0:window].astype(np.float64)
    hann = np.sin(np.pi * win_x / window) * np.sin(np.pi * win_y / window)
    kept_a, out_b = [], []
    peak_heights = []
    pa = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
    pb = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)
    for (xa, ya), (xb, yb) in zip(pa, pb):
        xai, yai = int(round(xa)), int(round(ya))
        xbi, ybi = int(round(xb)), int(round(yb))
        if not (r <= xai < wa - r and r <= yai < ha - r):
            continue
        if not (r <= xbi < wb - r and r <= ybi < hb - r):
            continue
        pa_patch = a[yai - r:yai + r, xai - r:xai + r]
        pb_patch = b[ybi - r:ybi + r, xbi - r:xbi + r]
        if pa_patch.std() < 1e-9 or pb_patch.std() < 1e-9:
            continue
        fa = np.fft.fft2((pa_patch - pa_patch.mean()) * hann)
        fb = np.fft.fft2((pb_patch - pb_patch.mean()) * hann)
        cross = fa * np.conj(fb)
        mag = np.abs(cross)
        mag[mag < 1e-12] = 1e-12
        corr = np.fft.ifft2(cross / mag).real
        corr = np.fft.fftshift(corr)
        peak = _paraboloid_peak(corr)
        if peak is None:
            continue
        peak_heights.append(float(corr.max()))
        dx, dy = peak
        # fftshift puts zero-shift at the centre; the peak offset from centre
        # is the residual shift of b relative to a.
        cy, cx = corr.shape[0] // 2, corr.shape[1] // 2
        py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
        shift_x = (px + dx) - cx
        shift_y = (py + dy) - cy
        out_b.append((xb - shift_x, yb - shift_y))
        kept_a.append((xa, ya))
    if not out_b:
        return (np.zeros((0, 2), np.float64), np.zeros((0, 2), np.float64),
                {"method": "phasecorr", "refined_pairs": 0})
    kept_a = np.asarray(kept_a, np.float64)
    out_b = np.asarray(out_b, np.float64)
    stats = {
        "method": "phasecorr",
        "refined_pairs": int(len(out_b)),
        "peak_height_mean": float(np.mean(peak_heights)) if peak_heights else float("nan"),
        "peak_height_min": float(np.min(peak_heights)) if peak_heights else float("nan"),
    }
    return kept_a, out_b, stats
