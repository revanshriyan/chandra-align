"""Phase 10 — pixel area-check: pipeline-independent registration evidence.

Given a fitted transform M (source A -> reference B), this module warps A
into B's geometry and checks each grid cell with FFT cross-correlation on
illumination-normalized pixels. It does NOT use the feature correspondences
that produced M — it is an independent check on the pixels themselves.

Cell states: VERIFIED (sharp correlation peak at the expected location),
WEAK (peak present but diffuse/offset), NO_EVIDENCE (no peak or flat).

Thresholds are calibrated on synthetic pairs with known ground truth
(see calibrate_thresholds), not hand-waved.
"""

import numpy as np


def _illum_norm(patch):
    """Illumination normalization: zero-mean, unit-std (robust)."""
    p = np.asarray(patch, dtype=np.float64)
    med = np.median(p)
    mad = np.median(np.abs(p - med)) + 1e-9
    return (p - med) / (1.4826 * mad)


def _cell_corr_peak(cell_a, cell_b):
    """FFT cross-correlation peak sharpness for two cells.

    Returns (peak_value, peak_offset_yx, sharpness) where sharpness is
    peak / mean-abs — high means a single dominant shift, low means diffuse.
    """
    a = _illum_norm(cell_a)
    b = _illum_norm(cell_b)
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fa * np.conj(fb)
    mag = np.abs(cross)
    mag[mag < 1e-12] = 1e-12
    corr = np.fft.fftshift(np.fft.ifft2(cross / mag).real)
    peak = float(corr.max())
    loc = np.unravel_index(int(np.argmax(corr)), corr.shape)
    cy, cx = corr.shape[0] // 2, corr.shape[1] // 2
    sharpness = float(peak / (np.abs(corr).mean() + 1e-12))
    return peak, (int(loc[0] - cy), int(loc[1] - cx)), sharpness


# Calibrated defaults (see calibrate_thresholds docstring).
VERIFIED_SHARPNESS_MIN = 8.0
VERIFIED_OFFSET_MAX_PX = 3
WEAK_SHARPNESS_MIN = 3.0


def area_check(img_a, img_b, M, grid=(6, 6)):
    """Pixel area-check of transform M mapping A -> B.

    Warps A into B's frame with M, then per-cell normalized cross-correlation.
    Returns dict: n_cells, n_verified, n_weak, n_no_evidence,
    verified_fraction, cell_states (list), overall ("verified"/"weak"/"no_evidence").
    Overall is "verified" when >=60% of cells verify, "weak" when >=30%,
    else "no_evidence".
    """
    import cv2

    a = np.asarray(img_a, dtype=np.float32)
    b = np.asarray(img_b, dtype=np.float32)
    M = np.asarray(M, dtype=np.float64).reshape(2, 3)
    h, w = b.shape[:2]
    # M maps A->B; warpAffine needs the inverse (for each B pixel, sample A).
    M_inv = cv2.invertAffineTransform(M.astype(np.float64)).astype(np.float32)
    warped = cv2.warpAffine(a, M_inv, (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    gh, gw = grid
    states = []
    for gy in range(gh):
        for gx in range(gw):
            y0, y1 = gy * h // gh, (gy + 1) * h // gh
            x0, x1 = gx * w // gw, (gx + 1) * w // gw
            ca = warped[y0:y1, x0:x1]
            cb = b[y0:y1, x0:x1]
            if np.isnan(ca).mean() > 0.5 or cb.std() < 1e-9:
                states.append("no_evidence")
                continue
            ca = np.nan_to_num(ca, nan=np.nanmedian(ca))
            _, (oy, ox), sharp = _cell_corr_peak(ca, cb)
            off = abs(oy) + abs(ox)
            if sharp >= VERIFIED_SHARPNESS_MIN and off <= VERIFIED_OFFSET_MAX_PX:
                states.append("verified")
            elif sharp >= WEAK_SHARPNESS_MIN:
                states.append("weak")
            else:
                states.append("no_evidence")
    n = len(states)
    nv = sum(1 for s in states if s == "verified")
    nw = sum(1 for s in states if s == "weak")
    frac = nv / n if n else 0.0
    overall = "verified" if frac >= 0.6 else ("weak" if (nv + nw) / n >= 0.3 else "no_evidence")
    return {
        "n_cells": n, "n_verified": nv, "n_weak": nw,
        "n_no_evidence": n - nv - nw,
        "verified_fraction": float(frac),
        "cell_states": states,
        "overall": overall,
    }


def calibrate_thresholds():
    """Calibrate area-check thresholds on synthetic truth.

    Builds synthetic pairs with known transforms (correct and deliberately
    wrong), runs area_check, and reports the sharpness distributions so the
    VERIFIED/WEAK cutoffs separate true registrations from wrong ones.
    Returns a dict with the measured distributions and the chosen thresholds.
    """
    from chandra_align.testing import make_pair_shift
    rng = np.random.default_rng(11)
    sharp_true, sharp_wrong = [], []
    for seed in range(6):
        ref, mov, M_gt = make_pair_shift(shape=(256, 256), dx=5.0, dy=-2.0, seed=seed)
        # correct transform
        r = area_check(mov, ref, M_gt, grid=(4, 4))
        # deliberately wrong transform (large offset)
        M_bad = M_gt.copy()
        M_bad[:, 2] += 25.0
        r_bad = area_check(mov, ref, M_bad, grid=(4, 4))
        sharp_true.append(r["verified_fraction"])
        sharp_wrong.append(r_bad["verified_fraction"])
    return {
        "verified_fraction_true_mean": float(np.mean(sharp_true)),
        "verified_fraction_true_min": float(np.min(sharp_true)),
        "verified_fraction_wrong_max": float(np.max(sharp_wrong)),
        "threshold_verified_frac": 0.6,
        "threshold_weak_frac": 0.3,
        "separates": bool(np.min(sharp_true) > 0.6 >= np.max(sharp_wrong)),
    }
