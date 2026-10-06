"""Phase-congruency matcher arm (Phase 14).

Detects and describes on phase-congruency maps instead of raw intensity:
keypoints come from structure (Shi-Tomasi on the PC map) and SIFT
descriptors are computed on the PC image itself, which carries no polarity.
A crater rim looks the same to this arm whether the sensor saw it bright
or dark.

match(a, b) -> (pa, pb): float64 (N,2) correspondence arrays, possibly
empty. Never raises on degenerate input; empty means "no evidence".
"""

import cv2
import numpy as np

from .phase_congruency import phase_congruency


def _pc_uint8(img, nscale=4, norient=6):
    """Phase-congruency map normalized to uint8 via robust percentiles."""
    pc, _ = phase_congruency(img, nscale=nscale, norient=norient)
    lo, hi = np.percentile(pc, [1, 99.5])
    u8 = np.clip((pc - lo) * (255.0 / (hi - lo + 1e-9)), 0, 255)
    return u8.astype(np.uint8)


def _detect_on_pc(pcu8, max_corners=8000):
    """Shi-Tomasi corners on the PC map; returns (N,2) float64 or empty."""
    corners = cv2.goodFeaturesToTrack(
        pcu8, maxCorners=max_corners, qualityLevel=0.01,
        minDistance=5, useHarrisDetector=False)
    if corners is None:
        return np.zeros((0, 2), np.float64)
    return corners.reshape(-1, 2).astype(np.float64)


def _describe(pcu8, pts):
    """SIFT descriptors at given points on the PC image."""
    if len(pts) == 0:
        return np.zeros((0, 2), np.float64), None
    kps = [cv2.KeyPoint(float(x), float(y), 16.0) for x, y in pts]
    sift = cv2.SIFT_create()
    kps, desc = sift.compute(pcu8, kps)
    if desc is None or len(kps) == 0:
        return np.zeros((0, 2), np.float64), None
    return np.array([k.pt for k in kps], np.float64), desc


def match(a, b, lowe_ratio=0.75, nscale=4, norient=6):
    """Match images a and b through phase-congruency maps.

    Returns (pa, pb): Lowe-filtered correspondences as float64 (N,2)
    arrays in each image's pixel coordinates. Empty when there is no
    matchable structure or no descriptor survived.
    """
    try:
        pau8 = _pc_uint8(a, nscale=nscale, norient=norient)
        pbu8 = _pc_uint8(b, nscale=nscale, norient=norient)
    except Exception:
        return np.zeros((0, 2), np.float64), np.zeros((0, 2), np.float64)
    pa_pts = _detect_on_pc(pau8)
    pb_pts = _detect_on_pc(pbu8)
    ka, da = _describe(pau8, pa_pts)
    kb, db = _describe(pbu8, pb_pts)
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return np.zeros((0, 2), np.float64), np.zeros((0, 2), np.float64)
    knn = cv2.BFMatcher().knnMatch(da, db, k=2)
    good_a, good_b = [], []
    for m_n in knn:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < lowe_ratio * n.distance:
            good_a.append(ka[m.queryIdx])
            good_b.append(kb[m.trainIdx])
    pa = np.asarray(good_a, np.float64).reshape(-1, 2)
    pb = np.asarray(good_b, np.float64).reshape(-1, 2)
    return pa, pb
