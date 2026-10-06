"""SIFT+Lowe+RANSAC rescue arm feeding a smoothing TPS (Phase 9).

When the primary deep matcher starves on a cross-modal pair, this arm
tries the classical fallback: SIFT keypoints, Lowe ratio test, RANSAC
affine, then — only if the affine is geometrically sane — a smoothing
thin-plate spline on the inliers for local refinement.

Gates (fail closed, never tuned for optics):
  - >= 13 RANSAC inliers (RESCUE_INLIER_MIN), else RESCUE_FAILED.
  - Gate 3 transform conditioning on the RANSAC affine
    (chandra_align.metrics.conditioning.check_transform_conditioning),
    else RESCUE_FAILED.
  - The smoothing TPS is fit on inliers only, lmbda=0.05.

What it does NOT do: it does not accept a fit the gates reject, it does
not run before verification, and the TPS warp is a *candidate* — scoring
belongs to the caller on held-out data, not here.
"""

import cv2
import numpy as np

from chandra_align.metrics.conditioning import check_transform_conditioning
from chandra_align.xmodal.tps import fit_tps

RESCUE_INLIER_MIN = 13
RESCUE_TPS_LAMBDA = 0.05
LOWE_RATIO = 0.75


def _to_uint8(img):
    img = np.asarray(img)
    if img.dtype == np.uint8:
        gray = img
    else:
        gray = np.clip(img, 0, 255).astype(np.uint8)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    return np.ascontiguousarray(gray)


def _sift_pairs(img_a, img_b):
    """SIFT detect+describe on both images, Lowe-ratio matches.

    Returns (pa, pb): (M,2) corresponding points (a-space, b-space).
    """
    ga, gb = _to_uint8(img_a), _to_uint8(img_b)
    sift = cv2.SIFT_create()
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) == 0 or len(kb) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(da, db, k=2)
    pa, pb = [], []
    for m_n in knn:
        if len(m_n) != 2:
            continue
        m, n_ = m_n
        if m.distance < LOWE_RATIO * n_.distance:
            pa.append(ka[m.queryIdx].pt)
            pb.append(kb[m.trainIdx].pt)
    if not pa:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.asarray(pa, dtype=np.float64), np.asarray(pb, dtype=np.float64)


def sift_rescue_tps(img_a, img_b, init_pa, init_pb):
    """Attempt a SIFT-based rescue fit, TPS-refined, fully gated.

    Parameters
    ----------
    img_a, img_b : images (any numeric dtype; converted internally).
    init_pa, init_pb : (K,2) array-likes
        Prior correspondences (a-space, b-space), e.g. from an earlier
        matcher stage. They are *pooled with* the fresh SIFT matches
        before RANSAC — they do not bypass it.

    Returns
    -------
    dict with keys:
        'verdict'    'TPS_RESCUED' or 'RESCUE_FAILED'
        'n_inliers'  RANSAC inlier count on the pooled correspondences
        'tps'        fit_tps() dict (with 'warp' callable) or None
        'affine'     (2,3) RANSAC affine (a-space -> b-space) or None
        'gate3'      Gate 3 (ok, report) on the RANSAC affine
        'reason'     short machine-readable failure reason on RESCUE_FAILED
    """
    sift_pa, sift_pb = _sift_pairs(img_a, img_b)
    init_pa = np.asarray(init_pa, dtype=np.float64).reshape(-1, 2)
    init_pb = np.asarray(init_pb, dtype=np.float64).reshape(-1, 2)
    if init_pa.shape[0] != init_pb.shape[0]:
        raise ValueError("init_pa and init_pb must have the same row count")
    # Pool prior correspondences with fresh SIFT matches; RANSAC decides.
    pa = np.vstack([sift_pa, init_pa]) if len(init_pa) else sift_pa
    pb = np.vstack([sift_pb, init_pb]) if len(init_pb) else sift_pb

    out = {"verdict": "RESCUE_FAILED", "n_inliers": 0, "tps": None,
           "affine": None, "gate3": (False, {"ok": False, "error": "no fit"}),
           "reason": "insufficient_correspondences"}
    if len(pa) < RESCUE_INLIER_MIN:
        return out

    affine, mask = cv2.estimateAffinePartial2D(
        pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        confidence=0.99, maxIters=2000)
    if affine is None or mask is None:
        out["reason"] = "ransac_no_model"
        return out
    inl = mask.ravel().astype(bool)
    n_inliers = int(inl.sum())
    out["n_inliers"] = n_inliers
    if n_inliers < RESCUE_INLIER_MIN:
        out["reason"] = "inlier_floor"
        return out

    ok, report = check_transform_conditioning(affine)
    out["gate3"] = (ok, report)
    if not ok:
        out["reason"] = "gate3_conditioning"
        return out

    tps_fit = fit_tps(pa[inl], pb[inl], lmbda=RESCUE_TPS_LAMBDA)
    out.update({"verdict": "TPS_RESCUED", "tps": tps_fit,
                "affine": affine, "reason": "ok"})
    return out
