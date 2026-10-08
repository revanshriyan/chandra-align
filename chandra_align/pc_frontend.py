"""Opt-in phase-congruency (PC) matcher front-end (no training).

Uses the Phase-14 phase-congruency arm: keypoints are detected on the
phase-congruency map (Shi-Tomasi) and SIFT descriptors are computed on
the PC image itself, which carries no polarity. A crater rim looks the
same whether the sun lights it from the left or the right, so this arm
is complementary to SIFT: SIFT wins at small sun-angle differences, PC
wins at near-opposite sun (tested on the 96-case sun-angle sweep: 10
DEGENERATE -> COARSE conversions at azimuth 165-195 deg, 0.068px truth
RMSE at 180 deg where SIFT gets 0 inliers).

Additive and opt-in via CHANDRA_PC=1. With the flag unset, this module
is never imported and the pipeline is bit-identical.

Fail-closed: any error returns (None, None) and the caller falls back
to the default matcher.
"""

import os

import numpy as np

_FLAG = "CHANDRA_PC"
_TRUE = frozenset({"1", "true", "yes", "on"})


def pc_enabled():
    """True iff the operator opted in via CHANDRA_PC=1."""
    return os.environ.get(_FLAG, "").strip().lower() in _TRUE


def pc_available():
    """True iff the PC arm and its deps import (pure numpy/cv2, no weights)."""
    try:
        import cv2  # noqa: F401
        # Verify SIFT is actually present in this OpenCV build.
        if not hasattr(cv2, "SIFT_create"):
            return False
        from chandra_align.xmodal import pc_arm  # noqa: F401
        return True
    except ImportError:
        return False


def match_pair_pc(img1, img2):
    """Match two grayscale images through phase-congruency maps.

    Returns (pts1, pts2) as float32 (N,2) arrays in each image's pixel
    coordinates, or (None, None) on failure or when fewer than 3
    correspondences survive.
    """
    try:
        import cv2

        a = np.asarray(img1)
        b = np.asarray(img2)
        if a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        if b.ndim == 3:
            b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        if a.size == 0 or b.size == 0:
            return None, None

        from chandra_align.xmodal.pc_arm import match as pc_match

        pa, pb = pc_match(a, b)
        pa = np.asarray(pa, dtype=np.float64).reshape(-1, 2)
        pb = np.asarray(pb, dtype=np.float64).reshape(-1, 2)
        if len(pa) < 3 or len(pb) < 3 or not (
            np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))
        ):
            return None, None
        return pa.astype(np.float32), pb.astype(np.float32)
    except Exception:
        return None, None
