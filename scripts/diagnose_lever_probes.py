"""A22 probes: (1) tmc2 clique-excluded RANSAC -- is there a real consensus
hiding behind the degenerate clique? (2) ohrc_05/06 full lambda sweep --
is lambda=0.1 at the grid edge (would extending the grid help)?
"""
import os
import sys

os.environ["CHANDRA_DEFORM_FIELD"] = "1"
sys.path.insert(0, "/tmp/stubs")
sys.path.insert(0, os.path.expanduser("~/workspace/chandra-align"))

import cv2
import numpy as np

import app
import chandra_align.deform_field as dfmod

CROP = os.path.expanduser("~/workspace/chandra-align/data/benchmark_crops")


def qcounts(pts, shape):
    h, w = shape[:2]
    q = np.zeros(4, dtype=int)
    for x, y in np.asarray(pts).reshape(-1, 2):
        qi = (0 if y < h / 2 else 2) + (0 if x < w / 2 else 1)
        q[qi] += 1
    return q.tolist()


def load(pair):
    ref = cv2.imread(os.path.join(CROP, "%s_reference.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(os.path.join(CROP, "%s_source.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    return ref, sec


print("=== (1) tmc2: RANSAC excluding the hook clique ===")
for pair in ["tmc2_01", "tmc2_02", "tmc2_03", "tmc2_04", "tmc2_05", "tmc2_06"]:
    ref, sec = load(pair)
    pts_ref, pts_sec, _, _ = app.match_pair_hf(ref, sec, 3.0)
    pr = np.asarray(pts_ref, dtype=np.float64)
    ps = np.asarray(pts_sec, dtype=np.float64)
    cv2.setRNGSeed(7)
    Mh, inl = app._estimate_partial_affine_with_threshold(ps, pr, 3.0)
    inl = np.asarray(inl).astype(bool)
    # Exclude the clique: drop hook inliers, RANSAC the rest at 3px and 6px.
    rest = ~inl
    for thr in (3.0, 6.0):
        cv2.setRNGSeed(7)
        M2, inl2 = app._estimate_partial_affine_with_threshold(
            ps[rest], pr[rest], thr)
        n2 = int(np.asarray(inl2).sum()) if inl2 is not None else 0
        qc = qcounts(pr[rest][np.asarray(inl2).astype(bool)], ref.shape) if n2 else None
        print("  %s thr=%.0f: non-clique inliers=%d quads=%s M=%s" % (
            pair, thr, n2, qc, "ok" if M2 is not None else None))

print("\n=== (2) ohrc_05/06: full lambda sweep ===")
for pair in ["ohrc_05", "ohrc_06", "ohrc_04"]:
    ref, sec = load(pair)
    pts_ref, pts_sec, _, _ = app.match_pair_hf(ref, sec, 3.0)
    pr = np.asarray(pts_ref, dtype=np.float64)
    ps = np.asarray(pts_sec, dtype=np.float64)
    cv2.setRNGSeed(7)
    Mh, inl = app._estimate_partial_affine_with_threshold(ps, pr, 3.0)
    inl = np.asarray(inl).astype(bool)
    info = dfmod.apply_deform_field_stage(ps[inl], pr[inl], Mh, seed=7)
    sweep = info.get("lambda_sweep") or []
    print("  %s: applied=%s aff_ho=%.4f" % (pair, info["applied"], info.get("heldout_rmse_affine_px")))
    for lam, chk in sweep:
        print("    lambda=%6s check=%.4f %s" % (lam, chk, "<-- best" if lam == info.get("lambda_chosen") else ""))
