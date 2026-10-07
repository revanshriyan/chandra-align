"""A22 probe: raw-match quadrant distribution for ohrc_04/05/06.

Calls app.match_pair_hf (the exact matcher entry the pipeline uses) and
reports per-quadrant counts of raw matches vs keypoints detected, to
separate DETECTION failure (no keypoints in Q1/Q2) from CONSISTENCY failure
(keypoints exist but nothing RANSAC-consistent).
"""
import os
import sys

os.environ["CHANDRA_DEFORM_FIELD"] = "1"
sys.path.insert(0, "/tmp/stubs")
sys.path.insert(0, os.path.expanduser("~/workspace/chandra-align"))

import cv2
import numpy as np

import app

CROP = os.path.expanduser("~/workspace/chandra-align/data/benchmark_crops")


def qcounts(pts, shape):
    h, w = shape[:2]
    q = np.zeros(4, dtype=int)
    for x, y in np.asarray(pts).reshape(-1, 2):
        qi = (0 if y < h / 2 else 2) + (0 if x < w / 2 else 1)
        q[qi] += 1
    return q.tolist()


for pair in ["ohrc_04", "ohrc_05", "ohrc_06"]:
    ref = cv2.imread(os.path.join(CROP, "%s_reference.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(os.path.join(CROP, "%s_source.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    pts_ref, pts_sec, engine, diag = app.match_pair_hf(ref, sec, 3.0)
    print("%s: engine=%s" % (pair, engine))
    print("  raw matches=%d ref_quads=%s sec_quads=%s" % (
        len(pts_ref), qcounts(pts_ref, ref.shape), qcounts(pts_sec, sec.shape)))
    # Hook RANSAC on the raw matches (same call the pipeline hook makes).
    cv2.setRNGSeed(7)
    Mh, inl = app._estimate_partial_affine_with_threshold(pts_sec, pts_ref, 3.0)
    n_in = int(np.asarray(inl).sum()) if inl is not None else 0
    print("  hook RANSAC: M=%s inliers=%d quads=%s" % (
        "ok" if Mh is not None else None, n_in,
        qcounts(np.asarray(pts_ref)[np.asarray(inl).astype(bool)], ref.shape)
        if n_in else None))
    # Keypoint detection counts per quadrant (SIFT path): use the pipeline's
    # own detector via a direct SIFT call for diagnosis.
    sift = cv2.SIFT_create(nfeatures=10000)
    for name, img in (("ref", ref), ("sec", sec)):
        kps = sift.detect(img, None)
        print("  SIFT %s: %d keypoints quads=%s" % (
            name, len(kps), qcounts([k.pt for k in kps], img.shape)))
