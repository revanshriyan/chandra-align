"""A22 probe: can the fitted field rescue Q1/Q2 raw matches on ohrc_04?

Fit: hook RANSAC inliers -> M_hook -> apply_deform_field_stage (as the
pipeline does). Then score ALL raw matches under the field model:
  field_resid = |p_ref - (M_hook @ p_src + d(p_src))|
Count per-quadrant matches with field_resid < 3px (RANSAC threshold) and
< 1px. If hundreds of Q1/Q2 matches are field-consistent, an iterative
field-guided matching lever exists. If ~0, the Q1/Q2 matches are mismatches
and there is no lever.
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


for pair in ["ohrc_04", "ohrc_05", "ohrc_06"]:
    ref = cv2.imread(os.path.join(CROP, "%s_reference.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(os.path.join(CROP, "%s_source.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    pts_ref, pts_sec, _, _ = app.match_pair_hf(ref, sec, 3.0)
    pr = np.asarray(pts_ref, dtype=np.float64)
    ps = np.asarray(pts_sec, dtype=np.float64)
    cv2.setRNGSeed(7)
    Mh, inl = app._estimate_partial_affine_with_threshold(ps, pr, 3.0)
    inl = np.asarray(inl).astype(bool)
    info = dfmod.apply_deform_field_stage(ps[inl], pr[inl], Mh, seed=7)
    print("%s: stage applied=%s reason=%s" % (pair, info["applied"], info.get("reason")))
    if not info["applied"]:
        continue
    field = info["field"]
    d_all = dfmod._eval_residual_field(field, ps)
    pred = (np.asarray(Mh, dtype=np.float64)[:, :2] @ ps.T).T + np.asarray(Mh)[:, 2] + d_all
    fres = np.linalg.norm(pr - pred, axis=1)
    # Affine residuals for comparison.
    pred_a = (np.asarray(Mh, dtype=np.float64)[:, :2] @ ps.T).T + np.asarray(Mh)[:, 2]
    ares = np.linalg.norm(pr - pred_a, axis=1)
    h, w = ref.shape[:2]
    for name, r in (("affine", ares), ("field", fres)):
        for thr in (1.0, 3.0):
            m = r < thr
            print("  %s < %.0fpx: n=%d quads=%s" % (name, thr, int(m.sum()), qcounts(pr[m], ref.shape)))
