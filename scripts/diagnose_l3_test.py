"""A22 decisive probe: if the hook used the NON-CLIQUE consensus (M2) instead
of the degenerate clique, would the stage apply on tmc2?

For each tmc2 pair: match -> hook RANSAC (clique) -> exclude clique ->
RANSAC on rest (M2) -> apply_deform_field_stage(ps2[inl2], pr2[inl2], M2).
Report: stage applied?, lambda, held-out vs affine, folding check.
This is the falsification test for lever L3 (clique-robust hook seeding).
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


def load(pair):
    ref = cv2.imread(os.path.join(CROP, "%s_reference.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(os.path.join(CROP, "%s_source.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    return ref, sec


for pair in ["tmc2_01", "tmc2_02", "tmc2_03", "tmc2_04", "tmc2_05", "tmc2_06"]:
    ref, sec = load(pair)
    pts_ref, pts_sec, _, _ = app.match_pair_hf(ref, sec, 3.0)
    pr = np.asarray(pts_ref, dtype=np.float64)
    ps = np.asarray(pts_sec, dtype=np.float64)
    cv2.setRNGSeed(7)
    Mh, inl = app._estimate_partial_affine_with_threshold(ps, pr, 3.0)
    inl = np.asarray(inl).astype(bool)
    rest = ~inl
    cv2.setRNGSeed(7)
    M2, inl2 = app._estimate_partial_affine_with_threshold(ps[rest], pr[rest], 3.0)
    if M2 is None:
        print("%s: M2 failed" % pair)
        continue
    inl2 = np.asarray(inl2).astype(bool)
    # Transform sanity of M2.
    A = M2[:, :2]
    scale = float(np.sqrt(abs(np.linalg.det(A))))
    rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    # Stage on the non-clique consensus.
    info = dfmod.apply_deform_field_stage(ps[rest][inl2], pr[rest][inl2], M2, seed=7)
    sweep = info.get("lambda_sweep") or []
    best = min(sweep, key=lambda s: s[1]) if sweep else (None, None)
    print("%s: n2=%d scale=%.4f rot=%+.2f | stage applied=%s reason=%s "
          "lambda=%s ho_field=%.3f ho_aff=%.3f minJac=%s" % (
              pair, int(inl2.sum()), scale, rot, info["applied"],
              info.get("reason"), info.get("lambda_chosen"), best[1],
              info.get("heldout_rmse_affine_px"),
              info.get("min_jacobian_det")))
