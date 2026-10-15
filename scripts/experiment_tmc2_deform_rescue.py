"""Task 2: deform-field rescue experiment on TMC-2 fore/nadir vs HUMAN landmarks.

Question: can the existing deform-field machinery (apply_deform_field_stage
/ L1 maybe_rescue_and_refit) beat the 28.03 px global-affine baseline when
evaluated against the 20 human-verified landmarks (which never touch the fit)?

Protocol (clean experiment, may be negative):
1. Read the same audited 2048x2048 windows as the landmark rerun
   (fore col_off=361 row_off=93328; nadir col_off=416 row_off=101611).
2. Raw SIFT correspondences via the production recipe
   (app._match_sift_ransac, nfeatures=10000, 3 Lowe ratios).
3. Hook RANSAC (cv2.setRNGSeed(7), threshold 3.0 px) -> M_hook.
4. apply_deform_field_stage on hook inliers (lambda grid + internal
   held-out guardrail, folding fuse).
5. Evaluate the field-corrected transform on the 20 HUMAN landmarks:
   pred_ref = M_hook @ sec_landmark + d(sec_landmark); RMSE vs ref_landmark.
6. Baseline: the saved 13-inlier global affine on the same landmarks
   (expected ~28.03 px).
7. Also attempt maybe_rescue_and_refit (L1) and evaluate its refit field
   on the landmarks.

Frozen gates are NOT changed. If the field does not beat 28 px, this is
reported as a clean negative.

Output: results/table_tmc2_deform_rescue_2026-10-15.csv
"""

import csv
import os
import sys
import time

sys.path.insert(0, "/home/hatch/workspace/landmark_run")
import landmark_bootstrap  # noqa: F401  (paths, rasterio, UI mocks)

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

REPO = "/home/hatch/workspace/chandra-align"
QA = "/home/hatch/workspace/qa_lunar/pradan"
sys.path.insert(0, REPO)
os.chdir(REPO)

import app  # noqa: E402
from chandra_align.deform_field import (  # noqa: E402
    apply_deform_field_stage,
    maybe_rescue_and_refit,
    _eval_residual_field,
    _check_export_field,
)

TMC_FORE = os.path.join(
    QA, "ch2_tmc_ncf_20231101T0125121344_d_img_d18/data/calibrated/20231101/"
    "ch2_tmc_ncf_20231101T0125121344_d_img_d18.img")
TMC_NADIR = os.path.join(
    QA, "ch2_tmc_ncn_20231101T0125121377_d_img_d18/data/calibrated/20231101/"
    "ch2_tmc_ncn_20231101T0125121377_d_img_d18.img")
FORE_WIN = Window(col_off=361, row_off=93328, width=2048, height=2048)
NADIR_WIN = Window(col_off=416, row_off=101611, width=2048, height=2048)
GT = os.path.join(REPO, "data/ground_truth/gt_tmc2_fore_nadir_human_verified.csv")
BASELINE_NPZ = "/home/hatch/workspace/landmark_run/fit_TMC2_fore_nadir_sift.npz"
OUT = os.path.join(REPO, "results/table_tmc2_deform_rescue_2026-10-15.csv")


def read_window(path, window):
    label = path[:-4] + ".xml"
    with rasterio.open(label) as src:
        raw = src.read(1, window=window, masked=True)
        raw = np.asarray(raw.astype(np.float32).filled(np.nan))
    return app.load_lunar_raster(raw, max_dimension=4096)


def load_landmarks():
    xs, ys, xr, yr = [], [], [], []
    with open(GT) as f:
        for row in csv.DictReader(f):
            xs.append(float(row["x_src"]))
            ys.append(float(row["y_src"]))
            xr.append(float(row["x_ref"]))
            yr.append(float(row["y_ref"]))
    return (np.array(xs), np.array(ys)), (np.array(xr), np.array(yr))


def landmark_rmse(M, field, sec_xy, ref_xy):
    """RMSE of (affine + TPS field) on human landmarks. Landmarks never fit."""
    ps = np.column_stack(sec_xy)
    pr = np.column_stack(ref_xy)
    pred = ps @ M[:, :2].T + M[:, 2]
    if field is not None:
        try:
            pred = pred + _eval_residual_field(_check_export_field(field), ps)
        except Exception as e:
            return None, "field_eval_failed:%s" % str(e)[:60]
    resid = np.linalg.norm(pred - pr, axis=1)
    return float(np.sqrt((resid ** 2).mean())), None


def main():
    t0 = time.time()
    print("reading windows...", flush=True)
    fore = read_window(TMC_FORE, FORE_WIN)
    nadir = read_window(TMC_NADIR, NADIR_WIN)
    print("fore", fore.shape, "nadir", nadir.shape, flush=True)

    print("SIFT raw matches...", flush=True)
    pts_ref_raw, pts_sec_raw, n_best = app._match_sift_ransac(fore, nadir, 3.0)
    pts_ref_raw = np.asarray(pts_ref_raw, np.float64)
    pts_sec_raw = np.asarray(pts_sec_raw, np.float64)
    print("raw correspondences:", len(pts_ref_raw), flush=True)

    cv2.setRNGSeed(7)
    M_hook, hook_mask = app._estimate_partial_affine_with_threshold(
        pts_sec_raw, pts_ref_raw, 3.0)
    hook_mask = np.asarray(hook_mask, bool).reshape(-1)
    n_hook = int(hook_mask.sum())
    print("hook inliers:", n_hook, flush=True)

    sec_xy, ref_xy = load_landmarks()

    # Baseline: saved global affine on landmarks.
    b = np.load(BASELINE_NPZ)
    M_base = np.asarray(b["affine_matrix"], np.float64)
    base_rmse, _ = landmark_rmse(M_base, None, sec_xy, ref_xy)
    print("baseline affine landmark RMSE: %.4f px" % base_rmse, flush=True)

    rows = [{
        "arm": "global_affine_baseline", "n_fit_points": 13,
        "landmark_rmse_px": round(base_rmse, 4),
        "internal_heldout_rmse_px": 1.7654,
        "lambda": "", "note": "saved 13-inlier fit; 28.03px baseline",
    }]

    # Arm 1: deform-field stage on hook inliers, evaluated on landmarks.
    stage = apply_deform_field_stage(
        pts_sec_raw[hook_mask], pts_ref_raw[hook_mask], M_hook, seed=7)
    print("stage applied:", stage.get("applied"),
          "reason:", stage.get("reason"),
          "heldout:", stage.get("heldout_rmse_px"),
          "lambda:", stage.get("lambda_chosen"), flush=True)
    if stage.get("applied"):
        rmse, err = landmark_rmse(M_hook, stage.get("field"), sec_xy, ref_xy)
        rows.append({
            "arm": "deform_field_stage", "n_fit_points": n_hook,
            "landmark_rmse_px": round(rmse, 4) if rmse else None,
            "internal_heldout_rmse_px": stage.get("heldout_rmse_px"),
            "lambda": stage.get("lambda_chosen"),
            "note": err or "field+affine on 20 human landmarks",
        })
        print("deform field landmark RMSE: %s px" % rmse, flush=True)
    else:
        rows.append({
            "arm": "deform_field_stage", "n_fit_points": n_hook,
            "landmark_rmse_px": None,
            "internal_heldout_rmse_px": None, "lambda": "",
            "note": "stage declined: %s" % stage.get("reason"),
        })

    # Arm 2: L1 rescue path.
    refit, rescue = maybe_rescue_and_refit(
        stage, pts_sec_raw, pts_ref_raw, hook_mask, M_hook, fore.shape[:2],
        seed=7)
    print("rescue:", rescue.get("rescued"), "reason:", rescue.get("reason"),
          flush=True)
    if rescue.get("rescued") and refit.get("applied"):
        rmse, err = landmark_rmse(
            np.asarray(refit.get("M_hook", M_hook), np.float64),
            refit.get("field"), sec_xy, ref_xy)
        rows.append({
            "arm": "l1_rescue_refit", "n_fit_points": rescue.get("n_admitted"),
            "landmark_rmse_px": round(rmse, 4) if rmse else None,
            "internal_heldout_rmse_px": refit.get("heldout_rmse_px"),
            "lambda": refit.get("lambda_chosen"),
            "note": err or "rescue refit field on 20 human landmarks",
        })
        print("rescue refit landmark RMSE: %s px" % rmse, flush=True)
    else:
        rows.append({
            "arm": "l1_rescue_refit", "n_fit_points": rescue.get("n_admitted", 0),
            "landmark_rmse_px": None, "internal_heldout_rmse_px": None,
            "lambda": "",
            "note": "rescue not triggered/applied: %s" % rescue.get("reason"),
        })

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT, "elapsed %.1fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
