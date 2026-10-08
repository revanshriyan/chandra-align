"""A27: re-run all six TMC-2 pairs with the CORRECT pixel scale (5.0 m/px).

A26 proved every tmc2 DEGENERATE in the 12-pair benchmark was measured with
pixel_scale_m defaulting to 0.25 (the OHRC value), forcing a ~16x asymmetric
downsample of the reference and a many-to-one matching collapse. The deployed
UI passes pixel_scale_m=get_sensor_pixel_scale(reference_sensor) = 5.0.

This script runs all six tmc2 pairs through the true app.py::_align_core twice
(flag OFF / flag ON) with pixel_scale_m=5.0 passed explicitly, plus an ohrc_01
control with pixel_scale_m=0.25 explicitly (must be bit-identical to
results/table_quota_12pair.csv, proving scale-passing changes nothing).

Experiment only: no app.py / chandra_align changes. Deterministic (seed 7).
"""
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/tmp/stubs")
sys.path.insert(0, "/home/hatch/workspace/chandra-align")

import app  # noqa: E402
from app import apply_clahe, keypoint_starvation_guard  # noqa: E402
from chandra_align.preprocessing.multimodal import resize_to_common_ground_sample  # noqa: E402
from chandra_align.metrics.enhanced import get_sensor_pixel_scale  # noqa: E402

CROP = "/home/hatch/workspace/chandra-align/data/benchmark_crops"
PAIRS = ["tmc2_01", "tmc2_02", "tmc2_03", "tmc2_04", "tmc2_05", "tmc2_06"]
DESCRIPTIONS = {
    "tmc2_01": "south-polar highlands; large shadowed crater walls and rugged relief (visual only; no DEM label)",
    "tmc2_02": "south-polar highlands; cratered surface with broad shadowed relief (visual only)",
    "tmc2_03": "south-polar highlands; large shadowed crater and cratered surroundings (visual only)",
    "tmc2_04": "south-polar highlands; crater rim and rugged inter-crater relief (visual only)",
    "tmc2_05": "south-polar highlands; densely cratered inter-crater terrain (visual only)",
    "tmc2_06": "south-polar highlands; isolated crater with shadowed rim relief (visual only)",
}
REF_PROD = "ch2_tmc_ncf_20231101T0125121344_d_img_d18"
SEC_PROD = "ch2_tmc_ncn_20231101T0125121377_d_img_d18"


def run_pair(pair, pixel_scale_m, flag):
    """One true _align_core run. Returns (row dict, error or None)."""
    ref = cv2.imread(os.path.join(CROP, "%s_reference.png" % pair), cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(os.path.join(CROP, "%s_source.png" % pair), cv2.IMREAD_GRAYSCALE)
    if ref is None or sec is None:
        return None, "missing_image"
    old = os.environ.get("CHANDRA_DEFORM_FIELD")
    try:
        if flag:
            os.environ["CHANDRA_DEFORM_FIELD"] = "1"
        else:
            os.environ.pop("CHANDRA_DEFORM_FIELD", None)
        cv2.setRNGSeed(7)
        out = app._align_core(ref, sec, pixel_scale_m=pixel_scale_m,
                              reference_sensor_name="TMC-2")
    except Exception as exc:
        return None, "exception:%s:%s" % (type(exc).__name__, str(exc)[:120])
    finally:
        if old is None:
            os.environ.pop("CHANDRA_DEFORM_FIELD", None)
        else:
            os.environ["CHANDRA_DEFORM_FIELD"] = old
    jm = out.get("judge_metrics", {}) or {}
    df = jm.get("deform_field_stage", {}) or {}
    gm = jm.get("global_metrics", {}) or {}
    return {
        "pair": pair,
        "pixel_scale_m": pixel_scale_m,
        "flag": "on" if flag else "off",
        "status_code": out.get("status_code"),
        "n_inliers": out.get("inlier_cnt", gm.get("inlier_count")),
        "rmse_in_sample_px": jm.get("rmse_in_sample_px"),
        "rmse_gate_px": jm.get("rmse_gate_px"),
        "entropy": out.get("quadrant_spatial_entropy", gm.get("spatial_entropy_score")),
        "quadrants": jm.get("active_quadrants_count"),
        "quadrant_counts": jm.get("quadrant_counts"),
        "df_applied": df.get("applied"),
        "df_reason": df.get("reason"),
        "df_lambda": df.get("lambda_chosen"),
        "df_heldout_px": df.get("heldout_rmse_px"),
        "df_min_jac_det": df.get("min_jacobian_det"),
        "warp_export_kind": jm.get("warp_export_kind"),
    }, None


def quantify_collapse(pair, pixel_scale_m, flag):
    """Many-to-one stats on the match set exactly as the hook sees it."""
    old = os.environ.get("CHANDRA_DEFORM_FIELD")
    try:
        if flag:
            os.environ["CHANDRA_DEFORM_FIELD"] = "1"
        else:
            os.environ.pop("CHANDRA_DEFORM_FIELD", None)
        ref = cv2.imread(os.path.join(CROP, "%s_reference.png" % pair), cv2.IMREAD_GRAYSCALE)
        sec = cv2.imread(os.path.join(CROP, "%s_source.png" % pair), cv2.IMREAD_GRAYSCALE)
        ref_p = apply_clahe(ref, clip_limit=3.0, normalize_range=(0.0, 255.0))
        sec_p = apply_clahe(sec, clip_limit=3.0, normalize_range=(0.0, 255.0))
        ref_p, sec_p, _, _, _ = keypoint_starvation_guard(
            ref_p, sec_p, None, None, preprocessing_triggered=False,
            min_candidates=30, clahe_clip_limit=3.0)
        target = max(float(pixel_scale_m), get_sensor_pixel_scale("TMC-2"))
        mref, _ = resize_to_common_ground_sample(ref_p, float(pixel_scale_m), target, 128)
        msec, _ = resize_to_common_ground_sample(sec_p, get_sensor_pixel_scale("TMC-2"), target, 128)
        cv2.setRNGSeed(7)
        pts_ref, pts_sec, engine, _ = app.match_pair_hf(mref, msec, 3.0)
    finally:
        if old is None:
            os.environ.pop("CHANDRA_DEFORM_FIELD", None)
        else:
            os.environ["CHANDRA_DEFORM_FIELD"] = old
    n = len(pts_ref)
    if n:
        uniq, counts = np.unique(np.round(np.asarray(pts_ref), 1), axis=0, return_counts=True)
        order = np.argsort(-counts)
        max_hits = int(counts[order[0]])
        unique_ratio = float(len(uniq) / n)
    else:
        max_hits, unique_ratio = 0, 0.0
    return {
        "pair": pair,
        "flag": "on" if flag else "off",
        "match_ref_shape": "%dx%d" % (mref.shape[0], mref.shape[1]),
        "match_sec_shape": "%dx%d" % (msec.shape[0], msec.shape[1]),
        "n_matches": n,
        "max_hits_one_target": max_hits,
        "unique_target_ratio": round(unique_ratio, 4),
        "collapse_gone": (max_hits < 10 and unique_ratio > 0.5),
        "engine": engine,
    }


def main():
    rows, collapses = [], []
    for pair in PAIRS:
        for flag in (False, True):
            row, err = run_pair(pair, 5.0, flag)
            if err:
                row = {"pair": pair, "pixel_scale_m": 5.0,
                       "flag": "on" if flag else "off", "error": err}
            row["pair_description"] = DESCRIPTIONS[pair]
            row["reference_product"] = REF_PROD
            row["source_product"] = SEC_PROD
            rows.append(row)
            print("%s flag=%s -> %s" % (pair, "on" if flag else "off",
                                       row.get("status_code", row.get("error"))), flush=True)
            cq = quantify_collapse(pair, 5.0, flag)
            cq["pair_description"] = DESCRIPTIONS[pair]
            collapses.append(cq)
            print("   collapse: matches=%d max_hits=%d unique_ratio=%.3f gone=%s" % (
                cq["n_matches"], cq["max_hits_one_target"],
                cq["unique_target_ratio"], cq["collapse_gone"]), flush=True)

    # Control: ohrc_01 with pixel_scale_m=0.25 EXPLICITLY (not defaulted)
    ctrl = []
    for flag in (False, True):
        ref = cv2.imread(os.path.join(CROP, "ohrc_01_reference.png"), cv2.IMREAD_GRAYSCALE)
        sec = cv2.imread(os.path.join(CROP, "ohrc_01_source.png"), cv2.IMREAD_GRAYSCALE)
        old = os.environ.get("CHANDRA_DEFORM_FIELD")
        try:
            if flag:
                os.environ["CHANDRA_DEFORM_FIELD"] = "1"
            else:
                os.environ.pop("CHANDRA_DEFORM_FIELD", None)
            cv2.setRNGSeed(7)
            out = app._align_core(ref, sec, pixel_scale_m=0.25,
                                  reference_sensor_name="OHRC")
        finally:
            if old is None:
                os.environ.pop("CHANDRA_DEFORM_FIELD", None)
            else:
                os.environ["CHANDRA_DEFORM_FIELD"] = old
        jm = out.get("judge_metrics", {}) or {}
        ctrl.append({"flag": "on" if flag else "off",
                     "status_code": out.get("status_code"),
                     "n_inliers": jm.get("n_inliers"),
                     "rmse_gate_px": jm.get("rmse_gate_px")})
        print("CONTROL ohrc_01 flag=%s -> %s inliers=%s gate=%.4f" % (
            "on" if flag else "off", out.get("status_code"), jm.get("n_inliers"),
            float(jm.get("rmse_gate_px") or 0)), flush=True)

    # Determinism re-run: tmc2_01 flag on
    r1, _ = run_pair("tmc2_01", 5.0, True)
    r2, _ = run_pair("tmc2_01", 5.0, True)
    det = (r1.get("status_code") == r2.get("status_code")
           and r1.get("n_inliers") == r2.get("n_inliers")
           and abs(float(r1.get("rmse_gate_px") or 0) - float(r2.get("rmse_gate_px") or 0)) < 1e-9)
    print("determinism tmc2_01 flag-on re-run:", "IDENTICAL" if det else "MISMATCH", flush=True)

    outp = "/home/hatch/workspace/chandra-align/results/table_tmc2_correct_scale.csv"
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    outc = "/home/hatch/workspace/chandra-align/results/table_tmc2_collapse_check.csv"
    with open(outc, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(collapses[0].keys()))
        w.writeheader()
        w.writerows(collapses)
    print("wrote", outp, "and", outc)
    print("CONTROL:", ctrl)
    print("DETERMINISM:", det)


if __name__ == "__main__":
    main()
