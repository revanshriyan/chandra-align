"""A26: diagnose the TMC-2 many-to-one matching collapse.

Reproduces the pipeline matching path on tmc2_01 under two configs:
  A (harness): pixel_scale_m=0.25 (what the benchmark scripts pass)
  B (UI-correct): pixel_scale_m=5.0 (what the Gradio UI passes via get_sensor_pixel_scale)
and quantifies the many-to-one collapse in match_pair_hf output.
Diagnosis only; does not modify app.py or chandra_align/.
"""
import csv
import os
import sys

sys.path.insert(0, "/tmp/stubs")
sys.path.insert(0, "/home/hatch/workspace/chandra-align")
os.chdir("/home/hatch/workspace/chandra-align")

import cv2
import numpy as np

import app

CROP = "data/benchmark_crops"
PAIR = "tmc2_01"
DESC = "south-polar highlands; large shadowed crater walls and rugged relief"
REF_PROD = "ch2_tmc_ncf_20231101T0125121344_d_img_d18"
SEC_PROD = "ch2_tmc_ncn_20231101T0125121377_d_img_d18"

# ---- instrument match_pair_hf to capture its input shapes ----
captured = {}
_orig_mph = app.match_pair_hf


def _spy_match_pair_hf(img1, img2, *a, **k):
    captured["ref_shape"] = tuple(img1.shape[:2])
    captured["sec_shape"] = tuple(img2.shape[:2])
    return _orig_mph(img1, img2, *a, **k)


app.match_pair_hf = _spy_match_pair_hf


def run_config(pixel_scale_m, label):
    captured.clear()
    ref = cv2.imread(f"{CROP}/{PAIR}_reference.png", cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(f"{CROP}/{PAIR}_source.png", cv2.IMREAD_GRAYSCALE)
    cv2.setRNGSeed(7)
    out = app._align_core(ref, sec, pixel_scale_m=pixel_scale_m,
                          reference_sensor_name="TMC-2")
    jm = out.get("judge_metrics", {}) or {}
    return {
        "label": label,
        "pixel_scale_m": pixel_scale_m,
        "match_ref_shape": captured.get("ref_shape"),
        "match_sec_shape": captured.get("sec_shape"),
        "status": out.get("status_code"),
        "n_matches": out.get("total_matches"),
        "n_inliers": jm.get("n_inliers"),
        "engine": jm.get("registration_engine"),
    }


def quantify_collapse(pixel_scale_m):
    """Run matching exactly as the hook sees it and count many-to-one hits."""
    from chandra_align.preprocessing.multimodal import resize_to_common_ground_sample
    from chandra_align.metrics.enhanced import get_sensor_pixel_scale
    ref = cv2.imread(f"{CROP}/{PAIR}_reference.png", cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(f"{CROP}/{PAIR}_source.png", cv2.IMREAD_GRAYSCALE)
    # replicate _align_core preprocessing relevant to matching: CLAHE only
    from app import apply_clahe, keypoint_starvation_guard
    ref_p = apply_clahe(ref, clip_limit=3.0, normalize_range=(0.0, 255.0))
    sec_p = apply_clahe(sec, clip_limit=3.0, normalize_range=(0.0, 255.0))
    ref_p, sec_p, _, _, _ = keypoint_starvation_guard(
        ref_p, sec_p, None, None, preprocessing_triggered=False,
        min_candidates=30, clahe_clip_limit=3.0)
    ref_gsd, sec_gsd = float(pixel_scale_m), get_sensor_pixel_scale("TMC-2")
    target = max(ref_gsd, sec_gsd)
    mref, rs = resize_to_common_ground_sample(ref_p, ref_gsd, target, 128)
    msec, ss = resize_to_common_ground_sample(sec_p, sec_gsd, target, 128)
    cv2.setRNGSeed(7)
    pts_ref, pts_sec, engine, _ = app.match_pair_hf(mref, msec, 3.0)
    # many-to-one: how many distinct ref targets, max hits on one
    if len(pts_ref):
        uniq, counts = np.unique(np.round(pts_ref, 1), axis=0, return_counts=True)
        order = np.argsort(-counts)
    else:
        uniq, counts, order = np.zeros((0, 2)), np.zeros(0), np.zeros(0, dtype=int)
    return {
        "n_matches": len(pts_ref),
        "n_unique_ref_targets": len(uniq),
        "max_hits_one_target": int(counts[order[0]]) if len(counts) else 0,
        "top3_targets": [(tuple(np.round(uniq[order[i]], 1)), int(counts[order[i]]))
                         for i in range(min(3, len(order)))],
        "n_targets_hit_by_gt5": int((counts > 5).sum()),
        "frac_matches_in_top5_targets": float(counts[order[:5]].sum() / len(pts_ref)) if len(pts_ref) else 0.0,
        "ref_scale": rs, "sec_scale": ss, "engine": engine,
    }


rows = []
print("=== Config A: harness (pixel_scale_m=0.25) ===")
ra = run_config(0.25, "harness")
print(ra)
print("=== Config B: UI-correct (pixel_scale_m=5.0) ===")
rb = run_config(5.0, "ui_correct")
print(rb)

print("\n=== Collapse quantification A ===")
qa = quantify_collapse(0.25)
for k, v in qa.items():
    print(f"  {k}: {v}")
print("\n=== Collapse quantification B ===")
qb = quantify_collapse(5.0)
for k, v in qb.items():
    print(f"  {k}: {v}")

with open("/home/hatch/workspace/chandra-align/results/table_tmc2_collapse.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pair", "pair_description", "reference_product", "source_product",
                "config", "pixel_scale_m", "match_ref_h", "match_ref_w",
                "match_sec_h", "match_sec_w", "status", "n_matches_hook",
                "n_unique_ref_targets", "max_hits_one_target",
                "n_targets_hit_by_gt5", "frac_matches_in_top5_targets",
                "engine"])
    for cfg, r, q in (("harness_misconfigured", ra, qa), ("ui_correct_scale", rb, qb)):
        w.writerow([PAIR, DESC, REF_PROD, SEC_PROD, cfg, r["pixel_scale_m"],
                    r["match_ref_shape"][0], r["match_ref_shape"][1],
                    r["match_sec_shape"][0], r["match_sec_shape"][1],
                    r["status"], r["n_matches"],
                    q["n_unique_ref_targets"], q["max_hits_one_target"],
                    q["n_targets_hit_by_gt5"], round(q["frac_matches_in_top5_targets"], 4),
                    q["engine"]])
print("\nwrote results/table_tmc2_collapse.csv")
