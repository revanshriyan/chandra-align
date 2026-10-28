#!/usr/bin/env python3
"""Phase 1 reconstruction: 19-pair arm comparison under the DEFAULT-ON strip
guard (2026-10-26). Benchmark Engineer, "Master the 2D benchmarks" plan.

RECONSTRUCTION, not a re-run: the original 19-pair runner was never
committed and was not found by two independent searches (see
~/workspace/2d-mastery-scoping-2026-10-26.md, Leg 1). This script rebuilds
it from the surviving evidence. Every deliberate deviation from the
(unknown) original is stated below; nothing is silently substituted.

Reference table: chandra-align/results/arm_comparison_19pair_2026-10-15.csv
(114 rows = 19 pairs x 6 arms; 57 COARSE / 56 DEGENERATE / 1 ERROR).

Pairs: ~/workspace/crossfeat_data/manifest.csv, status OK only (N11 is
SKIP: rmse 3.285 > 2.5, so 13 N + 6 W = 19 pairs). Pair order here is
manifest order (W0-W5, N1-N10, N12-N14); the Oct-15 row order was
scattered (parallel assembly) — order is immaterial to the comparison.

Arms (Oct-15 names) and their matcher entry points, all on the two PNG
windows at native pixel size, ref = IIRS window, sec = TMC window
(manifest order; matches the repo trial-harness convention in
scripts/iirs_breadth.py, where inliers_a is the IIRS side and the gate
frame is the IIRS window shape):
  default        plain SIFT: SIFT_create(nfeatures=8000), BF knnMatch k=2,
                 Lowe 0.75, sec->ref direction (matcher_loftr_experiment.py
                 recipe). NOT app.match_pair_hf: its SIFT fallback returns
                 only >=8-inlier candidates, which contradicts the Oct-15
                 default rows (2-3 inliers recorded).
  pc             chandra_align.pc_frontend.match_pair_pc (pure numpy/cv2).
  xoftr          chandra_align.xoftr_frontend.match_pair_xoftr (XoFTR-640).
  matchanything  chandra_align.matchanything_frontend.match_pair_matchanything.
  anymatch       chandra_align.anymatch_frontend.match_pair_anymatch.
  loftr          chandra_align.xmodal.loftr_arm.LoFTRMatcher(pretrained=
                 "outdoor").match() at NATIVE resolution, conf > 0.2.
                 (Gap 1 RESOLVED: this is the only base-LoFTR entry point
                 in the repo; scripts/matcher_loftr_experiment.py is the
                 same kornia outdoor model with a 1024-long-side reduction
                 for OHRC crops. Native resolution is chosen because the
                 Oct-15 W0 loftr ERROR row ("OOM on 4634px-tall input, CPU
                 resource limit") proves the original ran native-res on
                 CPU: a 1024-reduced W0 (1024x48) would not OOM. W0-loftr
                 is therefore EXPECTED to ERROR again here; it is run
                 last so a crash cannot block the other 113 rows.)

Scoring (trial-harness convention, identical to the repo's gate_verdict
in scripts/iirs_breadth.py and scripts/run_phase9_iirs_retry.py):
  verify_guarded (RANSAC 3.0 / max_iters 2000 / confidence 0.99,
  image_shape = ref window) -> inliers_a/b, model; rmse_gate_px = MEAN
  inlier residual magnitude (the trial convention; the pipeline's
  in-sample RMS is a different statistic and is NOT used here);
  quadrants/entropy via compute_quadrant_metrics; verdict via
  validate_registration_gate. THE ONLY intentional scoring difference
  vs Oct-15 is the strip guard: Oct-15 called the gate WITHOUT pts, so
  no strip backstop could fire; this run passes pts + frame_shape with
  the guard at its DEPLOYED DEFAULT (ON, 0.159). Guard state is asserted
  at startup and recorded per row (guard_fired + minor_ratio).

Recording rule (reconstructed from the Oct-15 CSV + app.py internals):
  n_corr < 8  -> DEGENERATE_FAILURE, note "arm <8 correspondences",
                 rmse/inliers/n_corr ALL BLANK in the CSV (the Oct-15
                 convention; the true n_corr is kept in the JSON
                 sidecar). This branch is fully consistent across all
                 Oct-15 rows.
  fit abstain -> DEGENERATE_FAILURE, note = abstain_code, rmse blank,
                 inliers 0, n_corr recorded.
  otherwise   -> rmse_gate_px (mean inlier residual), verdict, inliers,
                 n_corr all recorded.
  (Gap 2, residual anomaly, stated honestly: 4 Oct-15 default rows
  (N12, N1, W1, W5) record inliers=2 and n_corr but BLANK rmse, while 8
  other 2-inlier default rows record 0.0. No rule in current code
  distinguishes them: the pipeline's DEGENERATE dict always carries
  rmse_gate_px when the fit path runs, so those 4 rows must have taken
  an exception/early-rejection path in the original harness (or a
  different harness version for a subset of rows — the scattered row
  order is consistent with parallel workers). My single consistent rule
  records rmse for every fitted row (0.0 for an exact 2-point fit).
  Consequence for attribution: an Oct-15-blank vs mine-valued rmse on a
  row whose VERDICT is unchanged is classified as RECORDING-RULE drift,
  never as a guard effect. It cannot contaminate the verdict analysis:
  all such rows are already DEGENERATE and the guard only downgrades.)

GSD treatment (Gap 3 RESOLVED — trial-harness route, NO GSD
normalization): both PNG windows are used at native pixel size and
each frontend applies only its own internal scaling (anymatch shared
PE-limit downscale; xoftr DataIOWrapper rescale; loftr native).
Rationale: the Oct-15 numbers are irreproducible via the deployed GSD
path — W5 anymatch COARSE with 1223 inliers is impossible on the
deployed 320x22 strip (docs/iirs-gsd-investigation-2026-10-16.md proves
the deployed path gives DEGENERATE there for every arm). The deployed-
path comparison is a DIFFERENT experiment, already done 2026-10-16 in
results/table_iirs_deployed_path_2026-10-16.csv, and is out of scope
for this re-run. Points are returned in caller (window) frames by every
frontend, so gate coordinates are window coordinates throughout.

GPU/CPU feasibility (Gap 4): this VM is CPU-only (no nvidia-smi, torch
cuda False, 7 GB RAM). All six arms report available (weights present
or kornia cached: loftr_outdoor.ckpt, weights_xoftr_640.ckpt). CPU
feasibility is established by the smoke test (see --smoke) and by
per-row try/except in the full run: a row that OOMs or errors is
recorded as ERROR (never aborts the run) and reported as NOT
RERUNNABLE here for that arm, not as a failure of the arm. Frontends
swallow exceptions and return (None, None) — such a row lands in the
<8 branch, which is documented, not hidden; only the loftr arm (direct
LoFTRMatcher call) surfaces exceptions as ERROR rows, matching the
Oct-15 W0 loftr ERROR precedent.

Pre-registered bar: every verdict change vs Oct-15 attributed as
GUARD-CAUSED (pre-guard COARSE, final DEGENERATE, guard fired, minor
ratio quoted) vs OTHER (mechanism investigated) vs RECORDING-RULE
(rmse-field-only difference, verdict unchanged). Expected signal is
small: default is already 19/19 DEGENERATE, so only learned-arm COARSE
rows with thin strips can flip.

Usage:
  python3 rerun_arm_19pair_2026-10-26.py --smoke     # W5 x default/xoftr/loft
  python3 rerun_arm_19pair_2026-10-26.py --full      # all 19 x 6, checkpointed
  python3 rerun_arm_19pair_2026-10-26.py --full --resume
Outputs (all under ~/workspace/guardon_19pair_2026-10-26/):
  arm_comparison_19pair_guardon_2026-10-26.csv  (checkpointed per row)
  rows.jsonl                                    (full per-row detail)
  report.md                                     (written at the end)
No repo file is modified, nothing is committed or pushed, gates frozen.
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback

import cv2
import numpy as np

OUT_DIR = os.path.expanduser("~/workspace/guardon_19pair_2026-10-26")
MANIFEST = os.path.expanduser("~/workspace/crossfeat_data/manifest.csv")
REF_CSV = os.path.expanduser(
    "~/workspace/chandra-align/results/arm_comparison_19pair_2026-10-15.csv")
REPO = os.path.expanduser("~/workspace/chandra-align")

ARMS = ["default", "pc", "xoftr", "matchanything", "anymatch", "loftr"]

CSV_FIELDS = ["pair", "arm", "rmse_gate_px", "verdict", "inliers", "n_corr",
              "match_s", "notes", "guard_fired", "minor_ratio",
              "pre_guard_verdict", "verdict_changed_vs_oct15", "attribution"]


def load_pairs():
    pairs = []
    with open(MANIFEST, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"].strip() != "OK":
                continue
            pairs.append((row["window"].strip(), row["iirs_path"].strip(),
                          row["tmc_path"].strip()))
    return pairs


def load_reference():
    ref = {}
    with open(REF_CSV, newline="") as f:
        for row in csv.DictReader(f):
            ref[(row["pair"], row["arm"])] = row
    return ref


def sift_match(ref_u8, sec_u8):
    """Plain SIFT + BF + Lowe 0.75 (matcher_loftr_experiment recipe)."""
    sift = cv2.SIFT_create(nfeatures=8000)
    k_ref, d_ref = sift.detectAndCompute(ref_u8, None)
    k_sec, d_sec = sift.detectAndCompute(sec_u8, None)
    if d_ref is None or d_sec is None or len(d_ref) < 2 or len(d_sec) < 2:
        return (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32))
    raw = cv2.BFMatcher().knnMatch(d_sec, d_ref, k=2)
    good = []
    for pair_m in raw:
        if len(pair_m) == 2 and pair_m[0].distance < 0.75 * pair_m[1].distance:
            good.append(pair_m[0])
    pts_ref = np.float32([k_ref[m.trainIdx].pt for m in good])
    pts_sec = np.float32([k_sec[m.queryIdx].pt for m in good])
    return pts_ref, pts_sec


def run_matcher(arm, ref_u8, sec_u8):
    """Returns (pts_ref, pts_sec) or (None, None) on frontend failure.
    Raises only for the loftr arm (direct LoFTRMatcher call, no swallow)."""
    if arm == "default":
        return sift_match(ref_u8, sec_u8)
    if arm == "pc":
        from chandra_align.pc_frontend import match_pair_pc
        return match_pair_pc(ref_u8, sec_u8)
    if arm == "xoftr":
        from chandra_align.xoftr_frontend import match_pair_xoftr
        return match_pair_xoftr(ref_u8, sec_u8)
    if arm == "matchanything":
        from chandra_align.matchanything_frontend import match_pair_matchanything
        return match_pair_matchanything(ref_u8, sec_u8)
    if arm == "anymatch":
        from chandra_align.anymatch_frontend import match_pair_anymatch
        return match_pair_anymatch(ref_u8, sec_u8)
    if arm == "loftr":
        from chandra_align.xmodal.loftr_arm import LoFTRMatcher
        la, lb = LoFTRMatcher(pretrained="outdoor").match(ref_u8, sec_u8)
        return np.asarray(la, np.float32), np.asarray(lb, np.float32)
    raise ValueError(f"unknown arm {arm}")


def score_row(pair, arm, ref_u8, sec_u8, ref_row):
    """Run one (pair, arm) cell. Returns (csv_row_dict, json_detail)."""
    from chandra_align.metrics import compute_quadrant_metrics
    from chandra_align.metrics.quadrant import validate_registration_gate as pre_gate
    from chandra_align.metrics import validate_registration_gate as full_gate
    from chandra_align.metrics.spatial_support import (
        check_spatial_support, strip_guard_enabled)
    from chandra_align.refine import verify_guarded

    t0 = time.perf_counter()
    pts_ref, pts_sec = run_matcher(arm, ref_u8, sec_u8)
    match_s = round(time.perf_counter() - t0, 1)
    if pts_ref is None or pts_sec is None:
        n_corr_raw = 0
    else:
        n_corr_raw = int(len(pts_ref))
    detail = {"pair": pair, "arm": arm, "match_s": match_s,
              "n_corr_raw": n_corr_raw,
              "guard_enabled": bool(strip_guard_enabled()),
              "ref_shape": [int(ref_u8.shape[0]), int(ref_u8.shape[1])]}
    row = {"pair": pair, "arm": arm, "match_s": match_s, "notes": "",
           "guard_fired": "", "minor_ratio": "", "pre_guard_verdict": "",
           "verdict_changed_vs_oct15": "", "attribution": ""}

    if n_corr_raw < 8:
        row.update({"rmse_gate_px": "", "verdict": "DEGENERATE_FAILURE",
                    "inliers": "", "n_corr": "",
                    "notes": "arm <8 correspondences",
                    "attribution": "UNCHANGED" if ref_row and ref_row["verdict"] == "DEGENERATE_FAILURE" else "OTHER"})
        detail.update({"branch": "lt8", "verdict": "DEGENERATE_FAILURE"})
        _finish_attribution(row, ref_row)
        return row, detail

    row["n_corr"] = n_corr_raw
    cv2.setRNGSeed(7)
    cfg = {"ransac_reproj_threshold": 3.0, "max_iters": 2000, "confidence": 0.99}
    g = verify_guarded(pts_ref, pts_sec, cfg, image_shape=ref_u8.shape[:2])
    detail.update({"abstain_code": g["abstain_code"],
                   "span_guard": bool(g["span_guard"]),
                   "n_unique": int(g["n_unique"])})
    if not g["ok"]:
        row.update({"rmse_gate_px": "", "verdict": "DEGENERATE_FAILURE",
                    "inliers": 0,
                    "notes": f"fit abstain: {g['abstain_code']}"})
        detail.update({"branch": "abstain", "verdict": "DEGENERATE_FAILURE"})
        _finish_attribution(row, ref_row)
        return row, detail

    ia, ib, M = g["inliers_a"], g["inliers_b"], g["model"]
    proj = ia @ M[:, :2].T + M[:, 2]
    resid = np.hypot(proj[:, 0] - ib[:, 0], proj[:, 1] - ib[:, 1])
    rmse = float(resid.mean())
    qm_counts, qm_entropy = compute_quadrant_metrics(ia, resid, ref_u8.shape[:2])
    _, pre_code = pre_gate(rmse, len(ia), 8, qm_entropy, qm_counts)
    msg, code = full_gate(rmse, len(ia), 8, qm_entropy, qm_counts,
                          model=M, pts=ia, frame_shape=ref_u8.shape[:2])
    ok_s, rep = check_spatial_support(ia, ref_u8.shape[:2])
    minor = float(rep["minor_ratio"]) if rep.get("minor_ratio") is not None else None
    guard_fired = bool(rep.get("valid") and not ok_s and strip_guard_enabled()
                       and pre_code != "DEGENERATE_FAILURE"
                       and code == "DEGENERATE_FAILURE")
    row.update({"rmse_gate_px": round(rmse, 4), "verdict": code,
                "inliers": int(len(ia)),
                "guard_fired": "yes" if guard_fired else "no",
                "minor_ratio": round(minor, 4) if minor is not None else "",
                "pre_guard_verdict": pre_code,
                "notes": msg})
    detail.update({"branch": "gated", "rmse_gate_px": rmse,
                   "entropy": float(qm_entropy), "quadrants": str(qm_counts),
                   "pre_guard_verdict": pre_code, "final_verdict": code,
                   "guard_fired": guard_fired, "minor_ratio": minor,
                   "strip_valid": bool(rep.get("valid")),
                   "gate_message": msg})
    _finish_attribution(row, ref_row)
    return row, detail


def _finish_attribution(row, ref_row):
    if ref_row is None:
        row["verdict_changed_vs_oct15"] = ""
        row["attribution"] = "NEW (no Oct-15 row)"
        return
    ref_verdict = ref_row["verdict"]
    changed = "yes" if row["verdict"] != ref_verdict else "no"
    row["verdict_changed_vs_oct15"] = changed
    if changed == "no":
        # rmse-field-only differences on same-verdict rows are recording
        # drift (incl. the Oct-15 blank-vs-0.0 anomaly), never guard effects.
        row["attribution"] = "UNCHANGED"
    elif row.get("guard_fired") == "yes":
        row["attribution"] = "GUARD-CAUSED"
    elif row["verdict"] == "ERROR":
        row["attribution"] = "NOT-RERUNNABLE"
    else:
        row["attribution"] = "OTHER (investigate)"


def run_cell(pair, arm, iirs_path, tmc_path, ref_row):
    ref_u8 = cv2.imread(iirs_path, cv2.IMREAD_GRAYSCALE)
    sec_u8 = cv2.imread(tmc_path, cv2.IMREAD_GRAYSCALE)
    if ref_u8 is None or sec_u8 is None:
        raise RuntimeError(f"could not load pair {pair}")
    try:
        return score_row(pair, arm, ref_u8, sec_u8, ref_row)
    except Exception as exc:
        tb = traceback.format_exc(limit=3)
        row = {"pair": pair, "arm": arm, "rmse_gate_px": "",
               "verdict": "ERROR", "inliers": "", "n_corr": "",
               "match_s": "", "notes": f"ERROR:{type(exc).__name__}: {exc}",
               "guard_fired": "", "minor_ratio": "", "pre_guard_verdict": "",
               "verdict_changed_vs_oct15": "", "attribution": ""}
        _finish_attribution(row, ref_row)
        detail = {"pair": pair, "arm": arm, "branch": "error",
                  "error": f"{type(exc).__name__}: {exc}", "traceback": tb}
        return row, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if not (args.smoke or args.full):
        ap.error("pass --smoke or --full")

    sys.path.insert(0, REPO)
    os.makedirs(OUT_DIR, exist_ok=True)
    from chandra_align.metrics.spatial_support import strip_guard_enabled
    guard_on = bool(strip_guard_enabled())
    print(f"strip guard enabled (must be True): {guard_on}", flush=True)
    assert guard_on, "guard is not at deployed default-ON; refusing to run"
    assert os.environ.get("CHANDRA_STRIP_GUARD", "") != "0"

    pairs = load_pairs()
    ref = load_reference()
    print(f"pairs: {len(pairs)} (manifest OK only)", flush=True)

    csv_path = os.path.join(OUT_DIR, "arm_comparison_19pair_guardon_2026-10-26.csv")
    jsonl_path = os.path.join(OUT_DIR, "rows.jsonl")
    done = set()
    if args.resume and os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["pair"], r["arm"]))
        print(f"resuming: {len(done)} rows already done", flush=True)
    mode = "a" if (args.resume and os.path.exists(csv_path)) else "w"
    fcsv = open(csv_path, mode, newline="")
    writer = csv.DictWriter(fcsv, fieldnames=CSV_FIELDS)
    if mode == "w":
        writer.writeheader()
    fjson = open(jsonl_path, "a")

    if args.smoke:
        cells = [("W5", "default"), ("W5", "xoftr"), ("W5", "loftr")]
        pmap = {p[0]: p for p in pairs}
        for pair, arm in cells:
            _, iirs, tmc = pmap[pair]
            print(f"SMOKE {pair} x {arm} ...", flush=True)
            row, detail = run_cell(pair, arm, iirs, tmc, ref.get((pair, arm)))
            writer.writerow(row)
            fcsv.flush()
            fjson.write(json.dumps(detail) + "\n")
            fjson.flush()
            print(f"  -> {row['verdict']} rmse={row['rmse_gate_px']} "
                  f"inliers={row['inliers']} n_corr={row['n_corr']} "
                  f"match_s={row['match_s']} guard={row['guard_fired']}",
                  flush=True)
        print("smoke done", flush=True)
        return

    # Full run: W0-loftr last (expected native-res OOM on 4634px input;
    # a crash there must not block the other 113 rows).
    cells = [(p[0], a, p[1], p[2]) for p in pairs for a in ARMS]
    cells.sort(key=lambda c: (c[0] == "W0" and c[1] == "loftr", c[0], c[1]))
    n = 0
    for pair, arm, iirs, tmc in cells:
        if (pair, arm) in done:
            continue
        print(f"[{n+1}/{len(cells)}] {pair} x {arm} ...", flush=True)
        row, detail = run_cell(pair, arm, iirs, tmc, ref.get((pair, arm)))
        writer.writerow(row)
        fcsv.flush()
        fjson.write(json.dumps(detail) + "\n")
        fjson.flush()
        print(f"  -> {row['verdict']} rmse={row['rmse_gate_px']} "
              f"inliers={row['inliers']} n_corr={row['n_corr']} "
              f"match_s={row['match_s']} guard={row['guard_fired']} "
              f"changed={row['verdict_changed_vs_oct15']} "
              f"attr={row['attribution']}", flush=True)
        n += 1
    print(f"full run done: {n} new rows", flush=True)


if __name__ == "__main__":
    main()
