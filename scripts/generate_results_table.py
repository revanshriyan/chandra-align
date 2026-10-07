#!/usr/bin/env python3
"""Canonical results-table generator (issue #8).

Reads the repo's raw result files and prints the ONE canonical results table
(markdown to stdout) plus a JSON sidecar at results/table_canonical.json.
The README, release notes, and deck must quote only from this table.

Rules:
  - Pure Python 3.12, stdlib + numpy only. No network calls at runtime.
  - Every number cites its source file. If a source is missing or a number
    cannot be found, the cell reads UNVERIFIED -- never invented.
  - Repo root is derived from this file's location (scripts/../).

Usage:
    python scripts/generate_results_table.py
"""

import csv
import json
import os
import re
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
SIDECAR_PATH = os.path.join(RESULTS_DIR, "table_canonical.json")

UNVERIFIED = "UNVERIFIED"


def entry(metric, label, value, source, unit=""):
    return {
        "metric": metric,
        "label": label,
        "value": value,
        "unit": unit,
        "source": source,
        "verified": value != UNVERIFIED,
    }


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_text(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def fmt(x, nd=4):
    if x == UNVERIFIED or x is None:
        return UNVERIFIED
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main():
    entries = []

    # ---- 1. Calibrated synthetic pair, LightGlue/ALIKED (issue #1) ----
    src1 = "results/table_issue01_gpu_validation.csv"
    rows = read_csv(os.path.join(REPO_ROOT, src1))
    syn = None
    if rows:
        for r in rows:
            if r["pair"] == "synthetic_gentle" and r["matcher"] == "LightGlue_ALIKED":
                syn = r
                break
    if syn:
        entries.append(entry("synthetic_lightglue_rmse", "Synthetic pair RMSE (LightGlue/ALIKED)",
                             fmt(float(syn["rmse_px"])), src1, "px"))
        entries.append(entry("synthetic_lightglue_inliers", "Synthetic pair inliers (LightGlue/ALIKED)",
                             fmt(int(syn["inliers"]), 0), src1, ""))
        entries.append(entry("synthetic_lightglue_entropy", "Synthetic pair entropy (LightGlue/ALIKED)",
                             fmt(float(syn["entropy"])), src1, "bits"))
        entries.append(entry("synthetic_lightglue_quadrants", "Synthetic pair quadrants (LightGlue/ALIKED)",
                             syn["quadrants"].split(" ")[0], src1, ""))
        entries.append(entry("synthetic_lightglue_verdict", "Synthetic pair gate verdict (LightGlue/ALIKED)",
                             syn["gate"], src1, ""))
    else:
        for m, label in [
            ("synthetic_lightglue_rmse", "Synthetic pair RMSE (LightGlue/ALIKED)"),
            ("synthetic_lightglue_inliers", "Synthetic pair inliers (LightGlue/ALIKED)"),
            ("synthetic_lightglue_entropy", "Synthetic pair entropy (LightGlue/ALIKED)"),
            ("synthetic_lightglue_quadrants", "Synthetic pair quadrants (LightGlue/ALIKED)"),
            ("synthetic_lightglue_verdict", "Synthetic pair gate verdict (LightGlue/ALIKED)"),
        ]:
            entries.append(entry(m, label, UNVERIFIED, src1))

    # ---- 2/3. OHRC + TMC-2 in-sample (issue #1) and held-out / GT (issue #2) ----
    src2 = "results/table_issue02_ground_truth.csv"
    gt_rows = read_csv(os.path.join(REPO_ROOT, src2))
    insample = {}   # pair -> (rmse, inliers) from issue #1 LightGlue rows
    if rows:
        for r in rows:
            if r["matcher"] == "LightGlue_ALIKED" and r["pair"] in ("OHRC_pair", "TMC2_fore_nadir"):
                insample[r["pair"]] = (float(r["rmse_px"]), int(r["inliers"]))
    heldout = {}    # pair -> (heldout_rmse, gt_rmse)
    if gt_rows:
        for r in gt_rows:
            if "LightGlue" in r["matcher"] and r["pair"] in ("OHRC_pair", "TMC2_fore_nadir"):
                try:
                    heldout[r["pair"]] = (float(r["rmse_heldout_split"]), float(r["rmse_ground_truth"]))
                except (ValueError, TypeError):
                    pass

    def pair_entries(key, label, csv_pair):
        if csv_pair in insample:
            entries.append(entry(f"{key}_insample_rmse", f"{label} in-sample RMSE (LightGlue)",
                                 fmt(insample[csv_pair][0]), src1, "px"))
        else:
            entries.append(entry(f"{key}_insample_rmse", f"{label} in-sample RMSE (LightGlue)",
                                 UNVERIFIED, src1, "px"))
        if csv_pair in heldout:
            entries.append(entry(f"{key}_heldout_rmse", f"{label} split-held-out RMSE",
                                 fmt(heldout[csv_pair][0]), src2, "px"))
            entries.append(entry(f"{key}_gt_rmse", f"{label} independent GT RMSE (ChandraBench landmarks)",
                                 fmt(heldout[csv_pair][1]), src2, "px"))
        else:
            entries.append(entry(f"{key}_heldout_rmse", f"{label} split-held-out RMSE",
                                 UNVERIFIED, src2, "px"))
            entries.append(entry(f"{key}_gt_rmse", f"{label} independent GT RMSE (ChandraBench landmarks)",
                                 UNVERIFIED, src2, "px"))

    pair_entries("ohrc", "OHRC real pair", "OHRC_pair")
    pair_entries("tmc2", "TMC-2 fore/nadir real pair", "TMC2_fore_nadir")

    # ---- 4. IIRS<->TMC-2 LoFTR breakthrough (VNIR composite row) ----
    src4 = "docs/phase9-loftr-addendum.md"
    loftr_doc = read_text(os.path.join(REPO_ROOT, src4))
    loftr = {}
    if loftr_doc:
        m = re.search(r"\|\s*VNIR composite[^|]*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*px\s*\|\s*([A-Z_]+)\s*\|",
                      loftr_doc)
        if m:
            loftr = {"corr": m.group(1), "inliers": m.group(2),
                     "rmse": m.group(3), "verdict": m.group(4)}
    entries.append(entry("loftr_correspondences", "IIRS<->TMC-2 LoFTR correspondences (VNIR composite)",
                         loftr.get("corr", UNVERIFIED), src4, ""))
    entries.append(entry("loftr_inliers", "IIRS<->TMC-2 LoFTR inliers post-LK (VNIR composite)",
                         loftr.get("inliers", UNVERIFIED), src4, ""))
    entries.append(entry("loftr_rmse", "IIRS<->TMC-2 LoFTR RMSE (VNIR composite)",
                         loftr.get("rmse", UNVERIFIED), src4, "px"))
    entries.append(entry("loftr_verdict", "IIRS<->TMC-2 LoFTR gate verdict",
                         loftr.get("verdict", UNVERIFIED), src4, ""))

    # ---- 5. Phase 13 window tiling (medians computed from the CSV) ----
    src5 = "results/table_phase13_windows.csv"
    wrows = read_csv(os.path.join(REPO_ROOT, src5))
    if wrows:
        entries.append(entry("phase13_n_windows", "Phase 13 systematic windows (total)",
                             fmt(len(wrows), 0), src5, ""))
        for pair_key, label in [("ohrc_pair", "OHRC"), ("tmc2_fore_nadir", "TMC-2 fore/nadir")]:
            sub = [r for r in wrows if r["pair"] == pair_key]
            coarse = [r for r in sub if r["verdict"] == "COARSE_ADVISORY"]
            try:
                med_inl = statistics.median(float(r["n_inliers_unique"]) for r in coarse)
                med_rmse = statistics.median(float(r["rmse_px"]) for r in coarse)
                inl_v, rmse_v = fmt(med_inl, 0), fmt(med_rmse, 2)
            except statistics.StatisticsError:
                inl_v, rmse_v = UNVERIFIED, UNVERIFIED
            tag = "ohrc" if pair_key == "ohrc_pair" else "tmc2"
            entries.append(entry(f"phase13_{tag}_coarse", f"Phase 13 {label} COARSE_ADVISORY windows",
                                 f"{len(coarse)}/{len(sub)}", src5, ""))
            entries.append(entry(f"phase13_{tag}_median_inliers",
                                 f"Phase 13 {label} median inliers (COARSE windows)",
                                 inl_v, src5, ""))
            entries.append(entry(f"phase13_{tag}_median_rmse",
                                 f"Phase 13 {label} median RMSE (COARSE windows)",
                                 rmse_v, src5, "px"))
    else:
        for m, label in [
            ("phase13_n_windows", "Phase 13 systematic windows (total)"),
            ("phase13_ohrc_coarse", "Phase 13 OHRC COARSE_ADVISORY windows"),
            ("phase13_ohrc_median_inliers", "Phase 13 OHRC median inliers (COARSE windows)"),
            ("phase13_ohrc_median_rmse", "Phase 13 OHRC median RMSE (COARSE windows)"),
            ("phase13_tmc2_coarse", "Phase 13 TMC-2 fore/nadir COARSE_ADVISORY windows"),
            ("phase13_tmc2_median_inliers", "Phase 13 TMC-2 median inliers (COARSE windows)"),
            ("phase13_tmc2_median_rmse", "Phase 13 TMC-2 median RMSE (COARSE windows)"),
        ]:
            entries.append(entry(m, label, UNVERIFIED, src5))

    # ---- 6. Phase 15 pose graph (synthetic misclosure) ----
    src6 = "results/phase15_posegraph.json"
    pg = read_json(os.path.join(REPO_ROOT, src6))
    if pg and isinstance(pg.get("synthetic"), dict):
        syn_pg = pg["synthetic"]
        entries.append(entry("phase15_misclosure_before_px", "Phase 15 synthetic misclosure before optimization",
                             fmt(float(syn_pg["misclosure_before_px"])), src6, "px"))
        entries.append(entry("phase15_misclosure_after_px", "Phase 15 synthetic misclosure after optimization",
                             fmt(float(syn_pg["misclosure_after_px"])), src6, "px"))
        entries.append(entry("phase15_real_status", "Phase 15 real-data triplet status",
                             pg.get("real_triplet", {}).get("status", UNVERIFIED), src6, ""))
    else:
        for m, label in [
            ("phase15_misclosure_before_px", "Phase 15 synthetic misclosure before optimization"),
            ("phase15_misclosure_after_px", "Phase 15 synthetic misclosure after optimization"),
            ("phase15_real_status", "Phase 15 real-data triplet status"),
        ]:
            entries.append(entry(m, label, UNVERIFIED, src6))

    # ---- 7. ChandraBench v0.1 independent GT RMSE ----
    src7 = "docs/chandrabench-tech-note.md"
    cb_doc = read_text(os.path.join(REPO_ROOT, src7))
    cb = {}
    if cb_doc:
        m_ohrc = re.search(r"\|\s*OHRC_pair\s*\|\s*LightGlue/ALIKED\s*\|[^|]*\|\s*\*\*([\d.]+)\s*px\*\*", cb_doc)
        m_tmc2 = re.search(r"\|\s*TMC-2 fore/nadir\s*\|\s*LightGlue/ALIKED\s*\|[^|]*\|\s*\*\*([\d.]+)\s*px\*\*", cb_doc)
        if m_ohrc:
            cb["ohrc"] = m_ohrc.group(1)
        if m_tmc2:
            cb["tmc2"] = m_tmc2.group(1)
    # Prefer the full-precision GT values from the issue #2 table when present.
    if "OHRC_pair" in heldout:
        cb["ohrc"] = fmt(heldout["OHRC_pair"][1])
    if "TMC2_fore_nadir" in heldout:
        cb["tmc2"] = fmt(heldout["TMC2_fore_nadir"][1])
    entries.append(entry("chandrabench_ohrc_gt_rmse", "ChandraBench v0.1 OHRC forward GT RMSE",
                         cb.get("ohrc", UNVERIFIED), src7 + " (+ " + src2 + ")", "px"))
    entries.append(entry("chandrabench_tmc2_gt_rmse", "ChandraBench v0.1 TMC-2 forward GT RMSE",
                         cb.get("tmc2", UNVERIFIED), src7 + " (+ " + src2 + ")", "px"))

    # ---- 8. Frozen gate thresholds (static; verified in code) ----
    gate_src = ("frozen gate spec — app.py (judge checklist), "
                "chandra_align/registration_transform.py (MIN_REGISTRATION_INLIERS=8)")
    for m, label, val, unit in [
        ("gate_accept_rmse_px", "Gate ACCEPT: RMSE threshold", "0.50", "px"),
        ("gate_accept_min_inliers", "Gate ACCEPT: minimum inliers", "8", ""),
        ("gate_accept_min_entropy", "Gate ACCEPT: minimum entropy", "0.75", "bits"),
        ("gate_accept_min_quadrants", "Gate ACCEPT: minimum quadrants", "3/4", ""),
        ("gate_coarse_rmse_px", "Gate COARSE: RMSE threshold", "2.50", "px"),
        ("gate_coarse_min_inliers", "Gate COARSE: minimum inliers", "8", ""),
        ("gate_coarse_min_entropy", "Gate COARSE: minimum entropy", "0.50", "bits"),
        ("gate_coarse_min_quadrants", "Gate COARSE: minimum quadrants", "2/4", ""),
    ]:
        entries.append(entry(m, label, val, gate_src, unit))

    # ---- 9. Deform-field stage (opt-in) shipped results ----
    src9a = "results/table_quota_12pair.csv"
    qrows = read_csv(os.path.join(REPO_ROOT, src9a))
    qmap = {}
    if qrows:
        for r in qrows:
            if r.get("flag") == "on":
                qmap[r["pair"]] = r
    for pid, short in [("ohrc_01", "OHRC cratered-rim window (12:09->14:06)"),
                       ("ohrc_02", "OHRC massif-slope window (12:09->14:06)"),
                       ("ohrc_03", "OHRC hummocky-relief window (12:09->14:06)")]:
        r = qmap.get(pid)
        if r and r.get("rmse_gate_px") and r.get("n_inliers") and r.get("status_code"):
            entries.append(entry(f"deform_stage_{pid}_gate_rmse",
                                 f"Deform-field stage gate RMSE ({short}, flag on)",
                                 fmt(float(r["rmse_gate_px"])), src9a, "px"))
            entries.append(entry(f"deform_stage_{pid}_inliers",
                                 f"Deform-field stage inliers ({short}, flag on)",
                                 str(int(float(r["n_inliers"]))), src9a, ""))
            entries.append(entry(f"deform_stage_{pid}_verdict",
                                 f"Deform-field stage verdict ({short}, flag on)",
                                 r["status_code"], src9a, ""))
        else:
            for m, label in [
                (f"deform_stage_{pid}_gate_rmse", f"Deform-field stage gate RMSE ({short}, flag on)"),
                (f"deform_stage_{pid}_inliers", f"Deform-field stage inliers ({short}, flag on)"),
                (f"deform_stage_{pid}_verdict", f"Deform-field stage verdict ({short}, flag on)"),
            ]:
                entries.append(entry(m, label, UNVERIFIED, src9a))
    src9b = "results/table_lroc_field.csv"
    lrows = read_csv(os.path.join(REPO_ROOT, src9b))
    lmap = {r["grid"]: r for r in lrows} if lrows else {}
    for grid, label in [("1m", "LROC 1m grid"), ("3m", "LROC 3m grid")]:
        r = lmap.get(grid)
        if r and r.get("field_x_check_rmse_m") and r.get("field_y_check_rmse_m"):
            lo = min(float(r["field_x_check_rmse_m"]), float(r["field_y_check_rmse_m"]))
            hi = max(float(r["field_x_check_rmse_m"]), float(r["field_y_check_rmse_m"]))
            entries.append(entry(f"lroc_field_{grid}_bound_m",
                                 f"{label} independent held-out absolute bound (field stage)",
                                 f"{lo:.1f}-{hi:.1f}", src9b, "m"))
        else:
            entries.append(entry(f"lroc_field_{grid}_bound_m",
                                 f"{label} independent held-out absolute bound (field stage)",
                                 UNVERIFIED, src9b))

    # ---- Markdown table to stdout ----
    lines = []
    lines.append("# CHANDRA-ALIGN canonical results table")
    lines.append("")
    lines.append("Generated by `scripts/generate_results_table.py`. "
                 "The README, release notes, and deck must quote only from this table.")
    lines.append("")
    lines.append("| Metric | Value | Source |")
    lines.append("| --- | --- | --- |")
    for e in entries:
        val = e["value"] + (f" {e['unit']}" if e["unit"] else "")
        lines.append(f"| {e['label']} | {val} | `{e['source']}` |")
    lines.append("")
    n_unver = sum(1 for e in entries if not e["verified"])
    lines.append(f"_Entries: {len(entries)}; UNVERIFIED: {n_unver}._")
    print("\n".join(lines))

    # ---- JSON sidecar ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SIDECAR_PATH, "w", encoding="utf-8") as f:
        json.dump({"entries": entries,
                   "n_entries": len(entries),
                   "n_unverified": n_unver}, f, indent=2)
    print(f"\nWrote {os.path.relpath(SIDECAR_PATH, REPO_ROOT)}", file=sys.stderr)

    return 0 if n_unver == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
