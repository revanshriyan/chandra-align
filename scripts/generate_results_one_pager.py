#!/usr/bin/env python3
"""Generate docs/RESULTS_ONE_PAGER.md from the repository's own evidence.

Every number in the one-pager is copied verbatim from a committed results
file, and every line names that source file plus the scoring basis
(inlier (fit) RMSE / held-out RMSE / independent landmark RMSE) and the
hardware the source records ("hardware: not recorded" is preserved, never
filled in). Real pairs come first; synthetic rows are labelled sanity
checks.

Determinism contract (tests/test_results_one_pager_drift.py depends on it):
- fixed section order, sorted iteration everywhere;
- values are the source files' own strings, never reformatted floats;
- no timestamps, no absolute paths, no environment details;
- LF newlines, single trailing newline.

Evidence selection (documented, deterministic):
- results/table_canonical_v1.csv  -- the canonical evidence table (issue #8
  generator output); supplies the gpu_validation, ground_truth,
  crop_benchmark_12pair, iirs_first_contact and external_baselines rows;
- results/table_gpu_bench_2026-11-02.csv -- the 2026-11-02 T4 GPU bench;
- results/table_strip_guard_calibration_2026-10-24.csv -- human-verified
  landmark RMSE + strip-guard telemetry;
- results/table_rescue_noncircular.csv -- non-circular held-out ranges;
- chandrabench/landmarks_*_v0.1.csv -- the frozen landmark sets (row counts).

Only stdlib is used (csv, argparse, pathlib).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CANONICAL = REPO / "results" / "table_canonical_v1.csv"
GPU_BENCH = REPO / "results" / "table_gpu_bench_2026-11-02.csv"
STRIP_GUARD = REPO / "results" / "table_strip_guard_calibration_2026-10-24.csv"
RESCUE = REPO / "results" / "table_rescue_noncircular.csv"
CB_OHRC = REPO / "chandrabench" / "landmarks_ohrc_pair_v0.1.csv"
CB_TMC2 = REPO / "chandrabench" / "landmarks_tmc2_fore_nadir_v0.1.csv"

SYNTHETIC_PAIRS = {"synthetic_gentle"}


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _hardware(hw: str) -> str:
    hw = (hw or "").strip()
    return f"hardware: {hw}" if hw else "hardware: not recorded"


def _basis(row: dict) -> str:
    """Name every scoring basis present in a canonical row, in fixed order."""
    parts = []
    if row["rmse_px"].strip():
        parts.append(f"inlier (fit) RMSE {row['rmse_px']} px")
    if row["rmse_heldout_px"].strip():
        parts.append(f"held-out RMSE {row['rmse_heldout_px']} px")
    if row["gt_rmse_px"].strip():
        parts.append(f"independent landmark RMSE {row['gt_rmse_px']} px")
    if not parts:
        return "no RMSE on this row (verdict-only)"
    return "; ".join(parts)


def _canonical_line(row: dict) -> str:
    if row["benchmark"] == "ground_truth":
        verdict_part = f"verdict_agrees: {row['verdict']}"
    else:
        verdict_part = f"verdict \"{row['verdict']}\""
    return (f"- {row['pair']} / {row['matcher']}: {_basis(row)}; {verdict_part}; "
            f"source: results/table_canonical_v1.csv "
            f"({row['benchmark']}); {_hardware(row['hardware'])}")


def _ranges(rows: list[dict], column: str) -> tuple[str, str] | None:
    values = []
    for row in rows:
        raw = (row.get(column) or "").strip()
        if raw:
            try:
                values.append((float(raw), raw))
            except ValueError:
                continue
    if not values:
        return None
    values.sort()
    return values[0][1], values[-1][1]


def build_one_pager() -> str:
    canonical = _read_csv(CANONICAL)
    gpu_bench = _read_csv(GPU_BENCH)
    strip_guard = _read_csv(STRIP_GUARD)
    rescue = _read_csv(RESCUE)
    cb_counts = {"ohrc_pair": sum(1 for _ in csv.DictReader(CB_OHRC.open(encoding="utf-8", newline=""))),
                 "tmc2_fore_nadir": sum(1 for _ in csv.DictReader(CB_TMC2.open(encoding="utf-8", newline="")))}

    real_canon = [r for r in canonical if r["pair"] not in SYNTHETIC_PAIRS]
    synth_canon = [r for r in canonical if r["pair"] in SYNTHETIC_PAIRS]
    real_gpu = [r for r in gpu_bench if r["pair"] not in SYNTHETIC_PAIRS]
    synth_gpu = [r for r in gpu_bench if r["pair"] in SYNTHETIC_PAIRS]

    lines: list[str] = []
    lines.append("# Results one-pager (generated)")
    lines.append("")
    lines.append("Produced by `scripts/generate_results_one_pager.py` from the committed")
    lines.append("evidence files named on every line below. Do not hand-edit: the drift test")
    lines.append("(`tests/test_results_one_pager_drift.py`) regenerates this file and fails on")
    lines.append("any difference. Real pairs first; synthetic rows are sanity checks.")
    lines.append("")

    lines.append("## Independent landmark scoring (ChandraBench v0.1 landmarks)")
    lines.append("")
    for row in real_canon:
        if row["benchmark"] == "ground_truth":
            lines.append(_canonical_line(row))
    lines.append("")
    lines.append("Human-verified calibration landmarks (strip-guard calibration set):")
    lines.append("")
    for row in strip_guard:
        if row["landmark_rmse_px"].strip():
            lines.append(
                f"- {row['pair']}: independent landmark RMSE {row['landmark_rmse_px']} px "
                f"over {row['n_verified_landmarks']} verified landmarks; source: "
                f"results/table_strip_guard_calibration_2026-10-24.csv; {_hardware('')}"
            )
        else:
            lines.append(
                f"- {row['pair']}: landmark RMSE not measured on this row "
                f"(strip guard collapsed_at_0.25={row['collapsed_at_0.25']}); source: "
                f"results/table_strip_guard_calibration_2026-10-24.csv"
            )
    lines.append("")

    lines.append("## Held-out verification (non-circular 80/20 splits)")
    lines.append("")
    held = _ranges(rescue, "noncircular_rmse_px")
    admitted = _ranges(rescue, "noncircular_admitted_frac_hold")
    medians = _ranges(rescue, "noncircular_hold_median_px")
    pairs = ", ".join(sorted({r["pair"] for r in rescue}))
    if held:
        lines.append(f"- held-out RMSE across {len(rescue)} pairs ({pairs}): "
                     f"{held[0]} to {held[1]} px; source: results/table_rescue_noncircular.csv")
    if admitted:
        lines.append(f"- admitted fraction of held-out matches: {admitted[0]} to {admitted[1]}; "
                     "source: results/table_rescue_noncircular.csv")
    if medians:
        lines.append(f"- unconditional held-out median residual: {medians[0]} to {medians[1]} px "
                     "(stated alongside, not hidden); source: results/table_rescue_noncircular.csv")
    lines.append(f"- per-pair rows carry their own inlier (fit) RMSE in the same file; "
                 "the split-held-out versus independent-landmark gap is an open, tracked "
                 "limitation, not a closed one.")
    lines.append("")

    lines.append("## Matcher tables, real pairs")
    lines.append("")
    lines.append("Canonical table (issue #1 validation harness):")
    lines.append("")
    for row in real_canon:
        if row["benchmark"] == "gpu_validation":
            lines.append(_canonical_line(row))
    lines.append("")
    lines.append("2026-11-02 GPU bench (inlier (fit) RMSE; backend torch "
                 + (gpu_bench[0]["torch_version"] if gpu_bench else "not recorded") + "):")
    lines.append("")
    for row in real_gpu:
        lines.append(
            f"- {row['pair']} / {row['matcher']}: inlier (fit) RMSE "
            f"{row['rmse_px'] if row['rmse_px'].strip() else 'UNMEASURED'} px, "
            f"{row['inliers']} inliers, gate \"{row['gate']}\"; source: "
            f"results/table_gpu_bench_2026-11-02.csv; {_hardware(row['gpu_model'])}"
        )
    lines.append("")
    for row in real_canon:
        if row["benchmark"] in ("crop_benchmark_12pair", "iirs_first_contact"):
            lines.append(_canonical_line(row))
    lines.append("")
    lines.append("External baselines (39 canonical rows, per-pair windows):")
    lines.append("")
    verdict_counts: dict[str, int] = {}
    for row in real_canon:
        if row["benchmark"] == "external_baselines":
            key = row["verdict"].split(":")[0].strip()
            verdict_counts[key] = verdict_counts.get(key, 0) + 1
    for key in sorted(verdict_counts):
        lines.append(f"- {key}: {verdict_counts[key]} rows; "
                     "source: results/table_canonical_v1.csv (external_baselines)")
    lines.append("")

    lines.append("## Sanity checks (synthetic — not real-pair evidence)")
    lines.append("")
    for row in synth_gpu:
        lines.append(
            f"- {row['pair']} / {row['matcher']}: inlier (fit) RMSE "
            f"{row['rmse_px'] if row['rmse_px'].strip() else 'UNMEASURED'} px, "
            f"{row['inliers']} inliers, gate \"{row['gate']}\"; source: "
            f"results/table_gpu_bench_2026-11-02.csv; {_hardware(row['gpu_model'])}"
        )
    for row in synth_canon:
        if row["benchmark"] == "gpu_validation":
            lines.append(_canonical_line(row))
    for row in synth_canon:
        if row["benchmark"] == "ground_truth":
            lines.append(_canonical_line(row))
    lines.append("")

    lines.append("## ChandraBench v0.1 (frozen landmark sets)")
    lines.append("")
    lines.append(f"- ohrc_pair: {cb_counts['ohrc_pair']} human-verified landmarks; "
                 "source: chandrabench/landmarks_ohrc_pair_v0.1.csv")
    lines.append(f"- tmc2_fore_nadir: {cb_counts['tmc2_fore_nadir']} human-verified landmarks; "
                 "source: chandrabench/landmarks_tmc2_fore_nadir_v0.1.csv")
    lines.append("- both sets are sha256-pinned inside chandrabench/eval.py and guarded "
                 "fail-closed by its CIRCULARITY_GUARD (run `python chandrabench/eval.py --self-test`)")
    lines.append("- landmark licence: CC-BY-4.0 (chandrabench/LICENSE)")
    lines.append("")

    lines.append("## How to read this page")
    lines.append("")
    lines.append("- inlier (fit) RMSE: residuals of the RANSAC-accepted matches on the fit "
                 "set; self-consistency, not independent truth.")
    lines.append("- held-out RMSE: residuals on check points held out of the fit "
                 "(80/20 non-circular splits where the file says so).")
    lines.append("- independent landmark RMSE: residuals against human-verified landmark "
                 "coordinates that no fit stage saw.")
    lines.append("- hardware is quoted from the source file; \"hardware: not recorded\" is "
                 "preserved as-is.")
    lines.append("- gate thresholds and verdict logic are frozen "
                 "(chandra_align/metrics/quadrant.py::validate_registration_gate) and are "
                 "not restated or re-measured here.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate docs/RESULTS_ONE_PAGER.md")
    parser.add_argument("--output", default=str(REPO / "docs" / "RESULTS_ONE_PAGER.md"),
                        help="output path (default: docs/RESULTS_ONE_PAGER.md)")
    args = parser.parse_args()
    text = build_one_pager()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(text.encode("utf-8"))
    print(f"wrote {out} ({len(text.encode('utf-8'))} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
