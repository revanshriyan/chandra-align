#!/usr/bin/env python3
"""Regenerate the Issue #2 ground-truth comparison CSV from run output."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "issue13_gt_matcher_runs.json"
OUTPUT = ROOT / "results" / "table_issue02_ground_truth.csv"
FIELDS = [
    "pair", "matcher", "rmse_heldout_split", "rmse_ground_truth",
    "n_gt_points", "gt_source", "verdict_agrees",
]


def finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def gt_rmse_class(value: float | None) -> str:
    if value is None:
        return "UNMEASURED"
    if value <= 0.5:
        return "ACCEPTED"
    if value <= 2.5:
        return "COARSE_ADVISORY"
    return "REJECTED"


def split_class(status: str) -> str:
    normalized = status.upper()
    if normalized in {"SUCCESS", "SUCCESS_SUBPIXEL", "ACCEPTED"}:
        return "ACCEPTED"
    if normalized in {"COARSE_ADVISORY", "COARSE ALIGNMENT (REGIONAL FIT ADVISORY)"}:
        return "COARSE_ADVISORY"
    if normalized in {"", "UNMEASURED", "UNKNOWN"}:
        return "UNMEASURED"
    return "REJECTED"


def render() -> list[dict[str, str]]:
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    rows = [dict(row) for row in payload.get("previous_synthetic_rows", [])]
    for pair, slug in (("OHRC_pair", "ohrc_pair"), ("TMC2_fore_nadir", "tmc2_fore_nadir")):
        rows.append({
            "pair": pair,
            "matcher": "RIFT2",
            "rmse_heldout_split": "",
            "rmse_ground_truth": "",
            "n_gt_points": "20",
            "gt_source": f"data/ground_truth/gt_{slug}_human_verified.csv (RIFT2 produced no fit)",
            "verdict_agrees": "UNMEASURED_NO_FIT",
        })
    for run in payload["runs"]:
        pair = run["pair"]
        matcher = run["matcher"]
        source = f"data/ground_truth/gt_{'ohrc_pair' if pair == 'OHRC_pair' else 'tmc2_fore_nadir'}_human_verified.csv"
        if not run.get("fit_available") or run.get("circularity_guard") != "PASS":
            rmse_gt = None
            agreement = "UNMEASURED_NO_FIT" if not run.get("fit_available") else "GUARD_FAILED"
        else:
            rmse_gt = finite_number(run.get("ground_truth_rmse_px"))
            split_verdict = split_class(str(run.get("split_verdict", "")))
            candidate = gt_rmse_class(rmse_gt)
            agreement = (
                "YES" if candidate == split_verdict else "NO"
            ) if candidate != "UNMEASURED" and split_verdict != "UNMEASURED" else "UNMEASURED"
        matcher_label = matcher
        if matcher == "LightGlue_ALIKED" and run.get("fallback_triggered"):
            matcher_label += " (SIFT fallback used)"
        rows.append({
            "pair": pair,
            "matcher": matcher_label,
            "rmse_heldout_split": "" if finite_number(run.get("rmse_heldout_split")) is None
            else f"{finite_number(run['rmse_heldout_split']):.10f}",
            "rmse_ground_truth": "" if rmse_gt is None else f"{rmse_gt:.6f}",
            "n_gt_points": str(run.get("n_gt_points", 0)),
            "gt_source": source + " (human-reviewed NCC ties; ~0.5-1 px localization noise)",
            "verdict_agrees": agreement,
        })
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    generated = render()
    print(f"Wrote {OUTPUT} ({len(generated)} rows)")
