"""Coarse-to-fine re-run harness (pre-registered 2026-10-23).

Runs app._align_core in single_scale vs coarse_to_fine mode on the 12
benchmark crop pairs and writes a before/after table in the issue #4
layout. See docs/coarse2fine-rerun-prereg-2026-10-23.md for the protocol
and the win bar, fixed before this script was run.

CPU-only here: the LightGlue/ALIKED primary falls back to SIFT, so both
arms are SIFT-backed. That is recorded in the output, not hidden.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app
CROPS = ROOT / "data" / "benchmark_crops"
OUT = ROOT / "results" / "table_coarse2fine_rerun_2026-10-23.csv"

PAIRS = (
    [(f"ohrc_{i:02d}", "OHRC", 0.25) for i in range(1, 7)]
    + [(f"tmc2_{i:02d}", "TMC-2", 5.0) for i in range(1, 7)]
)

TIER = {"DEGENERATE_FAILURE": 0, "COARSE_ADVISORY": 1, "SUCCESS_SUBPIXEL": 2}


def run_arm(ref, sec, sensor, scale, mode):
    start = time.perf_counter()
    res = app._align_core(
        ref,
        sec,
        pixel_scale_m=scale,
        reference_sensor_name=sensor,
        matching_mode=mode,
    )
    return res, time.perf_counter() - start


def main():
    rows = []
    for pair_id, sensor, scale in PAIRS:
        ref = cv2.imread(str(CROPS / f"{pair_id}_reference.png"), cv2.IMREAD_GRAYSCALE)
        sec = cv2.imread(str(CROPS / f"{pair_id}_source.png"), cv2.IMREAD_GRAYSCALE)
        if ref is None or sec is None:
            raise RuntimeError(f"missing crops for {pair_id}")
        before, t_before = run_arm(ref, sec, sensor, scale, "single_scale")
        after, t_after = run_arm(ref, sec, sensor, scale, "coarse_to_fine")
        movement = "unchanged"
        if TIER[after["status_code"]] > TIER[before["status_code"]]:
            movement = "improved"
        elif TIER[after["status_code"]] < TIER[before["status_code"]]:
            movement = "regressed"
        row = {
            "pair_id": pair_id,
            "engine_before": before.get("engine_name") or before.get("engine_used"),
            "engine_after": after.get("engine_name") or after.get("engine_used"),
            "verdict_before": before["status_code"],
            "verdict_after": after["status_code"],
            "verdict_movement": movement,
            "rmse_heldout_before": round(float(before["rmse_heldout_px"]), 4),
            "rmse_heldout_after": round(float(after["rmse_heldout_px"]), 4),
            "rmse_delta_px": round(
                float(after["rmse_heldout_px"]) - float(before["rmse_heldout_px"]), 4
            ),
            "rmse_in_sample_before": round(float(before["rmse_in_sample_px"]), 4),
            "rmse_in_sample_after": round(float(after["rmse_in_sample_px"]), 4),
            "inliers_before": int(before["inlier_cnt"]),
            "inliers_after": int(after["inlier_cnt"]),
            "runtime_before_s": round(t_before, 2),
            "runtime_after_s": round(t_after, 2),
        }
        rows.append(row)
        print(row, flush=True)

    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = {"improved": 0, "regressed": 0, "unchanged": 0}
    for row in rows:
        counts[row["verdict_movement"]] += 1
    print("movement counts:", counts)
    print("win bar (>=10 improved AND 0 regressed):",
          counts["improved"] >= 10 and counts["regressed"] == 0)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
