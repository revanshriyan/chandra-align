"""Arm-vs-fallback arbitration trial harness (pre-registered 2026-10-23).

For each arm (PC, XoFTR) x pair (ohrc_01..05, tmc2_04), runs _align_core
in three conditions: arm-only (today's behavior), fallback-only (no arm
flag), arbitrated (arm flag + CHANDRA_ARBITRATE=1). Fallback-only is
arm-independent and run once per pair. See
docs/arm-arbitration-prereg-2026-10-23.md for the protocol and win bar.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app

CROPS = ROOT / "data" / "benchmark_crops"
OUT = ROOT / "results" / "table_arbitration_trial_2026-10-23.csv"

PAIRS = [(f"ohrc_{i:02d}", "OHRC", 0.25) for i in range(1, 6)] + [
    ("tmc2_04", "TMC-2", 5.0)
]
ARMS = {"PC": "CHANDRA_PC", "XoFTR": "CHANDRA_XOFTR"}
TIER = {"DEGENERATE_FAILURE": 0, "COARSE_ADVISORY": 1, "SUCCESS_SUBPIXEL": 2}
ARM_FLAGS = ["CHANDRA_PC", "CHANDRA_XOFTR", "CHANDRA_ARBITRATE"]


def run_condition(ref, sec, sensor, scale, arm_flag=None, arbitrate=False):
    for flag in ARM_FLAGS:
        os.environ.pop(flag, None)
    if arm_flag:
        os.environ[arm_flag] = "1"
    if arbitrate:
        os.environ["CHANDRA_ARBITRATE"] = "1"
    start = time.perf_counter()
    res = app._align_core(
        ref, sec, pixel_scale_m=scale, reference_sensor_name=sensor
    )
    diag = res.get("execution_diagnostics") or {}
    arb = diag.get("arbitration") or {}
    return {
        "verdict": res["status_code"],
        "rmse_heldout_px": round(float(res["rmse_heldout_px"]), 4),
        "rmse_in_sample_px": round(float(res["rmse_in_sample_px"]), 4),
        "inliers": int(res["inlier_cnt"]),
        "matcher": diag.get("matcher"),
        "arb_winner": arb.get("winner"),
        "arb_arm_score_px": arb.get("arm_score_px"),
        "arb_fallback_score_px": arb.get("fallback_score_px"),
        "runtime_s": round(time.perf_counter() - start, 2),
    }


def main():
    rows = []
    for pair_id, sensor, scale in PAIRS:
        ref = cv2.imread(str(CROPS / f"{pair_id}_reference.png"), cv2.IMREAD_GRAYSCALE)
        sec = cv2.imread(str(CROPS / f"{pair_id}_source.png"), cv2.IMREAD_GRAYSCALE)
        if ref is None or sec is None:
            raise RuntimeError(f"missing crops for {pair_id}")
        fallback = run_condition(ref, sec, sensor, scale)
        rows.append({"pair_id": pair_id, "arm": "-", "condition": "fallback_only", **fallback})
        print(rows[-1], flush=True)
        for arm_name, arm_flag in ARMS.items():
            arm_only = run_condition(ref, sec, sensor, scale, arm_flag=arm_flag)
            rows.append({"pair_id": pair_id, "arm": arm_name, "condition": "arm_only", **arm_only})
            print(rows[-1], flush=True)
            arbitrated = run_condition(
                ref, sec, sensor, scale, arm_flag=arm_flag, arbitrate=True
            )
            rows.append({"pair_id": pair_id, "arm": arm_name, "condition": "arbitrated", **arbitrated})
            print(rows[-1], flush=True)

    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", OUT)

    def tier(row):
        return TIER[row["verdict"]]

    by_key = {(r["pair_id"], r["arm"], r["condition"]): r for r in rows}
    safety_ok = True
    benefit = 0
    for pair_id, _, _ in PAIRS:
        fb = by_key[(pair_id, "-", "fallback_only")]
        for arm_name in ARMS:
            arm_only = by_key[(pair_id, arm_name, "arm_only")]
            arb = by_key[(pair_id, arm_name, "arbitrated")]
            if tier(arb) < tier(arm_only) or tier(arb) < tier(fb):
                safety_ok = False
                print("SAFETY FAIL", pair_id, arm_name, arm_only["verdict"], fb["verdict"], arb["verdict"])
            if pair_id.startswith("ohrc") and tier(arb) > tier(arm_only):
                benefit += 1
    rescue = by_key[("tmc2_04", "XoFTR", "arbitrated")]
    rescue_ok = TIER[rescue["verdict"]] >= TIER["COARSE_ADVISORY"]
    print("safety (arb >= both parents everywhere):", safety_ok)
    print("benefit cells (ohrc, arb > arm-only):", benefit, "of 10; bar >= 3")
    print("rescue preserved (tmc2_04 XoFTR arbitrated):", rescue["verdict"], rescue_ok)
    print("WIN BAR MET:", safety_ok and benefit >= 3 and rescue_ok)


if __name__ == "__main__":
    main()
