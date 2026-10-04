"""Generate before/after coarse-to-fine comparisons from the two run records."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT / "results" / "table_issue03_benchmark.csv"
AFTER = ROOT / "results" / "issue04_coarse2fine_runs.json"
OUTPUT = ROOT / "results" / "table_issue04_coarse2fine.csv"
FIELDS = [
    "pair_id", "matcher", "verdict_before", "verdict_after", "verdict_movement",
    "rmse_heldout_before", "rmse_heldout_after", "rmse_delta_px",
    "runtime_before", "runtime_after", "runtime_delta_s",
]
RANK = {"SUCCESS_SUBPIXEL": 0, "COARSE_ADVISORY": 1, "DEGENERATE_FAILURE": 2, "REJECTED": 3}


def fmt(value):
    return "" if value in (None, "", "UNMEASURED") else f"{float(value):.4f}"


def main():
    with BEFORE.open(newline="", encoding="utf-8") as handle:
        before = {(r["pair_id"], r["matcher"]): r for r in csv.DictReader(handle)}
    after = json.loads(AFTER.read_text(encoding="utf-8"))
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in after:
            baseline = before[(row["pair_id"], row["matcher"])]
            old, new = baseline["verdict"], row["verdict_after"]
            movement = "unchanged" if RANK.get(old, 99) == RANK.get(new, 99) else (
                "improved" if RANK.get(new, 99) < RANK.get(old, 99) else "regressed"
            )
            old_rmse, new_rmse = baseline["rmse_heldout"], row["rmse_heldout_after"]
            delta = ""
            if old_rmse not in (None, "") and new_rmse not in (None, ""):
                delta = fmt(float(new_rmse) - float(old_rmse))
            old_runtime = float(baseline["runtime_s"])
            new_runtime = float(row["runtime_after"])
            writer.writerow({
                "pair_id": row["pair_id"], "matcher": row["matcher"],
                "verdict_before": old, "verdict_after": new, "verdict_movement": movement,
                "rmse_heldout_before": fmt(old_rmse), "rmse_heldout_after": fmt(new_rmse),
                "rmse_delta_px": delta, "runtime_before": fmt(old_runtime),
                "runtime_after": fmt(new_runtime), "runtime_delta_s": fmt(new_runtime - old_runtime),
            })
    print(f"Wrote {len(after)} comparisons to {OUTPUT}")


if __name__ == "__main__":
    main()
