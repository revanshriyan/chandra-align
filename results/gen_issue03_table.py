"""Generate the Issue #3 CSV table from recorded matcher runs and crop manifest."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "issue03_benchmark_runs.json"
MANIFEST = ROOT / "data" / "benchmark_pairs.csv"
OUTPUT = ROOT / "results" / "table_issue03_benchmark.csv"
FIELDS = [
    "pair_id", "sensor", "terrain", "matcher", "correspondences",
    "rmse_insample", "rmse_heldout", "n_heldout_points", "inliers",
    "entropy", "quadrants", "verdict", "runtime_s", "gt_rmse", "notes",
    "confidence_score", "decision_reason", "fallback_path",
]


def main() -> None:
    runs = json.loads(RAW.read_text(encoding="utf-8"))
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        metadata = {row["pair_id"]: row for row in csv.DictReader(handle)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for run in runs:
            manifest = metadata[run["pair_id"]]
            counts = run.get("quadrant_counts", [0, 0, 0, 0])
            active = sum(int(value) > 0 for value in counts)
            def fmt(value):
                return "" if value in (None, "") else f"{float(value):.4f}"
            row = {
                "pair_id": run["pair_id"], "sensor": run["sensor"],
                "terrain": manifest["terrain_note"], "matcher": run["matcher"],
                "correspondences": run["correspondences"],
                "rmse_insample": fmt(run["rmse_insample"]),
                "rmse_heldout": fmt(run["rmse_heldout"]),
                "n_heldout_points": run["n_heldout_points"], "inliers": run["inliers"],
                "entropy": f"{float(run['entropy']):.4f}",
                "quadrants": f"{active}/4 ({','.join(str(int(v)) for v in counts)})",
                "verdict": run["verdict"], "runtime_s": f"{float(run['runtime_s']):.4f}",
                "gt_rmse": run["gt_rmse"], "notes": run["notes"],
                "confidence_score": f"{float(run.get('confidence_score', 0.0)):.2f}",
                "decision_reason": run.get("decision_reason", ""),
                "fallback_path": json.dumps(run.get("fallback_path", {}), sort_keys=True),
            }
            writer.writerow(row)
    print(f"Wrote {len(runs)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
