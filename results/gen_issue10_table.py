"""Join the Issue #3 baseline and Issue #10 run into the comparison CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AFTER = ROOT / "results" / "issue10_bucketing_runs.json"
OUTPUT = ROOT / "results" / "table_issue10_bucketing.csv"
FIELDS = [
    "pair_id", "sensor", "matcher", "verdict_before", "verdict_after",
    "rmse_heldout_before", "rmse_heldout_after", "quadrants_before",
    "quadrants_after", "runtime_before_s", "runtime_after_s",
    "correspondences_before", "correspondences_after", "inliers_before", "inliers_after",
    "entropy_before", "entropy_after",
    "final_engine_before", "final_engine_after", "secondary_fallback_before", "secondary_fallback_after",
]


def main() -> None:
    after_rows = json.loads(AFTER.read_text(encoding="utf-8"))
    before = {(row["pair_id"], row["matcher"]): row for row in after_rows if row["variant"] == "before"}
    after = {(row["pair_id"], row["matcher"]): row for row in after_rows if row["variant"] == "after"}
    if len(before) != 24 or len(after) != 24 or before.keys() != after.keys():
        raise RuntimeError("Expected 12 identical crop pairs × 2 matchers × before/after runs")
    rows = []
    for key in sorted(before):
        old, new = before[key], after[key]
        rows.append({
            "pair_id": key[0], "sensor": old["sensor"], "matcher": key[1],
            "verdict_before": old["verdict"], "verdict_after": new["verdict"],
            "rmse_heldout_before": old["rmse_heldout"],
            "rmse_heldout_after": new["rmse_heldout"],
            "quadrants_before": f"{sum(int(v) > 0 for v in old['quadrant_counts'])}/4 ({','.join(map(str, old['quadrant_counts']))})",
            "quadrants_after": f"{sum(int(v) > 0 for v in new['quadrant_counts'])}/4 ({','.join(map(str, new['quadrant_counts']))})",
            "runtime_before_s": old["runtime_s"], "runtime_after_s": f"{new['runtime_s']:.4f}",
            "correspondences_before": old["correspondences"],
            "correspondences_after": new["correspondences"],
            "inliers_before": old["inliers"], "inliers_after": new["inliers"],
            "entropy_before": old["entropy"], "entropy_after": new["entropy"],
            "final_engine_before": old["final_engine"], "final_engine_after": new["final_engine"],
            "secondary_fallback_before": old["secondary_fallback"],
            "secondary_fallback_after": new["secondary_fallback"],
        })
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUTPUT} ({len(rows)} matcher-pair rows)")


if __name__ == "__main__":
    main()
