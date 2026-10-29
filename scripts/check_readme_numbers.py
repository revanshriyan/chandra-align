"""Verify that headline numbers cited in README.md match the canonical table.

Run: python scripts/check_readme_numbers.py
Exit 0 when every checked number matches results/table_canonical_v1.csv
within tolerance; exit 1 with a diff otherwise. Wired into CI.

Only checks numbers the README states as measured facts. Thresholds and
prose are not checked here.
"""

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
CANONICAL = ROOT / "results" / "table_canonical_v1.csv"

TOL = 1e-3


def load_canonical():
    with CANONICAL.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    index = {}
    for row in rows:
        index[(row["benchmark"], row["pair"], row["matcher"])] = row
    return index


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# (readme_regex, benchmark, pair, matcher, canonical_column, label)
CHECKS = [
    (r"synthetic ACCEPTED \((?:Sub-Pixel Precision|Self-Consistent, Sub-Pixel)\) at inlier RMSE ([\d.]+) px / (\d+) inliers",
     "gpu_validation", "synthetic_gentle", "LightGlue_ALIKED",
     [("rmse_px", 0), ("inliers", 1)], "synthetic LightGlue"),
    (r"competitive real-pair results at inlier RMSE ([\d.]+) px \(OHRC\) and ([\d.]+) px \(TMC-2\)",
     None, None, None, None, "real-pair headline"),
    (r"OHRC pair — (\d+) inliers across all four quadrants at inlier RMSE ([\d.]+) px",
     "gpu_validation", "OHRC_pair", "LightGlue_ALIKED",
     [("inliers", 0), ("rmse_px", 1)], "OHRC LightGlue"),
    (r"synthetic_gentle \| LightGlue_ALIKED \| \d+ \| ([\d.]+) \| (\d+)",
     "gpu_validation", "synthetic_gentle", "LightGlue_ALIKED",
     [("rmse_px", 0), ("inliers", 1)], "table synthetic LightGlue"),
    (r"OHRC_pair \| LightGlue_ALIKED \| \d+ \| ([\d.]+) \| (\d+)",
     "gpu_validation", "OHRC_pair", "LightGlue_ALIKED",
     [("rmse_px", 0), ("inliers", 1)], "table OHRC LightGlue"),
    (r"TMC2_fore_nadir \| LightGlue_ALIKED \| \d+ \| ([\d.]+) \| (\d+)",
     "gpu_validation", "TMC2_fore_nadir", "LightGlue_ALIKED",
     [("rmse_px", 0), ("inliers", 1)], "table TMC-2 LightGlue"),
    (r"OHRC_pair \| SIFT_RANSAC \| \d+ \| ([\d.]+) \| (\d+)",
     "gpu_validation", "OHRC_pair", "SIFT_RANSAC",
     [("rmse_px", 0), ("inliers", 1)], "table OHRC SIFT"),
    (r"TMC2_fore_nadir \| SIFT_RANSAC \| \d+ \| ([\d.]+) \| (\d+)",
     "gpu_validation", "TMC2_fore_nadir", "SIFT_RANSAC",
     [("rmse_px", 0), ("inliers", 1)], "table TMC-2 SIFT"),
]


def main():
    text = README.read_text(encoding="utf-8")
    index = load_canonical()
    failures = []

    for pattern, benchmark, pair, matcher, cols, label in CHECKS:
        match = re.search(pattern, text)
        if not match:
            failures.append(f"[{label}] pattern not found in README: {pattern[:60]}")
            continue
        if benchmark is None:  # multi-value prose check handled below
            continue
        row = index.get((benchmark, pair, matcher))
        if row is None:
            failures.append(f"[{label}] no canonical row for {(benchmark, pair, matcher)}")
            continue
        for column, group in cols:
            readme_val = num(match.group(group + 1))
            canon_val = num(row[column])
            if readme_val is None or canon_val is None:
                failures.append(f"[{label}] non-numeric {column}")
            elif abs(readme_val - canon_val) > TOL:
                failures.append(
                    f"[{label}] README {column}={readme_val} != canonical {canon_val}"
                )

    # Real-pair headline: "1.7961 px (OHRC) and 1.5558 px (TMC-2)"
    match = re.search(
        r"competitive real-pair results at inlier RMSE ([\d.]+) px \(OHRC\) and ([\d.]+) px \(TMC-2\)", text
    )
    if match:
        for readme_val, pair in ((match.group(1), "OHRC_pair"), (match.group(2), "TMC2_fore_nadir")):
            row = index[("gpu_validation", pair, "LightGlue_ALIKED")]
            if abs(float(readme_val) - float(row["rmse_px"])) > TOL:
                failures.append(f"[real-pair] {pair}: README {readme_val} != {row['rmse_px']}")

    # Issue #3 aggregates cited in prose.
    for prose, matcher, verdict in [
        (r"LightGlue/ALIKED produced (\d+) COARSE_ADVISORY and (\d+) DEGENERATE_FAILURE",
         "LightGlue_ALIKED", ("COARSE_ADVISORY", "DEGENERATE_FAILURE")),
        (r"SIFT/RANSAC produced (\d+) COARSE_ADVISORY and (\d+) DEGENERATE_FAILURE",
         "SIFT_RANSAC", ("COARSE_ADVISORY", "DEGENERATE_FAILURE")),
    ]:
        match = re.search(prose, text)
        if match:
            row = index.get(("crop_benchmark_12pair",
                             "12 south-polar pairs (6 OHRC + 6 TMC-2)", matcher))
            if row:
                for readme_val, v in ((match.group(1), verdict[0]), (match.group(2), verdict[1])):
                    expected = f"{v}: {readme_val}"
                    if expected not in row["verdict"]:
                        failures.append(f"[issue03 {matcher}] '{expected}' not in canonical '{row['verdict']}'")
            else:
                failures.append(f"[issue03] no canonical aggregate for {matcher}")
        # if the prose pattern is absent, nothing to check

    # Dual-metrics check-set: new headline figures vs the generator sidecar.
    # The v1 checks above stay untouched; this set guards only the figures
    # the dual-metrics headline rewrite added (independent landmark RMSEs
    # for ohrc_01/ohrc_04, the exact held-out range, the exact admitted
    # fractions). Same 1e-3 tolerance, same fail-closed exit 1. The sidecar
    # must be regenerated first (scripts/generate_results_table.py); a
    # stale sidecar without the dual entries fails here rather than passing
    # silently.
    sidecar_path = ROOT / "results" / "table_canonical.json"
    by_metric = {}
    if not sidecar_path.exists():
        failures.append("[dual] generator sidecar missing: run scripts/generate_results_table.py first")
    else:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        by_metric = {e["metric"]: e for e in sidecar.get("entries", [])}

    dual_checks = [
        (r"ohrc_01 ([\d.]+) px \((\d+) points",
         "dual_landmark_ohrc_01_rmse_px", "landmark", "ohrc_01 landmark"),
        (r"ohrc_04 ([\d.]+) px \((\d+) points",
         "dual_landmark_ohrc_04_rmse_px", "landmark", "ohrc_04 landmark"),
        (r"Non-circular 80/20 holdout verification on all 9 pairs: \*\*([\d.]+)–([\d.]+) px\*\*, admitted fractions ([\d.]+)–([\d.]+)%",
         ("dual_heldout_rmse_min_px", "dual_heldout_rmse_max_px",
          "dual_admitted_frac_min_pct", "dual_admitted_frac_max_pct"),
         "range", "held-out + admitted"),
    ]
    for pattern, metric, kind, label in dual_checks:
        match = re.search(pattern, text)
        if not match:
            failures.append(f"[dual {label}] pattern not found in README")
            continue
        if kind == "landmark":
            block = by_metric.get(metric, {}).get("independent_landmark") or {}
            readme_rmse = num(match.group(1))
            readme_n = num(match.group(2))
            side_rmse = num(block.get("rmse_px"))
            side_n = block.get("n_points")
            if readme_rmse is None or side_rmse is None or abs(readme_rmse - side_rmse) > TOL:
                failures.append(
                    f"[dual {label}] README rmse={match.group(1)} != sidecar {block.get('rmse_px')}"
                )
            if readme_n is None or side_n is None or int(readme_n) != int(side_n):
                failures.append(
                    f"[dual {label}] README n_points={match.group(2)} != sidecar {side_n}"
                )
        else:
            for group, metric_name in enumerate(metric, start=1):
                readme_val = num(match.group(group))
                side_val = num(by_metric.get(metric_name, {}).get("value"))
                if readme_val is None or side_val is None:
                    failures.append(f"[dual {label}] non-numeric {metric_name}")
                elif abs(readme_val - side_val) > TOL:
                    failures.append(
                        f"[dual {label}] README {metric_name}={readme_val} != sidecar {side_val}"
                    )

    if failures:
        print("README number check FAILED:")
        for failure in failures:
            print("  -", failure)
        return 1
    print(f"README number check passed ({len(CHECKS)} patterns + {len(dual_checks)} dual patterns).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
