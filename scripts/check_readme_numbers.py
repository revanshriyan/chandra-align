"""Verify that headline numbers cited in README.md match the canonical table.

Run: python scripts/check_readme_numbers.py
Exit 0 when every checked number matches results/table_canonical_v1.csv
within tolerance; exit 1 with a diff otherwise. Wired into CI.

Only checks numbers the README states as measured facts. Thresholds and
prose are not checked here.
"""

import csv
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
    (r"synthetic sub-pixel ACCEPT at ([\d.]+) px / (\d+) inliers",
     "gpu_validation", "synthetic_gentle", "LightGlue_ALIKED",
     [("rmse_px", 0), ("inliers", 1)], "synthetic LightGlue"),
    (r"competitive real-pair results at ([\d.]+) px \(OHRC\) and ([\d.]+) px \(TMC-2\)",
     None, None, None, None, "real-pair headline"),
    (r"OHRC pair — (\d+) inliers across all four quadrants at ([\d.]+) px",
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
        r"competitive real-pair results at ([\d.]+) px \(OHRC\) and ([\d.]+) px \(TMC-2\)", text
    )
    if match:
        for readme_val, pair in ((match.group(1), "OHRC_pair"), (match.group(2), "TMC2_fore_nadir")):
            row = index[("gpu_validation", pair, "LightGlue_ALIKED")]
            if abs(float(readme_val) - float(row["rmse_px"])) > TOL:
                failures.append(f"[real-pair] {pair}: README {readme_val} != {row['rmse_px']}")

    # Issue #3 aggregates cited in prose.
    for prose, matcher, verdict in [
        (r"LightGlue/ALIKED produced (\d+) COARSE and (\d+) DEGENERATE_FAILURE",
         "LightGlue_ALIKED", ("COARSE_ADVISORY", "DEGENERATE_FAILURE")),
        (r"SIFT/RANSAC produced (\d+) COARSE and (\d+) DEGENERATE_FAILURE",
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

    if failures:
        print("README number check FAILED:")
        for failure in failures:
            print("  -", failure)
        return 1
    print(f"README number check passed ({len(CHECKS)} patterns).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
