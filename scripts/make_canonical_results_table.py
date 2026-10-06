"""Build the single versioned canonical results table from the per-issue tables.

Reads the headline per-issue CSVs under results/ and writes one unified,
versioned table: results/table_canonical_v1.csv plus a provenance sidecar
results/table_canonical_v1.meta.json.

The canonical table is the source of truth for every headline number cited
in README.md. scripts/check_readme_numbers.py verifies the README against it.
Hand-editing the CSV is not allowed; re-run this script instead.
"""

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

TABLE_VERSION = "v1"
RESULTS = Path(__file__).resolve().parent.parent / "results"

# Unified schema. Columns absent from a source table stay empty.
COLUMNS = [
    "table_version",
    "benchmark",
    "pair",
    "matcher",
    "correspondences",
    "inliers",
    "rmse_px",
    "rmse_heldout_px",
    "gt_rmse_px",
    "entropy",
    "quadrants",
    "verdict",
    "runtime_s",
    "hardware",
    "source_table",
    "notes",
]


def _read(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row(**kwargs):
    row = {key: "" for key in COLUMNS}
    row["table_version"] = TABLE_VERSION
    row.update({k: ("" if v is None else str(v)) for k, v in kwargs.items()})
    return row


def _issue01():
    rows = []
    for src in _read("table_issue01_gpu_validation.csv"):
        rows.append(
            _row(
                benchmark="gpu_validation",
                pair=src["pair"],
                matcher=src["matcher"],
                correspondences=src["correspondences"],
                inliers=src["inliers"],
                rmse_px=src["rmse_px"],
                entropy=src["entropy"],
                quadrants=src["quadrants"],
                verdict=src["gate"],
                runtime_s=src["runtime_s"],
                hardware="NVIDIA RTX 5070",
                source_table="table_issue01_gpu_validation.csv",
            )
        )
    return rows


def _issue02():
    rows = []
    for src in _read("table_issue02_ground_truth.csv"):
        rows.append(
            _row(
                benchmark="ground_truth",
                pair=src["pair"],
                matcher=src["matcher"],
                rmse_heldout_px=src["rmse_heldout_split"],
                gt_rmse_px=src["rmse_ground_truth"],
                verdict=src["verdict_agrees"],
                source_table="table_issue02_ground_truth.csv",
                notes=f"n_gt_points={src['n_gt_points']}; {src['gt_source']}",
            )
        )
    return rows


def _issue03():
    """Headline aggregates only; per-pair rows live in the source table."""
    counts = {}
    for src in _read("table_issue03_benchmark.csv"):
        key = src["matcher"]
        counts.setdefault(key, {}).setdefault(src["verdict"], 0)
        counts[key][src["verdict"]] += 1
    rows = []
    for matcher, verdicts in sorted(counts.items()):
        summary = ", ".join(f"{v}: {c}" for v, c in sorted(verdicts.items()))
        rows.append(
            _row(
                benchmark="crop_benchmark_12pair",
                pair="12 south-polar pairs (6 OHRC + 6 TMC-2)",
                matcher=matcher,
                verdict=summary,
                source_table="table_issue03_benchmark.csv",
                notes="No row ACCEPTED; gt_rmse empty pending independent landmarks; windows do not cover mare terrain.",
            )
        )
    return rows


def _issue05():
    rows = []
    for src in _read("table_issue05_baselines.csv"):
        rows.append(
            _row(
                benchmark="external_baselines",
                pair=src["pair"],
                matcher=src["matcher"],
                correspondences=src["correspondences"],
                inliers=src["inliers"],
                rmse_px=src["rmse_px"],
                rmse_heldout_px=src["rmse_heldout_px"],
                verdict=src["verdict"],
                runtime_s=src["runtime_s"],
                source_table="table_issue05_baselines.csv",
                notes=(src.get("notes") or "").strip(),
            )
        )
    return rows


def _issue12():
    rows = []
    for src in _read("table_issue12_iirs.csv"):
        rows.append(
            _row(
                benchmark="iirs_first_contact",
                pair=src["pair_id"],
                matcher=src["matcher"],
                correspondences=src["correspondences"],
                inliers=src["inliers"],
                rmse_px=src["rmse_insample_px"],
                rmse_heldout_px=src["rmse_heldout_px"],
                quadrants=src["quadrants"],
                verdict=src["verdict"],
                runtime_s=src["runtime_s"],
                source_table="table_issue12_iirs.csv",
                notes=f"IIRS {src['iirs_band_nm']} nm vs {src['tmc_product']}; {src['notes']}",
            )
        )
    return rows


def _source_commit():
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=RESULTS.parent,
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
        )
    except Exception:
        return "unrecorded"


def main():
    rows = _issue01() + _issue02() + _issue03() + _issue05() + _issue12()
    out_csv = RESULTS / f"table_canonical_{TABLE_VERSION}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "table_version": TABLE_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_commit": _source_commit(),
        "generator": "scripts/make_canonical_results_table.py",
        "row_count": len(rows),
        "source_tables": [
            "table_issue01_gpu_validation.csv",
            "table_issue02_ground_truth.csv",
            "table_issue03_benchmark.csv",
            "table_issue05_baselines.csv",
            "table_issue12_iirs.csv",
        ],
    }
    with (RESULTS / f"table_canonical_{TABLE_VERSION}.meta.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(meta, handle, indent=2)
    print(f"wrote {out_csv.name}: {len(rows)} rows (v{TABLE_VERSION})")


if __name__ == "__main__":
    main()
