"""SYNTHETIC_VALIDATION test for analyze_heldout.py

Validates that analyze_heldout.py correctly recovers known synthetic displacement shifts within 0.05 px,
and strictly enforces row count and column presence guards.
"""

import csv
import pytest
from pathlib import Path
from scripts.analyze_heldout import analyze_heldout_pair, parse_csv_points


def test_synthetic_validation_known_shift(tmp_path):
    """[SYNTHETIC_VALIDATION] Assert recovered RMSE matches known shift within 0.05 px."""
    ref_csv = tmp_path / "heldout_ref.csv"
    prod_csv = tmp_path / "heldout_prod.csv"

    # Known shift: dx = +0.3 px, dy = +0.4 px -> Expected Total RMSE = 0.5 px
    ref_points = [
        (100.0, 100.0, 1, "crater_rim"),
        (200.0, 200.0, 1, "boulder"),
        (300.0, 300.0, 2, "crater_center"),
        (400.0, 400.0, 2, "ridge"),
        (500.0, 500.0, 3, "crater_rim"),
    ]

    with open(ref_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y", "confidence", "feature_type"])
        for x, y, conf, ft in ref_points:
            writer.writerow([x, y, conf, ft])

    with open(prod_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y", "confidence", "feature_type"])
        for x, y, conf, ft in ref_points:
            writer.writerow([x + 0.3, y + 0.4, conf, ft])

    summary, rpath = analyze_heldout_pair(ref_csv, prod_csv, gsd_m=0.25, pair_id="synthetic_test")

    recovered_rmse = summary["overall"]["rmse_total_px"]
    expected_rmse = 0.500

    # SYNTHETIC_VALIDATION assertion: recovered RMSE within 0.05 px of known shift
    assert abs(recovered_rmse - expected_rmse) <= 0.05, (
        f"[SYNTHETIC_VALIDATION FAIL] Expected RMSE {expected_rmse} px, got {recovered_rmse} px"
    )
    assert Path(rpath).exists()
    assert Path(summary["scatter_plot"]).exists()


def test_guard_mismatched_row_counts(tmp_path):
    """Assert guard refuses to run when CSV row counts differ."""
    ref_csv = tmp_path / "heldout_ref.csv"
    prod_csv = tmp_path / "heldout_prod.csv"

    with open(ref_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y", "confidence", "feature_type"])
        writer.writerow([100.0, 100.0, 1, "crater_rim"])
        writer.writerow([200.0, 200.0, 1, "boulder"])

    with open(prod_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y", "confidence", "feature_type"])
        writer.writerow([100.3, 100.4, 1, "crater_rim"])  # Only 1 row

    with pytest.raises(ValueError, match="Row count mismatch error"):
        analyze_heldout_pair(ref_csv, prod_csv, gsd_m=0.25, pair_id="mismatch_test")


def test_guard_missing_confidence_column(tmp_path):
    """Assert guard refuses to run when 'confidence' column is missing."""
    ref_csv = tmp_path / "heldout_ref.csv"
    prod_csv = tmp_path / "heldout_prod.csv"

    with open(ref_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y", "feature_type"])  # No confidence
        writer.writerow([100.0, 100.0, "crater_rim"])

    with open(prod_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y", "feature_type"])
        writer.writerow([100.3, 100.4, "crater_rim"])

    with pytest.raises(ValueError, match="Missing required 'confidence' column"):
        analyze_heldout_pair(ref_csv, prod_csv, gsd_m=0.25, pair_id="noconf_test")
