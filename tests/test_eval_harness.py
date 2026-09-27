"""Tests for evaluation harness (M7.1-M7.3)."""

import csv
import json
import os
import tempfile

import pytest

from scripts.eval_harness import (build_method_region_table, check_circularity_guard,
                                  collect_run_metrics, emit_method_region_table_markdown,
                                  merge_run_with_log, read_download_log, report_heldout_rmse,
                                  METHODS)


def test_collect_run_metrics(tmp_path):
    """Collect metrics from run directories."""
    runs_dir = tmp_path / "outputs"
    run_dir = runs_dir / "test_run"
    run_dir.mkdir(parents=True)
    metrics = {"trust_flag": "Trusted", "matcher": "rift2"}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
    
    collected = collect_run_metrics(str(runs_dir))
    assert len(collected) == 1
    assert collected[0]["trust_flag"] == "Trusted"
    assert collected[0]["_run_id"] == "test_run"


def test_read_download_log(tmp_path):
    """Read download_log.csv."""
    log_path = tmp_path / "download_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "product_id"])
        writer.writeheader()
        writer.writerow({"run_id": "run1", "product_id": "prod1"})
        writer.writerow({"run_id": "run2", "product_id": "prod2"})
    
    entries = read_download_log(str(log_path))
    assert len(entries) == 2
    assert entries[0]["run_id"] == "run1"


def test_merge_run_with_log():
    """Merge metrics with download log by run_id."""
    metrics_list = [{"_run_id": "run1", "trust_flag": "Trusted"}, {"_run_id": "run2", "trust_flag": "Not Trusted"}]
    log_entries = [{"run_id": "run1", "product_id": "prod1"}, {"run_id": "run2", "product_id": "prod2"}]
    
    merged = merge_run_with_log(metrics_list, log_entries)
    assert merged[0].get("log", {}).get("product_id") == "prod1"
    assert merged[1].get("log", {}).get("product_id") == "prod2"


def test_report_heldout_rmse():
    """Extract held-out RMSE from metrics dict."""
    # With held-out
    m1 = {"rmse": {"held_out": True, "rmse_px": 0.25}}
    assert report_heldout_rmse(m1) == 0.25
    
    # Without held-out
    m2 = {"rmse": {"held_out": False, "rmse_px": "UNMEASURED"}}
    assert report_heldout_rmse(m2) == "UNMEASURED"
    
    # Missing rmse key
    m3 = {}
    assert report_heldout_rmse(m3) == "UNMEASURED"


def test_check_circularity_guard():
    """Verify circularity guard is documented."""
    result = check_circularity_guard({})
    assert result["circularity_guard_enforced"] is True
    assert "fit-set RMSE never computed" in result["note"]


def test_build_method_region_table():
    """Build per-method × per-region table with UNMEASURED defaults."""
    metrics = [
        {"matcher": "rift2", "log": {}, "rmse": {"held_out": True, "rmse_px": 0.3}},
        {"matcher": "lightglue_aliked", "log": {}, "rmse": {"held_out": False, "rmse_px": "UNMEASURED"}},
    ]
    
    table = build_method_region_table(metrics)
    assert "rift2" in table
    assert "lightglue_aliked" in table
    assert "equatorial" in table["rift2"]
    # rift2 has held_out=True so it gets the RMSE value
    assert table["rift2"]["equatorial"] == "0.300"
    # lightglue_aliked has held_out=False so it stays UNMEASURED
    assert table["lightglue_aliked"]["equatorial"] == "UNMEASURED"


def test_emit_method_region_table_markdown():
    """Emit table as markdown."""
    table = {"rift2": {"equatorial": "0.300", "polar": "UNMEASURED"}}
    md = emit_method_region_table_markdown(table)
    assert "| Method | Equatorial RMSE (px) | Polar RMSE (px) |" in md
    assert "| rift2 | 0.300 | UNMEASURED |" in md
    assert "UNMEASURED" in md


if __name__ == "__main__":
    pytest.main([__file__, "-v"])