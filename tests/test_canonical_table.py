"""Tests for the canonical results-table generator (issue #8).

The generator is the single source of truth for quoted numbers. These tests
fail loudly if a source file goes missing or a core metric becomes
UNVERIFIED, so CI catches drift.
"""

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "generate_results_table.py")
SIDECAR = os.path.join(REPO_ROOT, "results", "table_canonical.json")

# The 8 core metric groups from the issue #8 spec; every one must verify.
CORE_METRIC_PREFIXES = [
    "synthetic_lightglue_",
    "ohrc_",
    "tmc2_",
    "loftr_",
    "phase13_",
    "phase15_",
    "chandrabench_",
    "gate_",
]


def _run_generator():
    return subprocess.run(
        [sys.executable, SCRIPT],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=300,
    )


def _load_sidecar():
    with open(SIDECAR, encoding="utf-8") as f:
        return json.load(f)


def test_generator_runs_without_error():
    proc = _run_generator()
    assert proc.returncode in (0, 2), (
        f"generator crashed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "| Metric | Value | Source |" in proc.stdout, "markdown table header missing from stdout"
    assert os.path.exists(SIDECAR), "JSON sidecar was not written"


def test_sidecar_valid_and_every_entry_has_source():
    _run_generator()
    data = _load_sidecar()
    assert "entries" in data and isinstance(data["entries"], list)
    assert data["entries"], "sidecar has no entries"
    for e in data["entries"]:
        for key in ("metric", "label", "value", "source", "verified"):
            assert key in e, f"entry missing key {key!r}: {e}"
        assert isinstance(e["source"], str) and e["source"].strip(), (
            f"entry has empty source: {e['metric']}"
        )
        assert e["verified"] == (e["value"] != "UNVERIFIED")


def test_no_unverified_core_metrics():
    _run_generator()
    data = _load_sidecar()
    bad = [e["metric"] for e in data["entries"] if not e["verified"]]
    core_bad = [m for m in bad
                if any(m.startswith(p) for p in CORE_METRIC_PREFIXES)]
    assert not core_bad, (
        "UNVERIFIED core metrics (a source file is missing or a number "
        f"could not be found): {core_bad}"
    )


def test_phase13_medians_recomputed_not_hardcoded():
    """Spot-check that Phase 13 medians match an independent recomputation."""
    import csv
    import statistics

    _run_generator()
    data = _load_sidecar()
    by_metric = {e["metric"]: e["value"] for e in data["entries"]}
    path = os.path.join(REPO_ROOT, "results", "table_phase13_windows.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for tag, pair in [("ohrc", "ohrc_pair"), ("tmc2", "tmc2_fore_nadir")]:
        coarse = [r for r in rows if r["pair"] == pair and r["verdict"] == "COARSE_ADVISORY"]
        med_inl = statistics.median(float(r["n_inliers_unique"]) for r in coarse)
        med_rmse = statistics.median(float(r["rmse_px"]) for r in coarse)
        assert by_metric[f"phase13_{tag}_median_inliers"] == f"{med_inl:.0f}"
        assert by_metric[f"phase13_{tag}_median_rmse"] == f"{med_rmse:.2f}"
