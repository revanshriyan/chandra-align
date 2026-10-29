"""Tests for dual metrics (payload block, UI display, generator, checker).

The gate never computes landmark RMSE: the independent_landmark block is
populated only by an evaluation path that has landmarks, and it never
feeds the gate or promotes the verdict's evidence_tier.
"""

import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIDECAR = os.path.join(REPO_ROOT, "results", "table_canonical.json")


def _summary(**overrides):
    from chandra_align.metrics.quadrant import build_judge_metrics_summary
    kwargs = dict(
        rmse_pixels=0.30, inlier_count=20, total_correspondences=40,
        spatial_entropy_score=1.50,
        quadrant_metrics={q: {"inlier_count": 5, "rmse_px": 0.30}
                          for q in ("Q1", "Q2", "Q3", "Q4")},
        min_inliers=8,
    )
    kwargs.update(overrides)
    return build_judge_metrics_summary(**kwargs)


def test_payload_block_absent_by_default():
    s = _summary()
    assert s["independent_landmark"] == {
        "rmse_px": None, "reason": "no human-verified landmarks for this pair"}
    assert s["evidence_tier"] == "self_consistency"


def test_payload_block_populated():
    s = _summary(independent_landmark={
        "rmse_px": 1.8869, "rmse_m": 0.47, "n_points": 21,
        "source": "human-verified landmarks (clicker, 2026-10-24)"})
    block = s["independent_landmark"]
    assert block["rmse_px"] == 1.8869
    assert block["rmse_m"] == 0.47
    assert block["n_points"] == 21
    assert block["evidence_tier"] == "independent_landmarks"
    # The block never promotes the verdict's own tier.
    assert s["evidence_tier"] == "self_consistency"


def test_payload_block_never_changes_verdict():
    base = _summary()
    with_block = _summary(independent_landmark={
        "rmse_px": 28.0317, "n_points": 20, "source": "2026-10-15 landmark rerun"})
    for key in ("status_code", "status_message", "registration_status"):
        assert base[key] == with_block[key]


def test_payload_block_malformed_fails_closed():
    for bad in ("not-a-dict", {"rmse_px": float("inf")}, {"rmse_px": "nonsense"}):
        s = _summary(independent_landmark=bad)
        assert s["independent_landmark"]["rmse_px"] is None
        assert "reason" in s["independent_landmark"]
    s = _summary(independent_landmark={
        "rmse_px": None,
        "reason": "landmarks exist but below the protocol floor"})
    assert s["independent_landmark"]["reason"] == (
        "landmarks exist but below the protocol floor")


def test_telemetry_shows_both_figures_side_by_side():
    from app import format_telemetry_report
    telemetry = format_telemetry_report(
        status="SELF_CONSISTENT_SUBPIXEL", inliers=20, total_pts=40,
        inlier_ratio=50.0, spatial_entropy=1.50, quad_counts=[5, 5, 5, 5],
        rmse=0.30,
        independent_landmark={"rmse_px": 1.8869, "rmse_m": 0.47, "n_points": 21,
                              "source": "human-verified landmarks (clicker, 2026-10-24)"},
    )
    assert "Gate RMSE: 0.3000 px" in telemetry
    assert "Independent landmark RMSE: 1.8869 px (21 points, 0.47 m" in telemetry


def test_telemetry_states_reason_when_absent():
    from app import format_telemetry_report
    telemetry = format_telemetry_report(
        status="SELF_CONSISTENT_SUBPIXEL", inliers=20, total_pts=40,
        inlier_ratio=50.0, spatial_entropy=1.50, quad_counts=[5, 5, 5, 5],
        rmse=0.30,
    )
    assert ("Independent landmark RMSE: unavailable — "
            "no human-verified landmarks for this pair") in telemetry
    assert "Gate RMSE: 0.3000 px" in telemetry


def _run_generator():
    proc = subprocess.run(
        [sys.executable,
         os.path.join(REPO_ROOT, "scripts", "generate_results_table.py")],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=300)
    assert proc.returncode in (0, 2), proc.stdout + proc.stderr
    with open(SIDECAR, encoding="utf-8") as f:
        return {e["metric"]: e for e in json.load(f)["entries"]}


def test_generator_dual_landmark_entries():
    by_metric = _run_generator()
    ohrc_01 = by_metric["dual_landmark_ohrc_01_rmse_px"]["independent_landmark"]
    assert ohrc_01["rmse_px"] == 1.8869
    assert ohrc_01["n_points"] == 21
    assert ohrc_01["rmse_m"] == 0.47
    ohrc_04 = by_metric["dual_landmark_ohrc_04_rmse_px"]["independent_landmark"]
    assert ohrc_04["rmse_px"] == 2.2591
    assert ohrc_04["n_points"] == 18
    assert "below the 20-point target" in ohrc_04["floor_note"]
    anchor = by_metric["dual_landmark_ohrc_pair_rmse_px"]["independent_landmark"]
    assert anchor["rmse_px"] == 5.471325
    assert anchor["rmse_m"] is None
    tmc2 = by_metric["dual_landmark_tmc2_fore_nadir_rmse_px"]["independent_landmark"]
    assert tmc2["rmse_px"] == 28.031705
    assert tmc2["rmse_m"] is None
    for key in ("dual_landmark_ohrc_02_rmse_px", "dual_landmark_ohrc_03_rmse_px"):
        block = by_metric[key]["independent_landmark"]
        assert block["rmse_px"] is None
        assert block["reason"] == "no human-verified landmarks for this pair"


def test_generator_dual_range_entries():
    by_metric = _run_generator()
    assert by_metric["dual_heldout_rmse_min_px"]["value"] == "0.2459"
    assert by_metric["dual_heldout_rmse_max_px"]["value"] == "0.3631"
    assert by_metric["dual_admitted_frac_min_pct"]["value"] == "8.16"
    assert by_metric["dual_admitted_frac_max_pct"]["value"] == "32.25"


def test_checker_passes_with_dual_set():
    proc = subprocess.run(
        [sys.executable,
         os.path.join(REPO_ROOT, "scripts", "check_readme_numbers.py")],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "3 dual patterns" in proc.stdout
