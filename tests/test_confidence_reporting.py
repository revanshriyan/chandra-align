from chandra_align.metrics.reporting import build_confidence_assessment


def _assessment(status):
    return build_confidence_assessment(
        status_code=status,
        status_message="Gate rejected: low spatial spread",
        rmse_px=0.4 if status == "SUCCESS_SUBPIXEL" else 1.2,
        inlier_count=20,
        correspondence_count=25,
        entropy=1.4,
        quadrant_counts={f"Q{i}": {"inlier_count": 5} for i in range(1, 5)},
        min_inliers=8,
        execution_diagnostics={
            "primary_engine": "LightGlue/ALIKED",
            "registration_engine": "SIFT/RANSAC",
            "fallback_triggered": True,
            "fallback_reason": "primary failed",
            "fallback_error": "test failure",
        },
    )


def test_assessment_has_fields_for_accepted_coarse_and_rejected_verdicts():
    for verdict in ("SUCCESS_SUBPIXEL", "COARSE_ADVISORY", "DEGENERATE_FAILURE"):
        assessment = _assessment(verdict)
        assert 0 <= assessment["confidence_score"] <= 100
        assert "not a calibrated probability" in assessment["confidence_basis"]
        assert assessment["decision_reason"]
        assert assessment["fallback_path"]["steps"] == ["LightGlue/ALIKED", "SIFT/RANSAC"]
        assert assessment["fallback_path"]["fallback_triggered"] is True


def test_telemetry_prints_assessment_for_accepted_and_rejected_runs():
    from app import format_telemetry_report

    assessment = _assessment("SUCCESS_SUBPIXEL")
    for verdict, rmse in (("SUCCESS_SUBPIXEL", 0.4), ("DEGENERATE_FAILURE", None)):
        text = format_telemetry_report(
            verdict, 20, 25, 80, 1.4, [5, 5, 5, 5], rmse=rmse,
            confidence_assessment=assessment,
        )
        assert "Confidence score:  " in text or "Confidence score:" in text
        assert "Decision reason:" in text
        assert "Fallback path: LightGlue/ALIKED -> SIFT/RANSAC" in text


def test_pre_fit_failure_summary_also_has_explicit_assessment():
    from app import _failed_judge_metrics_summary

    summary = _failed_judge_metrics_summary()
    assert summary["confidence_score"] == 0.0
    assert summary["decision_reason"]
    assert summary["fallback_path"]["primary_engine"] == "Unavailable"


def test_registration_dossier_schema_accepts_confidence_and_fallback_fields():
    from src.schemas.models import RegistrationDossier

    fields = RegistrationDossier.model_fields
    assert {"confidence_score", "confidence_basis", "decision_reason", "fallback_path"} <= set(fields)
