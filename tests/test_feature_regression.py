import numpy as np
import shutil
from pathlib import Path
import pytest

pytest.importorskip("gradio", reason="application tests require the pinned Gradio runtime")
pytest.importorskip("multipart", reason="application tests require python-multipart")

from app import load_lunar_raster, process_alignment
from chandra_align.metrics.quadrant import validate_registration_gate
from chandra_align.testing import make_pair_shift


def test_uint8_raster_arrays_keep_their_original_intensity_values():
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)

    loaded = load_lunar_raster(image)

    assert loaded.dtype == np.uint8
    assert loaded.flags.c_contiguous
    np.testing.assert_array_equal(loaded, image)


def test_ui_adapter_does_not_assume_tmc2_for_unlabelled_secondary(monkeypatch):
    import app

    captured = {}

    def capture(*args):
        captured["args"] = args
        return "captured"

    monkeypatch.setattr(app, "process_wrapper", capture)
    result = app._process_alignment_from_ui(
        "synthetic_reference.png", "synthetic_secondary.png", "OHRC",
        True, True, False, False,
    )

    assert result == "captured"
    assert captured["args"][2:5] == (
        "OHRC", "OHRC", "Optical <-> Optical",
    )


def test_ui_adapter_infers_sensors_for_native_cross_sensor_examples(monkeypatch):
    import app

    captured = {}
    monkeypatch.setattr(app, "process_wrapper", lambda *args: captured.setdefault("args", args))
    examples = Path(__file__).resolve().parents[1] / "docs" / "assets" / "examples"
    app._process_alignment_from_ui(
        str(examples / "nac_reference_iirs.png"),
        str(examples / "iirs_band125_secondary.png"), "OHRC",
        True, True, False, False,
    )

    assert captured["args"][2:5] == (
        "LROC_NAC", "IIRS", "Optical <-> Infrared",
    )


def test_input_sensor_inference_reads_pds4_sidecar_label(tmp_path):
    from app import _infer_input_sensor

    raster = tmp_path / "unlabeled.img"
    raster.touch()
    raster.with_suffix(".xml").write_text(
        "<Product_Observational><instrument_id>CH2_OHRC</instrument_id></Product_Observational>",
        encoding="utf-8",
    )

    assert _infer_input_sensor(raster, "TMC-2") == "OHRC"


def test_calibrated_pair_keeps_subpixel_registration_after_raster_loading():
    reference, secondary, _ = make_pair_shift(
        dx=7.3, dy=-3.9, angle_deg=0.4, seed=7
    )

    outputs = process_alignment(
        reference, secondary,
        pixel_scale_m=0.25,
        enable_clahe=True,
        enable_wallis=False,
        sensor_name="OHRC",
        secondary_sensor_name="OHRC",
        enforce_uniform_distribution=True,
    )
    try:
        metrics = outputs[5]
        global_metrics = metrics["global_metrics"]

        assert len(outputs) == 14
        assert "ACCEPTED (Sub-Pixel Precision)" in outputs[4]
        assert global_metrics["inlier_count"] > 30
        assert global_metrics["rmse_pixels"] <= 0.50
        assert global_metrics["spatial_entropy_score"] >= 0.75
        assert metrics["active_quadrants_count"] >= 3
    finally:
        if outputs[8]:
            shutil.rmtree(Path(outputs[8]).parent, ignore_errors=True)


def test_gate_reports_failed_fit_criteria_without_false_quadrant_failure():
    quad_counts = {"Q1": 3, "Q2": 4, "Q3": 2, "Q4": 5}

    status, code = validate_registration_gate(
        rmse=1.0465,
        inliers=5,
        min_inliers=15,
        spatial_entropy=1.9219,
        quad_counts=quad_counts,
    )

    assert code == "DEGENERATE_FAILURE"
    assert "High Residual RMSE" in status
    assert "Insufficient Inlier Yield (5 < 15)" in status
    assert "Degenerate Spatial Cluster" not in status
    assert "Degenerate Single-Quadrant Cluster" not in status
