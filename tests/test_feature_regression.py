import numpy as np
import shutil
from pathlib import Path

from app import load_lunar_raster, process_alignment
from chandra_align.metrics.quadrant import validate_registration_gate
from chandra_align.testing import make_pair_shift


def test_uint8_raster_arrays_keep_their_original_intensity_values():
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)

    loaded = load_lunar_raster(image)

    assert loaded.dtype == np.uint8
    assert loaded.flags.c_contiguous
    np.testing.assert_array_equal(loaded, image)


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
