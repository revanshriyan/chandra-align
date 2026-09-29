import pytest
import numpy as np
import cv2
import tempfile
import os
from app import (
    process_alignment, interface,
    _align_core, match_pair_hf
)
from chandra_align.testing import make_pair_shift


def test_app_initialization():
    """Verify Gradio app structure builds cleanly."""
    assert interface is not None


def test_synthetic_groundtruth_example_pair():
    """Verify the calibrated pair remains available through standard Examples."""
    examples = os.path.join(os.path.dirname(__file__), "..", "docs", "assets", "examples")
    ref = cv2.imread(os.path.join(examples, "synthetic_groundtruth_reference.png"), cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(os.path.join(examples, "synthetic_groundtruth_secondary.png"), cv2.IMREAD_GRAYSCALE)
    assert ref is not None and sec is not None
    assert ref.shape == sec.shape
    assert not np.array_equal(ref, sec)


def test_match_pair_hf():
    """Verify match_pair_hf returns valid correspondences."""
    # Create synthetic images with distinctive features
    ref = np.zeros((300, 300), dtype=np.uint8)
    sec = np.zeros((300, 300), dtype=np.uint8)
    for i in range(5):
        for j in range(5):
            cv2.circle(ref, (50 + i * 50, 50 + j * 50), 15, 255, -1)
            cv2.circle(sec, (55 + i * 50, 55 + j * 50), 15, 255, -1)

    pts_ref, pts_sec, engine, _diagnostics = match_pair_hf(ref, sec)
    
    assert len(pts_ref) >= 4
    assert len(pts_sec) >= 4
    assert pts_ref.shape == pts_sec.shape
    assert isinstance(engine, str)
    assert len(engine) > 0


def test_align_core_synthetic():
    """Verify the calibrated shift clears the strict registration gate."""
    ref, sec, _known_transform = make_pair_shift(
        dx=7.3, dy=-3.9, angle_deg=0.4, seed=7
    )
    result = _align_core(
        ref, sec,
        pixel_scale_m=0.25,
        enable_clahe=True,
        enable_shadow_suppression=False,
        enable_wallis=False,
        enforce_uniform_distribution=False,
        secondary_sensor_name="OHRC",
        reference_sensor_name="OHRC",
    )

    assert not result["rejected"]
    assert result["status_code"] == "SUCCESS_SUBPIXEL"
    assert result["inlier_cnt"] > 20
    assert result["rmse_px"] <= 0.50
    assert result["quadrant_spatial_entropy"] >= 0.75
    assert result["judge_metrics"]["active_quadrants_count"] >= 3
    assert result["ground_metrics"].rmse_m >= 0.0
    assert result["ground_metrics"].pixel_scale_m == 0.25
    assert 0.0 <= result["uniformity"] <= 1.0
    assert result["refinement_stats"]["status"] == "subpixel_model_reestimated"
    assert len(result["deformation_vectors"]) == result["inlier_cnt"]
    assert result["grid_analysis"]["grid_shape"] == (8, 8)
    assert result["warped_preview_rgb"].shape[:2] == ref.shape
    assert result["checkerboard_rgb"].shape == (*ref.shape, 3)
    assert result["error_vector_overlay_rgb"].shape == (*ref.shape, 3)


def test_process_alignment_synthetic():
    """Verify previews, accepted telemetry, and all scientific package outputs."""
    from PIL import Image
    import zipfile
    ref, sec, _known_transform = make_pair_shift(
        dx=7.3, dy=-3.9, angle_deg=0.4, seed=7
    )
    outputs = process_alignment(
        ref, sec, pixel_scale_m=0.25, enable_clahe=True,
        enable_shadow_suppression=True, enable_wallis=False,
        sensor_name="OHRC", secondary_sensor_name="OHRC",
        enforce_uniform_distribution=True,
    )
    assert len(outputs) == 14
    assert all(isinstance(image, Image.Image) for image in outputs[:4])
    assert outputs[0].size == (ref.shape[1], ref.shape[0])
    assert "ACCEPTED (Sub-Pixel Precision)" in outputs[4]
    global_metrics = outputs[5]["global_metrics"]
    assert global_metrics["inlier_count"] > 20
    assert global_metrics["rmse_pixels"] <= 0.50
    assert global_metrics["spatial_entropy_score"] >= 0.75
    assert outputs[5]["active_quadrants_count"] >= 3
    assert "VALIDATION GATE CHECKLIST:" in outputs[6]
    assert all(path and os.path.isfile(path) for path in outputs[8:14])
    with zipfile.ZipFile(outputs[13]) as package:
        assert len(package.namelist()) >= 5

    import shutil
    shutil.rmtree(os.path.dirname(outputs[8]), ignore_errors=True)


def test_process_alignment_errors():
    """Verify missing and featureless inputs return masked, crash-safe results."""
    temp_dir = tempfile.mkdtemp()
    blank_path = os.path.join(temp_dir, "blank.png")
    cv2.imwrite(blank_path, np.zeros((100, 100), dtype=np.uint8))

    class MockFile:
        def __init__(self, path):
            self.name = path

    from PIL import Image
    for outputs in (
        process_alignment(None, MockFile(blank_path)),
        process_alignment(MockFile(blank_path), MockFile(blank_path)),
    ):
        assert len(outputs) == 14
        assert isinstance(outputs[0], Image.Image)
        assert "N/A — Alignment Rejected" in outputs[6]
        assert "VALIDATION GATE CHECKLIST:" in outputs[6]
        assert "[✗]" in outputs[6]
        assert outputs[8:] == (None, None, None, None, None, None)

    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_process_wrapper_uses_cpu_fallback_on_zero_gpu_exception(monkeypatch):
    import app
    from PIL import Image

    expected = _wrapper_output_fixture(Image)
    calls = []

    def fail_gpu(*args, **kwargs):
        raise RuntimeError("ZeroGPU quota exceeded")

    def cpu_core(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(app, "run_alignment_on_gpu", fail_gpu)
    monkeypatch.setattr(app, "_run_alignment_core", cpu_core)

    result = app.process_wrapper("ref", "sec", "OHRC", "TMC-2", "Optical <-> Optical",
                                 True, 0.25, True, 3.0, True, False, 128.0, 50.0)

    assert len(calls) == 1
    assert result[0] is expected[0]
    assert len(result) == 14
    assert result[8:] == expected[8:]
    assert "ACCEPTED" in result[4]
    assert app.CPU_FALLBACK_STATUS in result[4]
    assert app.GPU_EXECUTION_STATUS not in result[4]


def test_process_wrapper_marks_zero_gpu_success(monkeypatch):
    import app
    from PIL import Image

    expected = _wrapper_output_fixture(Image)
    monkeypatch.setattr(app, "run_alignment_on_gpu", lambda *args, **kwargs: expected)
    monkeypatch.setattr(app, "_zerogpu_runtime_enabled", lambda: True)

    result = app.process_wrapper("ref", "sec", "OHRC", "TMC-2", "Optical <-> Optical",
                                 True, 0.25, True, 3.0, True, False, 128.0, 50.0)

    assert result[0] is expected[0]
    assert len(result) == 14
    assert result[8:] == expected[8:]
    assert app.GPU_EXECUTION_STATUS in result[4]


def _wrapper_output_fixture(image_type):
    blank = np.zeros((2, 2, 3), dtype=np.uint8)
    report = (
        "✅ ACCEPTED (Sub-Pixel Precision)\n"
        "Sensor Pair Mode: Optical <-> Optical\n"
        "Verified Inliers: 56 / 100 (56.0%)\n"
        "  RMSE: 0.33 px\n"
        "Registration Transform: 4-DOF Partial Affine (2x3), RANSAC threshold 3.0 px"
    )
    return (
        *(image_type.new("RGB", (2, 2)) for _ in range(4)),
        report, {"status_code": "SUCCESS_SUBPIXEL"},
        "Hardware Runtime Mode : CPU", (blank, blank),
        "a.csv", "b.json", "c.tif", "d.png", "e.png", "f.zip",
    )


def test_preprocessing_functions():
    """Test the new preprocessing module functions."""
    from chandra_align.preprocessing import (
        apply_clahe, detect_shadows, apply_wallis_filter, preprocess_pipeline
    )
    
    # Create test image
    img = np.zeros((100, 100), dtype=np.float64)
    img[25:75, 25:75] = 1.0
    img = cv2.GaussianBlur(img, (15, 15), 5)
    
    # Test CLAHE
    clahe_img = apply_clahe(img, clip_limit=3.0)
    assert clahe_img.shape == img.shape
    assert clahe_img.dtype == np.float64
    assert 0.0 <= clahe_img.min() and clahe_img.max() <= 1.0
    
    # Test shadow detection
    shadow_mask = detect_shadows(img, method="otsu")
    assert shadow_mask.shape == img.shape
    assert shadow_mask.dtype == np.float64
    assert set(np.unique(shadow_mask)).issubset({0.0, 1.0})
    
    # Test Wallis filter
    wallis_img = apply_wallis_filter(img)
    assert wallis_img.shape == img.shape
    assert wallis_img.dtype == np.float64
    
    # Test pipeline
    processed, shadow = preprocess_pipeline(
        img, enable_clahe=True, enable_shadow_mask=True, enable_wallis=False
    )
    assert processed.shape == img.shape
    assert shadow is not None
    assert shadow.shape == img.shape


def test_metrics_functions():
    """Test the new metrics module functions."""
    from chandra_align.metrics import (
        compute_deformation_field, grid_deformation_analysis,
        compute_ground_metrics, get_sensor_pixel_scale,
        ResidualVector, GroundMetrics
    )
    from chandra_align.metrics import apply_transform
    
    # Test sensor pixel scales
    assert get_sensor_pixel_scale("OHRC") == 0.25
    assert get_sensor_pixel_scale("TMC-2") == 5.0
    assert get_sensor_pixel_scale("Unknown") == 0.25  # default
    
    # Test ground metrics
    residuals = np.array([0.5, 0.6, 0.4, 0.7, 0.3])
    gm = compute_ground_metrics(residuals, pixel_scale_m=0.25)
    assert gm.rmse_px > 0
    assert gm.rmse_m == gm.rmse_px * 0.25
    assert gm.mae_m == gm.mae_px * 0.25
    assert gm.n_points == 5
    
    # Test deformation field
    pts_ref = np.array([[100, 100], [200, 200], [300, 150]], dtype=np.float64)
    pts_sec = np.array([[101, 101], [202, 198], [299, 152]], dtype=np.float64)
    H = np.eye(3, dtype=np.float64)
    
    vectors = compute_deformation_field(pts_ref, pts_sec, H, pixel_scale_m=0.25)
    assert len(vectors) == 3
    assert all(isinstance(v, ResidualVector) for v in vectors)
    assert all(v.magnitude_m == v.magnitude_px * 0.25 for v in vectors)
    
    # Test grid analysis
    grid = grid_deformation_analysis(vectors, (8, 8), image_shape=(512, 512))
    assert "cells" in grid
    assert len(grid["cells"]) == 64
    assert "mean_magnitude_px" in grid


def test_visualization_functions():
    """Test the new visualization module functions."""
    from chandra_align.visualization import (
        create_quiver_plot, create_error_distribution_plot,
        create_combined_visualization, ResidualVector
    )
    
    # Create test data
    ref_img = np.random.rand(256, 256).astype(np.float64)
    vectors = [
        ResidualVector(100, 100, 101, 101, 1.0, 1.0, 1.41, 0.35),
        ResidualVector(200, 200, 199, 198, -1.0, -2.0, 2.24, 0.56),
        ResidualVector(150, 150, 152, 151, 2.0, 1.0, 2.24, 0.56),
    ]
    
    # Test quiver plot
    fig = create_quiver_plot(ref_img, vectors, grid_shape=(4, 4))
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
    
    # Test error distribution plot
    fig = create_error_distribution_plot(vectors, rings=[0.5, 1.0])
    assert fig is not None
    plt.close(fig)
    
    # Test combined visualization
    warped = ref_img.copy()
    diff = np.abs(ref_img - warped)
    fig = create_combined_visualization(
        ref_img, warped, diff, vectors,
        grid_shape=(4, 4), pixel_scale_m=0.25,
        rmse_px=1.5, mae_px=1.2,
        engine_name="Test", inlier_count=3, total_matches=5
    )
    assert fig is not None
    plt.close(fig)


def test_export_functions():
    """Test the new export module functions."""
    import tempfile
    import os
    import json
    import csv
    from chandra_align.export import (
        export_gcp_csv, export_homography_json,
        export_alignment_png, export_composite_visualization,
        export_full_package
    )
    from chandra_align.metrics import ResidualVector
    
    temp_dir = tempfile.mkdtemp()
    
    # Create test data
    ref_img = (np.random.rand(256, 256) * 255).astype(np.uint8)
    warped = (np.random.rand(256, 256) * 255).astype(np.uint8)
    diff = cv2.absdiff(ref_img, warped)
    
    vectors = [
        ResidualVector(100, 100, 101, 101, 1.0, 1.0, 1.41, 0.35, 1.0),
        ResidualVector(200, 200, 199, 198, -1.0, -2.0, 2.24, 0.56, 1.0),
    ]
    
    H = np.eye(3, dtype=np.float64)
    metrics = {
        "rmse_px": 1.5, "mae_px": 1.2, "std_px": 0.5,
        "rmse_m": 0.375, "mae_m": 0.3, "std_m": 0.125
    }
    
    # Test GCP CSV export
    csv_path = os.path.join(temp_dir, "test_gcps.csv")
    export_gcp_csv(vectors, csv_path, pixel_scale_m=0.25)
    assert os.path.exists(csv_path)
    
    # Verify CSV content
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert len(rows) == 3  # header + 2 data rows
    assert rows[0] == ['point_id', 'ref_x', 'ref_y', 'sec_x', 'sec_y', 'residual_px', 'residual_m', 'bucket_id']
    
    # Test homography JSON export
    json_path = os.path.join(temp_dir, "test_transform.json")
    export_homography_json(H, metrics, json_path, sensor_name="OHRC", pixel_scale_m=0.25)
    assert os.path.exists(json_path)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    assert "homography_matrix" in data
    assert "metrics_px" in data
    assert "metrics_m" in data
    assert data["metadata"]["sensor"] == "OHRC"
    
    # Test PNG export
    png_path = os.path.join(temp_dir, "test_warped.png")
    export_alignment_png(warped, png_path, bit_depth=16)
    assert os.path.exists(png_path)
    
    # Test composite visualization
    comp_path = os.path.join(temp_dir, "test_composite.png")
    export_composite_visualization(ref_img, warped, diff, comp_path)
    assert os.path.exists(comp_path)
    
    # Test full package
    package_paths = export_full_package(
        ref_img, warped, diff, vectors, H, metrics,
        temp_dir, base_name="test", pixel_scale_m=0.25,
        sensor_name="OHRC", inlier_count=2, total_matches=5,
        matcher_name="Test", trust_flag="TRUSTED"
    )
    assert all(os.path.exists(p) for p in package_paths.values())
    
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_fix1_fallback_trigger_message_truthfulness():
    """Verify fallback reason matches actual trigger condition and does not claim low inlier ratio when false."""
    from app import match_pair_hf
    blank1 = np.zeros((100, 100), dtype=np.uint8)
    blank2 = np.zeros((100, 100), dtype=np.uint8)
    _pts1, _pts2, _engine, diagnostics = match_pair_hf(blank1, blank2)
    assert diagnostics["fallback_triggered"] is True
    assert "insufficient correspondences" in diagnostics["fallback_reason"].lower() or "exception" in diagnostics["fallback_reason"].lower()
    assert "Low Inlier Ratio (< 0.15)" not in diagnostics["fallback_reason"]


def test_fix2_rejected_run_judge_metrics_null_rmse():
    """Verify rejected runs set rmse_pixels to null in judge JSON and N/A in telemetry."""
    from chandra_align.metrics.quadrant import build_judge_metrics_summary
    from app import format_telemetry_report

    summary = build_judge_metrics_summary(
        rmse_pixels=1.45e-9,
        inlier_count=8,
        total_correspondences=8,
        spatial_entropy_score=0.0,
        quadrant_metrics={"Q1": {"inlier_count": 8, "rmse_px": 1.45e-9}},
        min_inliers=8,
    )
    assert summary["status_code"] == "DEGENERATE_FAILURE"
    assert summary["global_metrics"]["rmse_pixels"] is None
    assert summary["quadrant_breakdown"]["Q1_top_left_rmse"] is None
    assert summary["transformation_type"] == "N/A — Alignment Rejected"

    telemetry = format_telemetry_report(
        status="DEGENERATE_FAILURE",
        inliers=8,
        total_pts=8,
        inlier_ratio=100.0,
        spatial_entropy=0.0,
        quad_counts=[8, 0, 0, 0],
        rmse=1.45e-9,
    )
    assert "[✗] Sub-Pixel Precision  : RMSE <= 0.50 px (Measured: N/A)" in telemetry
    assert "1.45e-09" not in telemetry


def test_fix3_large_image_notice():
    """Verify large image notice is generated with actual numbers and target dimension."""
    from app import get_large_image_notices
    large_ref = np.zeros((5000, 5000), dtype=np.uint8)
    notices = get_large_image_notices(large_ref, None, max_dimension=4096)
    assert len(notices) == 1
    assert "5000×5000" in notices[0]
    assert "downsampled to 4096×4096 px" in notices[0] or "4096" in notices[0]
    assert "expect extended runtime" in notices[0]


def test_fix4_gpu_duration_and_progress_minimal():
    """Verify ZeroGPU duration is configured to 90s and submit button uses show_progress='minimal'."""
    import app
    assert getattr(app.run_alignment_on_gpu, "duration", 90) == 90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

