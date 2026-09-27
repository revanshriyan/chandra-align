import pytest
import numpy as np
import cv2
import tempfile
import os
from app import (
    process_alignment, ensure_sample_files, interface,
    _align_core, match_pair_hf
)


def test_app_initialization():
    """Verify Gradio app structure builds cleanly."""
    assert interface is not None


def test_ensure_sample_files():
    """Verify sample files are generated on startup."""
    sample_ref, sample_sec = ensure_sample_files()
    assert os.path.exists(sample_ref)
    assert os.path.exists(sample_sec)
    # Should be valid images
    ref = cv2.imread(sample_ref, cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(sample_sec, cv2.IMREAD_GRAYSCALE)
    assert ref is not None and sec is not None
    assert ref.shape == (512, 512)
    assert sec.shape == (512, 512)


def test_match_pair_hf():
    """Verify match_pair_hf returns valid correspondences."""
    # Create synthetic images with distinctive features
    ref = np.zeros((300, 300), dtype=np.uint8)
    sec = np.zeros((300, 300), dtype=np.uint8)
    for i in range(5):
        for j in range(5):
            cv2.circle(ref, (50 + i * 50, 50 + j * 50), 15, 255, -1)
            cv2.circle(sec, (55 + i * 50, 55 + j * 50), 15, 255, -1)

    pts_ref, pts_sec, engine = match_pair_hf(ref, sec)
    
    assert len(pts_ref) >= 4
    assert len(pts_sec) >= 4
    assert pts_ref.shape == pts_sec.shape
    assert isinstance(engine, str)
    assert len(engine) > 0


def test_align_core_synthetic():
    """Verify _align_core executes on CPU synthetic image pair."""
    temp_dir = tempfile.mkdtemp()
    img1_path = os.path.join(temp_dir, "ref.png")
    img2_path = os.path.join(temp_dir, "sec.png")

    # Create synthetic images with sufficient distinctive features
    ref = np.zeros((300, 300), dtype=np.uint8)
    sec = np.zeros((300, 300), dtype=np.uint8)
    for i in range(5):
        for j in range(5):
            cv2.circle(ref, (50 + i * 50, 50 + j * 50), 15, 255, -1)
            cv2.circle(sec, (55 + i * 50, 55 + j * 50), 15, 255, -1)

    cv2.imwrite(img1_path, ref)
    cv2.imwrite(img2_path, sec)

    # Test core alignment logic (bypassing @spaces.GPU decorator)
    result = _align_core(
        ref, sec,
        pixel_scale_m=0.25,
        enable_clahe=True,
        enable_shadow_suppression=True,
        enable_wallis=False
    )

    # Verify result structure
    assert result is not None
    assert "side_by_side_rgb" in result
    assert "engine_name" in result
    assert "inlier_cnt" in result
    assert "total_matches" in result
    assert "rmse_px" in result
    assert "mae_px" in result
    assert "H" in result
    assert "deformation_vectors" in result
    assert "grid_analysis" in result
    assert "ground_metrics" in result
    assert "uniformity" in result

    # Verify metrics
    assert result["inlier_cnt"] > 0
    assert result["rmse_px"] >= 0.0
    assert result["mae_px"] >= 0.0
    assert result["ground_metrics"].rmse_m > 0.0
    assert result["ground_metrics"].pixel_scale_m == 0.25
    assert 0.0 <= result["uniformity"] <= 1.0
    assert result["refinement_stats"]["status"] in (
        "subpixel_model_reestimated", "subpixel_pairs_refined", "no_subpixel_pairs",
        "insufficient_verified_matches",
    )
    assert result["inlier_cnt"] <= 8 * 8 * 8
    assert len(result["deformation_vectors"]) == result["inlier_cnt"]
    assert result["grid_analysis"]["grid_shape"] == (8, 8)

    # Verify output image
    output_img = result["side_by_side_rgb"]
    assert isinstance(output_img, np.ndarray)
    assert output_img.ndim == 3
    assert output_img.shape[2] == 3  # RGB

    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_process_alignment_synthetic():
    """Verify full CPU alignment returns metrics and every downloadable artifact."""
    temp_dir = tempfile.mkdtemp()
    img1_path = os.path.join(temp_dir, "ref.png")
    img2_path = os.path.join(temp_dir, "sec.png")

    # Create synthetic images with sufficient distinctive features for SIFT
    ref = np.zeros((300, 300), dtype=np.uint8)
    sec = np.zeros((300, 300), dtype=np.uint8)
    for i in range(5):
        for j in range(5):
            cv2.circle(ref, (50 + i * 50, 50 + j * 50), 15, 255, -1)
            cv2.circle(sec, (55 + i * 50, 55 + j * 50), 15, 255, -1)

    cv2.imwrite(img1_path, ref)
    cv2.imwrite(img2_path, sec)

    class MockFile:
        def __init__(self, path):
            self.name = path

    result_img, status, csv_path, json_path, geotiff_path, png_path, viz_path = process_alignment(
        MockFile(img1_path), MockFile(img2_path)
    )

    from PIL import Image
    assert isinstance(result_img, Image.Image)
    assert result_img.size[0] > 0 and result_img.size[1] > 0
    assert "REGISTRATION COMPLETE" in status
    assert "RMSE" in status
    assert "MAE" in status
    assert "GROUND METRICS" in status
    assert "Spatial Uniformity U:" in status
    assert "DEFORMATION FIELD" in status
    assert all(path and os.path.isfile(path) for path in (
        csv_path, json_path, geotiff_path, png_path, viz_path
    ))

    # Cleanup
    import shutil
    shutil.rmtree(os.path.dirname(csv_path), ignore_errors=True)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_process_alignment_errors():
    """Verify error handling for missing files and insufficient features."""
    temp_dir = tempfile.mkdtemp()
    blank_path = os.path.join(temp_dir, "blank.png")
    cv2.imwrite(blank_path, np.zeros((100, 100), dtype=np.uint8))

    class MockFile:
        def __init__(self, path):
            self.name = path

    # Missing file
    result_img, status, *_ = process_alignment(None, MockFile(blank_path))
    assert result_img is None
    assert "Error" in status

    # Insufficient features
    result_img, status, *_ = process_alignment(MockFile(blank_path), MockFile(blank_path))
    assert result_img is None
    assert "Registration Failed" in status

    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


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
    assert get_sensor_pixel_scale("TMC-2") == 0.5
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
