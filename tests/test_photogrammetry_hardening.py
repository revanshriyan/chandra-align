import csv

import cv2
import numpy as np

from chandra_align.features.distribution import (
    select_distributed_matches,
    spatial_distribution_metrics,
)
from chandra_align.preprocessing.multimodal import (
    gaussian_scale_pyramid,
    preprocess_multimodal_pair,
    resize_to_common_ground_sample,
)


def test_match_bucketing_enforces_per_bucket_ceiling_and_preserves_pairs():
    points = np.array([[20 + i % 80, 20 + (i * 7) % 80] for i in range(40)], dtype=float)
    paired = points + np.array([3.0, -2.0])
    ref, sec, indices = select_distributed_matches(
        points, paired, (100, 100), grid_shape=(1, 1), max_per_bucket=6
    )
    assert len(indices) == 6
    assert np.allclose(sec - ref, [3.0, -2.0])


def test_entropy_uniformity_is_one_for_uniform_grid():
    points = np.array([[x, y] for y in (32, 96, 160, 224) for x in (32, 96, 160, 224)])
    metrics = spatial_distribution_metrics(points, (256, 256), (4, 4))
    assert metrics["spatial_entropy"] == np.log(16)
    assert metrics["uniformity"] == 1.0
    assert metrics["occupied_buckets"] == 16


def test_gradient_multimodal_features_are_inversion_tolerant():
    rng = np.random.default_rng(7)
    optical = rng.integers(10, 245, (128, 128), dtype=np.uint8)
    infrared = 255 - optical
    ref_features, sec_features = preprocess_multimodal_pair(optical, infrared)
    assert ref_features.dtype == np.uint8
    assert np.array_equal(ref_features, sec_features)


def test_resolution_matching_uses_gaussian_pyramid_and_returns_scale():
    image = np.random.default_rng(8).integers(0, 255, (1024, 1024), dtype=np.uint8)
    levels = gaussian_scale_pyramid(image, max_levels=6)
    assert len(levels) >= 5
    matched, scale = resize_to_common_ground_sample(image, 0.25, 5.0)
    assert matched.shape[0] < image.shape[0] / 4
    assert 0 < scale < 0.2
    assert np.isfinite(matched).all()


def test_subpixel_parabolic_offset_has_correct_sign():
    from chandra_align.refine import refine_subpixel_ncc

    rng = np.random.default_rng(22)
    reference = cv2.GaussianBlur(rng.integers(0, 256, (128, 128), dtype=np.uint8), (3, 3), 0.7)
    expected = np.array([0.35, -0.25])
    secondary = cv2.warpAffine(
        reference, np.float32([[1, 0, expected[0]], [0, 1, expected[1]]]), (128, 128),
        flags=cv2.INTER_CUBIC,
    )
    ref_points, sec_points, stats = refine_subpixel_ncc(
        reference, secondary, np.array([[64.0, 64.0]]), np.array([[64.0, 64.0]]),
        ncc_window=11, search_range_px=1,
    )
    assert len(ref_points) == 1 and stats["refined_pairs"] == 1
    assert np.allclose(sec_points[0] - ref_points[0], expected, atol=0.2)


def test_gcp_export_includes_bucket_id(tmp_path):
    from chandra_align.export import export_gcp_csv
    from chandra_align.metrics import ResidualVector

    vectors = [ResidualVector(10, 10, 9, 10, 1, 0, 1, 0.25)]
    path = tmp_path / "gcp.csv"
    export_gcp_csv(vectors, str(path), image_shape=(100, 100), grid_shape=(8, 8))
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == ["point_id", "ref_x", "ref_y", "sec_x", "sec_y",
                             "residual_px", "residual_m", "bucket_id"]
    assert rows[0]["bucket_id"] == "0"


def test_geotiff_exports_pixel_scale_affine_and_bounds(tmp_path):
    import rasterio
    from chandra_align.export import export_alignment_geotiff

    path = tmp_path / "registered.tif"
    export_alignment_geotiff(np.zeros((20, 30), dtype=np.uint16), str(path), pixel_scale_m=0.25)
    with rasterio.open(path) as dataset:
        assert dataset.width == 30 and dataset.height == 20
        assert dataset.transform.a == 0.25
        assert dataset.transform.e == -0.25
        assert dataset.bounds.right == 7.5
        assert dataset.bounds.bottom == -5.0


def test_alignment_core_optical_to_iirs_uses_structural_matching():
    from app import _align_core

    rng = np.random.default_rng(17)
    optical = np.zeros((400, 400), dtype=np.uint8)
    for _ in range(80):
        x, y = rng.integers(20, 380, size=2)
        radius = int(rng.integers(5, 22))
        intensity = int(rng.integers(40, 230))
        cv2.circle(optical, (int(x), int(y)), radius, intensity, -1)
        cv2.circle(optical, (int(x), int(y)), radius + 2, 255 - intensity, 1)
    optical = cv2.GaussianBlur(optical, (3, 3), 0.6)
    infrared = 255 - cv2.warpAffine(optical, np.float32([[1, 0, 5], [0, 1, -4]]), (400, 400))

    result = _align_core(
        optical, infrared, enable_clahe=False, enable_shadow_suppression=False,
        sensor_pair_mode="Optical <-> Infrared", secondary_sensor_name="IIRS",
        reference_sensor_name="OHRC",
    )
    assert result["sensor_pair_mode"] == "Optical <-> Infrared"
    assert result["inlier_cnt"] >= 4
    assert result["refinement_stats"]["status"] == "subpixel_model_reestimated"
