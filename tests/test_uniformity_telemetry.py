"""Spatial uniformity telemetry integration (diagnostic-only, never gated)."""

import numpy as np

from app import _compute_spatial_uniformity, _failed_judge_metrics_summary
from chandra_align.metrics.quadrant import compute_spatial_uniformity_metrics

IMG_SHAPE = (512, 512)


def _spread_points(n=64, seed=7):
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(
        rng.uniform(8, 504, n // 4), rng.uniform(8, 504, n // 4)
    )
    return np.column_stack([xs.ravel(), ys.ravel()])


def test_metrics_present_and_sane_on_synthetic_set():
    pts = _spread_points()
    metrics = compute_spatial_uniformity_metrics(pts, IMG_SHAPE)
    assert metrics["grid_shape"] == [8, 8]
    assert 0.0 < metrics["grid_occupancy"] <= 1.0
    assert 0 < metrics["occupied_cells"] <= 64
    assert metrics["n_points"] == len(pts)
    assert metrics["nn_mean_px"] > 0.0
    assert metrics["nn_median_px"] > 0.0
    assert 0.0 <= metrics["nn_p5_px"] <= metrics["nn_median_px"] <= metrics["nn_mean_px"]
    assert all(np.isfinite(float(metrics[k])) for k in (
        "grid_occupancy", "nn_mean_px", "nn_median_px", "nn_std_px", "nn_p5_px"
    ))


def test_spread_beats_clustered_on_occupancy_and_p5():
    # 64 points, one near the center of each of the 64 cells vs 64 points
    # inside a single 64x64 cell region.
    xs, ys = np.meshgrid(np.linspace(32, 480, 8), np.linspace(32, 480, 8))
    spread = np.column_stack([xs.ravel(), ys.ravel()])
    rng = np.random.default_rng(11)
    clustered = np.clip(rng.normal(256, 12, (64, 2)), 8, 504)
    m_spread = compute_spatial_uniformity_metrics(spread, IMG_SHAPE)
    m_clust = compute_spatial_uniformity_metrics(clustered, IMG_SHAPE)
    assert m_spread["occupied_cells"] == 64
    assert m_spread["grid_occupancy"] == 1.0
    assert m_spread["occupied_cells"] > m_clust["occupied_cells"]
    assert m_spread["nn_p5_px"] > m_clust["nn_p5_px"]


def test_helper_marks_valid_on_good_input():
    metrics = _compute_spatial_uniformity(_spread_points(), IMG_SHAPE)
    assert metrics["valid"] is True
    assert metrics["diagnostic_only"] is True
    assert metrics["n_points"] > 0


def test_helper_empty_input_marked_invalid_not_fabricated():
    metrics = _compute_spatial_uniformity(np.empty((0, 2)), IMG_SHAPE)
    assert metrics["valid"] is False
    assert metrics["n_points"] == 0
    assert metrics["occupied_cells"] == 0
    assert metrics["grid_occupancy"] == 0.0


def test_helper_degenerate_inputs_never_crash():
    bad_inputs = [
        None,                                   # no points at all
        np.full((4, 2), np.nan),                # all NaN
        np.full((4, 2), np.inf),                # all non-finite
        np.array([[900.0, 900.0], [-5.0, -5.0]]),  # all out of frame
        np.array([["a", "b"], ["c", "d"]]),     # unparseable dtype
    ]
    for bad in bad_inputs:
        metrics = _compute_spatial_uniformity(bad, IMG_SHAPE)
        assert metrics["valid"] is False
        assert metrics["n_points"] == 0
        for key in ("nn_mean_px", "nn_median_px", "nn_std_px", "nn_p5_px",
                    "grid_occupancy"):
            assert np.isfinite(float(metrics[key]))


def test_helper_single_point_kept_but_marked_invalid():
    # One inlier is real (n_points=1) but uniformity stats are meaningless,
    # so the block is marked invalid rather than reporting fabricated NN data.
    metrics = _compute_spatial_uniformity(np.array([[256.0, 256.0]]), IMG_SHAPE)
    assert metrics["n_points"] == 1
    assert metrics["valid"] is False
    assert metrics["occupied_cells"] == 1
    assert metrics["nn_mean_px"] == 0.0


def test_helper_bad_img_shape_marked_invalid_not_fabricated():
    for shape in [(0, 0), (-4, 512), None, (), (512,)]:
        metrics = _compute_spatial_uniformity(_spread_points(), shape)
        assert metrics["valid"] is False, shape
        assert metrics["n_points"] == 0


def test_failed_summary_carries_invalid_uniformity_block():
    summary = _failed_judge_metrics_summary()
    block = summary["spatial_uniformity_metrics"]
    assert block["valid"] is False
    assert block["diagnostic_only"] is True
    assert block["n_points"] == 0
    assert summary["status_code"] == "DEGENERATE_FAILURE"


def test_failed_summary_with_inliers_still_invalid_block():
    summary = _failed_judge_metrics_summary(inlier_count=3, total_correspondences=10)
    block = summary["spatial_uniformity_metrics"]
    assert block["valid"] is False
    assert block["grid_occupancy"] == 0.0
