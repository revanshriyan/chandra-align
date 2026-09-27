import numpy as np
import pytest
from chandra_align.evaluators.quadtree import (
    QuadtreeMetrics,
    evaluate_quadtree_uniformity,
)


class TestQuadtreeUniformity:
    """Unit tests for Quadtree Spatial Uniformity Evaluator."""

    def test_empty_keypoints_returns_zero(self):
        """Test that empty or None keypoints return zero metrics."""
        result = evaluate_quadtree_uniformity(None, (512, 512))
        assert result.uniformity_score == 0.0
        assert result.occupied_leaf_cells == 0
        assert result.total_leaf_cells == 256

        result = evaluate_quadtree_uniformity(np.array([]).reshape(0, 2), (512, 512))
        assert result.uniformity_score == 0.0

    def test_single_keypoint(self):
        """Test single keypoint distribution."""
        kp = np.array([[256.0, 256.0]], dtype=np.float32)
        result = evaluate_quadtree_uniformity(kp, (512, 512), depth=4)
        assert result.occupied_leaf_cells == 1
        assert round(result.occupancy_ratio, 4) == round(1.0 / 256, 4)
        assert result.uniformity_score >= 0.0
        assert result.uniformity_score <= 1.0

    def test_uniform_grid_distribution(self):
        """Test perfectly uniform grid distribution yields appropriate uniformity."""
        # Create uniform grid of points across 8x8 grid within 512x512
        grid_size = 8
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = (i + 0.5) * 512 / grid_size
                y = (j + 0.5) * 512 / grid_size
                points.append([x, y])
        kp = np.array(points, dtype=np.float32)
        
        result = evaluate_quadtree_uniformity(kp, (512, 512), depth=4)
        # With 64 points in 256 cells at depth=4, we get good distribution
        assert result.occupied_leaf_cells >= 60
        # Uniformity = occupancy_ratio * normalized_entropy = (64/256) * 1.0 = 0.25
        assert result.uniformity_score > 0.2
        assert result.uniformity_score < 0.35

    def test_clustered_keypoints_low_uniformity(self):
        """Test clustered keypoints yield low uniformity score."""
        # All points in one small region
        points = []
        for _ in range(100):
            x = np.random.uniform(250, 270)
            y = np.random.uniform(250, 270)
            points.append([x, y])
        kp = np.array(points, dtype=np.float32)
        
        result = evaluate_quadtree_uniformity(kp, (512, 512), depth=4)
        assert result.occupied_leaf_cells < 10  # Very few cells occupied
        assert result.uniformity_score < 0.3  # Low uniformity

    def test_depth_parameter(self):
        """Test different depth values."""
        kp = np.array([[100.0, 100.0], [400.0, 400.0]], dtype=np.float32)
        
        result_d2 = evaluate_quadtree_uniformity(kp, (512, 512), depth=2)
        assert result_d2.total_leaf_cells == 16
        
        result_d3 = evaluate_quadtree_uniformity(kp, (512, 512), depth=3)
        assert result_d3.total_leaf_cells == 64
        
        result_d4 = evaluate_quadtree_uniformity(kp, (512, 512), depth=4)
        assert result_d4.total_leaf_cells == 256
        
        result_d5 = evaluate_quadtree_uniformity(kp, (512, 512), depth=5)
        assert result_d5.total_leaf_cells == 1024

    def test_edge_coordinates(self):
        """Test keypoints at image boundaries."""
        kp = np.array([
            [0.0, 0.0],
            [511.0, 511.0],
            [0.0, 511.0],
            [511.0, 0.0],
        ], dtype=np.float32)
        
        result = evaluate_quadtree_uniformity(kp, (512, 512), depth=4)
        assert result.occupied_leaf_cells == 4
        assert result.uniformity_score > 0.0

    def test_invalid_image_shape(self):
        """Test invalid image dimensions return zero metrics."""
        kp = np.array([[100.0, 100.0]], dtype=np.float32)
        result = evaluate_quadtree_uniformity(kp, (0, 512))
        assert result.uniformity_score == 0.0
        result = evaluate_quadtree_uniformity(kp, (512, 0))
        assert result.uniformity_score == 0.0
        result = evaluate_quadtree_uniformity(kp, (-1, 512))
        assert result.uniformity_score == 0.0

    def test_large_number_of_keypoints(self):
        """Test with large number of keypoints."""
        np.random.seed(42)
        kp = np.random.uniform(0, 512, (10000, 2)).astype(np.float32)
        result = evaluate_quadtree_uniformity(kp, (512, 512), depth=4)
        assert result.occupied_leaf_cells <= 256
        assert result.uniformity_score > 0.8  # Random uniform should be high

    def test_checkerboard_pattern(self):
        """Test checkerboard pattern - should have moderate uniformity."""
        # Create checkerboard: points only in alternating cells
        points = []
        for i in range(16):
            for j in range(16):
                if (i + j) % 2 == 0:
                    x = (i + 0.5) * 512 / 16
                    y = (j + 0.5) * 512 / 16
                    points.append([x, y])
        kp = np.array(points, dtype=np.float32)
        
        result = evaluate_quadtree_uniformity(kp, (512, 512), depth=4)
        # 128 points in 256 cells, but only half cells occupied
        assert result.occupied_leaf_cells == 128
        assert result.uniformity_score > 0.0
        assert result.uniformity_score < 0.8  # Not perfectly uniform due to gaps

    def test_metrics_rounding(self):
        """Test that metrics are rounded to 4 decimal places."""
        kp = np.array([[256.0, 256.0]], dtype=np.float32)
        result = evaluate_quadtree_uniformity(kp, (512, 512), depth=4)
        
        # Check rounding
        assert round(result.uniformity_score, 4) == result.uniformity_score
        assert round(result.occupancy_ratio, 4) == result.occupancy_ratio
        assert round(result.shannon_entropy, 4) == result.shannon_entropy

    def test_output_dataclass_fields(self):
        """Test all QuadtreeMetrics fields are populated."""
        kp = np.array([[100.0, 100.0], [200.0, 200.0]], dtype=np.float32)
        result = evaluate_quadtree_uniformity(kp, (512, 512), depth=4)
        
        assert isinstance(result.uniformity_score, float)
        assert isinstance(result.occupied_leaf_cells, int)
        assert isinstance(result.total_leaf_cells, int)
        assert isinstance(result.occupancy_ratio, float)
        assert isinstance(result.shannon_entropy, float)
        assert isinstance(result.depth, int)
        assert result.depth == 4

    def test_nan_handling(self):
        """Test NaN coordinates are handled gracefully."""
        kp = np.array([[np.nan, 100.0], [100.0, np.nan]], dtype=np.float32)
        # Should not crash, NaN will be clipped or produce 0
        result = evaluate_quadtree_uniformity(kp, (512, 512), depth=4)
        assert isinstance(result.uniformity_score, float)

    def test_coordinate_types(self):
        """Test various input coordinate types."""
        # int coordinates
        kp_int = np.array([[100, 100], [200, 200]], dtype=np.int32)
        result_int = evaluate_quadtree_uniformity(kp_int, (512, 512))
        
        # float coordinates
        kp_float = np.array([[100.0, 100.0], [200.0, 200.0]], dtype=np.float32)
        result_float = evaluate_quadtree_uniformity(kp_float, (512, 512))
        
        # Should produce same results
        assert result_int.uniformity_score == result_float.uniformity_score
        assert result_int.occupied_leaf_cells == result_float.occupied_leaf_cells

    def test_entropy_calculation(self):
        """Test entropy calculation correctness."""
        # Two points in same cell -> probability 1.0 -> entropy 0
        kp = np.array([[100.0, 100.0], [101.0, 101.0]], dtype=np.float32)
        result = evaluate_quadtree_uniformity(kp, (512, 512), depth=4)
        assert result.shannon_entropy == 0.0
        
        # Two points in different cells -> probability 0.5 each -> entropy 1.0
        kp2 = np.array([[100.0, 100.0], [400.0, 400.0]], dtype=np.float32)
        result2 = evaluate_quadtree_uniformity(kp2, (512, 512), depth=4)
        assert result2.shannon_entropy > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])