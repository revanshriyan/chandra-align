import pytest
import numpy as np
import cv2
from chandra_align.preprocessing.sar import (
    refined_lee_filter,
    frost_filter,
    prepare_sar_intensity,
)

class TestSARPreprocessing:
    """Unit tests for SAR preprocessing functions."""

    def test_refined_lee_filter_basic(self):
        """Test Refined Lee filter on synthetic speckled image."""
        # Create smooth base image
        base = np.ones((100, 100), dtype=np.float32) * 100.0
        # Add multiplicative speckle (Gamma distributed with ENL=1)
        np.random.seed(42)
        speckle = np.random.gamma(shape=1.0, scale=1.0, size=(100, 100))
        speckled = base * speckle

        filtered = refined_lee_filter(speckled, win_size=5, num_looks=1.0)
        
        # Check output shape
        assert filtered.shape == (100, 100)
        # Check output is finite
        assert np.all(np.isfinite(filtered))
        # Check output is non-negative
        assert np.all(filtered >= 0)
        # Variance in flat region should be reduced
        center_region = filtered[40:60, 40:60]
        input_center = speckled[40:60, 40:60]
        assert np.var(center_region) < np.var(input_center) * 0.5

    def test_refined_lee_filter_win_size_validation(self):
        """Test win_size validation for Refined Lee filter."""
        img = np.ones((50, 50), dtype=np.float32)
        
        # Even win_size should raise
        with pytest.raises(ValueError, match="odd integer >= 3"):
            refined_lee_filter(img, win_size=4)
        
        # win_size < 3 should raise
        with pytest.raises(ValueError, match="odd integer >= 3"):
            refined_lee_filter(img, win_size=1)
        
        # Valid win_size
        result = refined_lee_filter(np.ones((10, 10)), win_size=3)
        assert result.shape == (10, 10)

    def test_frost_filter_basic(self):
        """Test Frost filter on synthetic speckled image."""
        base = np.ones((50, 50), dtype=np.float32) * 100.0
        np.random.seed(42)
        speckle = np.random.gamma(shape=1.0, scale=1.0, size=(50, 50))
        speckled = base * speckle

        filtered = frost_filter(speckled, win_size=5, damp_factor=1.0)
        
        assert filtered.shape == (50, 50)
        assert np.all(np.isfinite(filtered))
        assert np.all(filtered >= 0)

    def test_frost_filter_win_size_validation(self):
        """Test win_size validation for Frost filter."""
        img = np.ones((50, 50), dtype=np.float32)
        
        with pytest.raises(ValueError, match="odd integer >= 3"):
            frost_filter(img, win_size=4)
        
        with pytest.raises(ValueError, match="odd integer >= 3"):
            frost_filter(img, win_size=1)

    def test_prepare_sar_intensity_handles_zeros_and_negatives(self):
        """Test log-scale preparation handles zeros and negatives safely."""
        img = np.array([
            [0.0, 1.0, 10.0],
            [-5.0, 100.0, 1000.0],
            [0.0, -1.0, 0.0]
        ], dtype=np.float32)
        
        result = prepare_sar_intensity(img)
        
        assert result.dtype == np.uint8
        assert result.shape == img.shape
        # No NaN or Inf should be present
        assert np.all(np.isfinite(result.astype(np.float32)))
        # Values should be in valid uint8 range
        assert np.all(result >= 0)
        assert np.all(result <= 255)

    def test_prepare_sar_intensity_uniform_image(self):
        """Test preparation on uniform image."""
        img = np.ones((50, 50), dtype=np.float32) * 100.0
        result = prepare_sar_intensity(img)
        
        assert result.shape == (50, 50)
        assert result.dtype == np.uint8
        # Uniform image should result in uniform output
        assert np.std(result) == 0

    def test_refined_lee_preserves_edges(self):
        """Test Refined Lee preserves edges in synthetic step edge."""
        # Create image with sharp edge
        img = np.ones((100, 100), dtype=np.float32) * 10.0
        img[:, 50:] = 100.0  # Sharp step edge at column 50
        
        # Add mild speckle
        np.random.seed(42)
        speckle = np.random.gamma(shape=4.0, scale=0.25, size=(100, 100))
        speckled = img * speckle

        filtered = refined_lee_filter(speckled, win_size=7, num_looks=4.0)
        
        # Edge should still be detectable (gradient preserved)
        # Convert to uint8 for Sobel (OpenCV limitation)
        speckled_8u = cv2.normalize(speckled, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        filtered_8u = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        grad_before = cv2.Sobel(speckled_8u, cv2.CV_32F, 1, 0, ksize=3)
        grad_after = cv2.Sobel(filtered_8u, cv2.CV_32F, 1, 0, ksize=3)
        
        # Edge at column 50 should still have high gradient
        edge_grad_before = np.max(np.abs(grad_before[:, 48:52]))
        edge_grad_after = np.max(np.abs(grad_after[:, 48:52]))
        # Gradient should not be completely smoothed out
        assert edge_grad_after > edge_grad_before * 0.3

    def test_refined_lee_homogeneous_region(self):
        """Test Refined Lee on homogeneous region (low variance)."""
        img = np.ones((50, 50), dtype=np.float32) * 100.0
        np.random.seed(42)
        speckle = np.random.gamma(shape=1.0, scale=1.0, size=(50, 50))
        speckled = img * speckle

        filtered = refined_lee_filter(speckled, win_size=5, num_looks=1.0)
        
        # Should not produce NaN or Inf
        assert np.all(np.isfinite(filtered))
        # Mean should be close to original mean
        assert abs(np.mean(filtered) - 100.0) < 20.0

    def test_frost_filter_homogeneous(self):
        """Test Frost filter on homogeneous region."""
        img = np.ones((30, 30), dtype=np.float32) * 100.0
        np.random.seed(42)
        speckle = np.random.gamma(shape=1.0, scale=1.0, size=(30, 30))
        speckled = img * speckle

        filtered = frost_filter(speckled, win_size=5, damp_factor=1.0)
        
        assert np.all(np.isfinite(filtered))
        assert filtered.shape == (30, 30)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])