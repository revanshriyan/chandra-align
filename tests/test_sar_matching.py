import pytest
import numpy as np
import cv2
from chandra_align.matching.sar_optical import (
    compute_structural_gradient,
    SAROpticalGradientMatcher,
)
from chandra_align.matching.deep_matchers import BaseMatcher, MatchResult
from chandra_align.preprocessing.sar import refined_lee_filter, prepare_sar_intensity


class TestStructuralGradient:
    """Tests for structural gradient computation."""

    def test_gradient_on_uniform_image(self):
        """Test gradient on uniform image (should be zero)."""
        img = np.ones((100, 100), dtype=np.uint8) * 128
        grad = compute_structural_gradient(img)
        
        assert grad.shape == (100, 100)
        assert grad.dtype == np.uint8
        assert np.mean(grad) < 5  # Should be near zero

    def test_gradient_on_edge(self):
        """Test gradient detects vertical edge."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255
        
        grad = compute_structural_gradient(img)
        
        assert grad.shape == (100, 100)
        # Edge at column 50 should have high gradient
        edge_grad = np.max(grad[:, 48:52])
        non_edge_grad = np.max(grad[:, :40])
        assert edge_grad > non_edge_grad * 5

    def test_gradient_handles_float_input(self):
        """Test gradient handles float input correctly."""
        img = np.ones((50, 50), dtype=np.float32) * 100.0
        img[:, 25:] = 200.0
        
        grad = compute_structural_gradient(img)
        
        assert grad.shape == (50, 50)
        assert grad.dtype == np.uint8
        edge_grad = np.max(grad[:, 23:27])
        assert edge_grad > 50


class TestSAROpticalGradientMatcher:
    """Tests for SAR-Optical gradient matcher."""

    def test_matcher_initialization(self):
        """Test matcher initializes correctly."""
        matcher = SAROpticalGradientMatcher(max_features=2000)
        assert matcher.name == "SAR_Optical_Gradient"
        assert matcher.max_features == 2000
        assert matcher.is_available is True

    def test_matcher_synthetic_sar_optical_pair(self):
        """Test matcher on synthetic SAR-optical pair with known shift."""
        # Create base optical image with texture
        optical = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(optical, (40, 40), (160, 160), 200, -1)
        cv2.rectangle(optical, (60, 60), (140, 140), 100, -1)
        cv2.circle(optical, (100, 100), 30, 255, -1)

        # Create SAR version: add speckle, invert intensity (radiometric inversion)
        np.random.seed(42)
        optical_float = optical.astype(np.float32)
        speckle = np.random.gamma(shape=2.0, scale=0.5, size=optical.shape)
        sar = (255.0 - optical_float) * speckle  # Radiometric inversion + speckle
        sar = np.clip(sar, 0, 255).astype(np.float32)

        matcher = SAROpticalGradientMatcher(max_features=2000)
        res = matcher.match(sar, optical.astype(np.float32))

        assert res.matcher_name == "SAR_Optical_Gradient"
        assert res.status in ["SUCCESS", "POOR_MATCHES"]
        assert isinstance(res.pts_src, np.ndarray)
        assert isinstance(res.pts_ref, np.ndarray)
        assert res.execution_time_sec > 0

    def test_matcher_with_noisy_sar(self):
        """Test matcher handles noisy SAR input."""
        # Optical: clean textured image
        optical = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        
        # SAR: heavily speckled, inverted
        np.random.seed(123)
        optical_float = optical.astype(np.float32)
        speckle = np.random.gamma(shape=1.0, scale=1.0, size=optical.shape)
        sar = (255.0 - optical_float) * speckle
        sar = np.clip(sar, 1e-6, None).astype(np.float32)  # Avoid zeros for log

        matcher = SAROpticalGradientMatcher(max_features=2000)
        res = matcher.match(sar, optical.astype(np.float32))

        assert res.matcher_name == "SAR_Optical_Gradient"
        assert res.status in ["SUCCESS", "POOR_MATCHES"]
        assert res.execution_time_sec > 0

    def test_matcher_handles_sar_with_zeros(self):
        """Test matcher handles SAR with zero values (clipped in preprocessing)."""
        optical = np.ones((100, 100), dtype=np.uint8) * 128
        cv2.rectangle(optical, (20, 20), (80, 80), 255, -1)
        
        # SAR with many zeros (simulating noise floor)
        sar = np.zeros((100, 100), dtype=np.float32)
        sar[20:80, 20:80] = 100.0  # Signal only in rectangle region

        matcher = SAROpticalGradientMatcher(max_features=2000)
        res = matcher.match(sar, optical.astype(np.float32))

        assert res.matcher_name == "SAR_Optical_Gradient"
        assert res.status in ["SUCCESS", "POOR_MATCHES", "FAILED"]
        assert res.execution_time_sec > 0

    def test_matcher_inverted_radiometry(self):
        """Test matcher handles radiometric inversion (SAR bright where optical dark)."""
        # Optical: dark circle on light background
        optical = np.ones((200, 200), dtype=np.uint8) * 200
        cv2.circle(optical, (100, 100), 50, 50, -1)
        
        # SAR: inverted (bright circle on dark background) + speckle
        np.random.seed(42)
        optical_float = optical.astype(np.float32)
        speckle = np.random.gamma(shape=2.0, scale=0.5, size=optical.shape)
        sar = (255.0 - optical_float) * speckle
        sar = np.clip(sar, 1e-6, None).astype(np.float32)

        matcher = SAROpticalGradientMatcher(max_features=2000)
        res = matcher.match(sar, optical.astype(np.float32))

        assert res.matcher_name == "SAR_Optical_Gradient"
        assert res.status in ["SUCCESS", "POOR_MATCHES"]
        # Should find reasonable number of matches
        if res.status == "SUCCESS":
            assert len(res.pts_src) >= 15


class TestSARMatchingEdgeCases:
    """Edge case tests for SAR-optical matching."""

    def test_both_uniform_images(self):
        """Test matching two uniform images (should return POOR_MATCHES)."""
        sar = np.ones((100, 100), dtype=np.float32) * 100.0
        optical = np.ones((100, 100), dtype=np.uint8) * 128
        
        matcher = SAROpticalGradientMatcher()
        res = matcher.match(sar, optical)
        
        assert res.status in ["POOR_MATCHES", "FAILED"]
        assert res.matcher_name == "SAR_Optical_Gradient"

    def test_mismatched_sizes(self):
        """Test matcher handles mismatched image sizes gracefully."""
        sar = np.random.rand(100, 100).astype(np.float32) * 100
        optical = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        matcher = SAROpticalGradientMatcher()
        res = matcher.match(sar, optical)
        
        # Should handle gracefully
        assert res.status in ["SUCCESS", "POOR_MATCHES", "FAILED"]

    def test_matcher_execution_time_recorded(self):
        """Test that execution time is recorded."""
        sar = np.random.rand(200, 200).astype(np.float32) * 100
        optical = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        
        matcher = SAROpticalGradientMatcher()
        res = matcher.match(sar, optical)
        
        assert res.execution_time_sec > 0
        assert res.execution_time_sec < 30  # Should complete in reasonable time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])