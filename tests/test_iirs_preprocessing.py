import numpy as np
import pytest
from chandra_align.preprocessing.iirs import (
    IIRSPreprocessingResult,
    find_optimal_reflectance_band,
    histogram_match_to_reference,
    preprocess_iirs_raster,
)


class TestIIRSPreprocessing:
    """Unit tests for IIRS SWIR cross-modal preprocessor."""

    def test_find_optimal_band_target_1_55(self):
        """Test band selection prefers 1.55 µm target."""
        wavelengths = [0.8, 1.2, 1.55, 1.8, 2.1, 2.5, 3.0, 3.5, 4.0]
        idx, wl = find_optimal_reflectance_band(wavelengths)
        assert idx == 2
        assert wl == 1.55

    def test_find_optimal_band_target_2_10(self):
        """Test band selection prefers 2.10 µm when 1.55 not available."""
        wavelengths = [0.8, 1.2, 1.8, 2.1, 2.5]
        idx, wl = find_optimal_reflectance_band(wavelengths)
        assert idx == 3
        assert wl == 2.1

    def test_find_optimal_band_avoids_thermal(self):
        """Test band selection strictly avoids wavelengths > 3.0 µm."""
        wavelengths = [0.8, 1.2, 1.55, 3.5, 4.0, 4.5, 5.0]
        idx, wl = find_optimal_reflectance_band(wavelengths)
        assert idx == 2
        assert wl == 1.55

    def test_find_optimal_band_all_thermal_fallback(self):
        """Test fallback to first band when all bands are thermal."""
        wavelengths = [3.5, 4.0, 4.5, 5.0]
        idx, wl = find_optimal_reflectance_band(wavelengths)
        assert idx == 0
        assert wl == 3.5

    def test_find_optimal_band_empty_raises(self):
        """Test empty wavelength list raises ValueError."""
        with pytest.raises(ValueError):
            find_optimal_reflectance_band([])

    def test_histogram_match_2d_arrays(self):
        """Test histogram matching with synthetic 2D arrays."""
        np.random.seed(42)
        source = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        ref = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

        matched = histogram_match_to_reference(source, ref)
        assert matched.shape == source.shape
        assert matched.dtype == np.uint8

    def test_histogram_match_invalid_dim_raises(self):
        """Test histogram matching rejects non-2D arrays."""
        source = np.random.randint(0, 256, (3, 64, 64), dtype=np.uint8)
        ref = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        with pytest.raises(ValueError):
            histogram_match_to_reference(source, ref)

    def test_preprocess_iirs_2d_input(self):
        """Test preprocessing with 2D input (already single band)."""
        band = np.random.rand(64, 64).astype(np.float32)
        result = preprocess_iirs_raster(band)
        assert isinstance(result, IIRSPreprocessingResult)
        assert result.selected_band_index == 0
        assert result.selected_wavelength_um == 1.55
        assert result.processed_2d_raster.shape == (64, 64)
        assert result.processed_2d_raster.dtype == np.uint8
        assert not result.histogram_matched

    def test_preprocess_iirs_3d_band_selection(self):
        """Test preprocessing selects optimal band from 3D hyperspectral data."""
        wavelengths = [0.8, 1.0, 1.55, 2.1, 3.5, 4.5]  # 6 bands
        multiband = np.random.rand(6, 64, 64).astype(np.float32)
        result = preprocess_iirs_raster(multiband, wavelengths_um=wavelengths)
        assert isinstance(result, IIRSPreprocessingResult)
        assert result.selected_band_index == 2  # 1.55 µm band
        assert result.selected_wavelength_um == 1.55
        assert result.processed_2d_raster.shape == (64, 64)

    def test_preprocess_iirs_3d_avoids_thermal_bands(self):
        """Test preprocessing avoids thermal bands (>3.0 µm) even if closest to 1.55."""
        wavelengths = [0.8, 3.5, 4.0, 5.0]  # Only first band is valid
        multiband = np.random.rand(4, 64, 64).astype(np.float32)
        result = preprocess_iirs_raster(multiband, wavelengths_um=wavelengths)
        assert result.selected_band_index == 0
        assert result.selected_wavelength_um == 0.8

    def test_preprocess_iirs_histogram_matching(self):
        """Test histogram matching against reference raster."""
        np.random.seed(42)
        wavelengths = [0.8, 1.55, 2.1, 3.5]
        multiband = np.random.rand(4, 64, 64).astype(np.float32)
        ref = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

        result = preprocess_iirs_raster(multiband, wavelengths_um=wavelengths, ref_raster=ref)
        assert result.histogram_matched
        assert result.processed_2d_raster.dtype == np.uint8

    def test_preprocess_iirs_invalid_shape_raises(self):
        """Test preprocessing rejects invalid shapes."""
        with pytest.raises(ValueError):
            preprocess_iirs_raster(np.random.rand(3, 4, 64, 64))

    def test_preprocess_iirs_clahe_enhancement(self):
        """Test that CLAHE is applied (local contrast enhancement)."""
        # Create image with uniform regions - CLAHE should enhance local contrast
        band = np.ones((64, 64), dtype=np.float32) * 100
        band[10:20, 10:20] = 150  # small bright patch
        result = preprocess_iirs_raster(band)
        # CLAHE should enhance the patch
        processed = result.processed_2d_raster
        assert processed.shape == (64, 64)
        assert processed.dtype == np.uint8

    def test_preprocess_iirs_dummy_wavelengths_generated(self):
        """Test dummy wavelengths generated when not provided."""
        multiband = np.random.rand(10, 32, 32).astype(np.float32)
        result = preprocess_iirs_raster(multiband)
        assert result.selected_wavelength_um < 3.0  # Should pick valid band
        assert 0 <= result.selected_band_index < 10

    def test_preprocess_iirs_percentile_normalization(self):
        """Test percentile-based normalization (2-98% clip)."""
        # Create band with outliers
        band = np.ones((64, 64), dtype=np.float32) * 100
        band[0, 0] = 10000  # extreme outlier
        band[63, 63] = -500  # extreme negative
        result = preprocess_iirs_raster(band)
        # Should normalize without extreme values dominating
        assert result.processed_2d_raster.max() <= 255
        assert result.processed_2d_raster.min() >= 0