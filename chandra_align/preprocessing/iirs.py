from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import cv2


@dataclass
class IIRSPreprocessingResult:
    selected_band_index: int
    selected_wavelength_um: float
    processed_2d_raster: np.ndarray  # uint8 array ready for feature extraction (H, W)
    histogram_matched: bool
    status_msg: str


def find_optimal_reflectance_band(wavelengths_um: List[float]) -> Tuple[int, float]:
    """
    Selects the optimal SWIR reflectance band index from a list of band wavelengths (µm).
    Target preference order:
    1. Closest to 1.55 µm (high solar reflectance, low atmospheric/thermal noise)
    2. Closest to 2.10 µm
    3. Any band < 2.5 µm
    Strictly avoids thermal emission wavelengths (> 3.0 µm).
    """
    if wavelengths_um is None or len(wavelengths_um) == 0:
        raise ValueError("Wavelength list cannot be empty.")

    valid_bands = [(idx, float(wl)) for idx, wl in enumerate(wavelengths_um)
                   if np.isfinite(wl) and float(wl) < 3.0]
    if not valid_bands:
        finite_bands = [(idx, float(wl)) for idx, wl in enumerate(wavelengths_um) if np.isfinite(wl)]
        if finite_bands:
            # Retain the documented fallback when metadata contains only thermal bands.
            return finite_bands[0]
        raise ValueError("No finite wavelength values were found.")

    # Check if 1.55 µm is available (within tolerance)
    target_155 = 1.55
    bands_near_155 = [(idx, wl) for idx, wl in valid_bands if abs(wl - target_155) < 0.15]
    if bands_near_155:
        best_idx, best_wl = min(bands_near_155, key=lambda item: abs(item[1] - target_155))
        return best_idx, best_wl

    # Preference 2: Closest to 2.10 µm
    target_210 = 2.10
    bands_near_210 = [(idx, wl) for idx, wl in valid_bands if abs(wl - target_210) < 0.15]
    if bands_near_210:
        best_idx, best_wl = min(bands_near_210, key=lambda item: abs(item[1] - target_210))
        return best_idx, best_wl

    # Preference 3: Any band < 2.5 µm
    bands_under_25 = [(idx, wl) for idx, wl in valid_bands if wl < 2.5]
    if bands_under_25:
        # Pick the one closest to 1.55
        best_idx, best_wl = min(bands_under_25, key=lambda item: abs(item[1] - target_155))
        return best_idx, best_wl

    # Fallback: closest to 1.55 among valid
    best_idx, best_wl = min(valid_bands, key=lambda item: abs(item[1] - target_155))
    return best_idx, best_wl


def histogram_match_to_reference(
    source_band: np.ndarray,
    ref_raster: np.ndarray
) -> np.ndarray:
    """
    Matches the cumulative histogram of the source 2D raster to the reference 2D raster.
    Both inputs should be 2D arrays normalized to uint8 (0..255).
    """
    source_band = np.asarray(source_band)
    ref_raster = np.asarray(ref_raster)
    if source_band.ndim != 2 or ref_raster.ndim != 2 or source_band.size == 0 or ref_raster.size == 0:
        raise ValueError("Histogram matching requires 2D arrays.")
    source_band = np.nan_to_num(source_band)
    ref_raster = np.nan_to_num(ref_raster)

    # Calculate CDFs
    src_hist, _ = np.histogram(source_band.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(ref_raster.flatten(), 256, [0, 256])

    src_cdf = src_hist.cumsum().astype(np.float32)
    src_cdf /= src_cdf[-1] + 1e-8

    ref_cdf = ref_hist.cumsum().astype(np.float32)
    ref_cdf /= ref_cdf[-1] + 1e-8

    # Create lookup table
    lookup_table = np.zeros(256, dtype=np.uint8)
    for src_val in range(256):
        # Find ref value with closest CDF
        diff = np.abs(ref_cdf - src_cdf[src_val])
        lookup_table[src_val] = np.argmin(diff)

    return cv2.LUT(source_band, lookup_table)


def preprocess_iirs_raster(
    multiband_data: np.ndarray,
    wavelengths_um: Optional[List[float]] = None,
    ref_raster: Optional[np.ndarray] = None
) -> IIRSPreprocessingResult:
    """
    Main entry point for IIRS hyperspectral preprocessing.
    - multiband_data shape: (C, H, W) or (H, W)
    """
    multiband_data = np.asarray(multiband_data)
    if multiband_data.size == 0:
        raise ValueError("IIRS raster must not be empty.")
    if multiband_data.ndim == 2:
        band_data = multiband_data
        selected_idx = 0
        selected_wl = 1.55
    elif multiband_data.ndim == 3:
        num_bands = multiband_data.shape[0]
        if wavelengths_um is None or len(wavelengths_um) != num_bands:
            # Generate dummy linear range 0.8 -> 5.0 µm if not provided
            wavelengths_um = list(np.linspace(0.8, 5.0, num_bands))

        selected_idx, selected_wl = find_optimal_reflectance_band(wavelengths_um)
        band_data = multiband_data[selected_idx]
    else:
        raise ValueError(f"Invalid raster shape: {multiband_data.shape}")

    # Normalize to uint8 (min-max scaling with 2% clip)
    band_data = np.nan_to_num(np.asarray(band_data, dtype=np.float32))
    p2, p98 = np.percentile(band_data, (2, 98))
    if p98 > p2:
        norm_band = np.clip((band_data - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
    else:
        norm_band = np.zeros(band_data.shape, dtype=np.uint8)

    # Apply CLAHE first to equalize local SWIR contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_band = clahe.apply(norm_band)

    # Perform histogram matching if reference raster provided
    hist_matched = False
    if ref_raster is not None:
        ref_raster = np.asarray(ref_raster)
        if ref_raster.ndim != 2 or ref_raster.size == 0:
            raise ValueError("Reference raster for IIRS histogram matching must be a non-empty 2D image.")
        ref_raster = np.nan_to_num(ref_raster)
        if ref_raster.dtype != np.uint8:
            ref_p2, ref_p98 = np.percentile(ref_raster, (2, 98))
            if ref_p98 > ref_p2:
                ref_raster = np.clip((ref_raster - ref_p2) / (ref_p98 - ref_p2) * 255.0, 0, 255).astype(np.uint8)
            else:
                ref_raster = ref_raster.astype(np.uint8)

        enhanced_band = histogram_match_to_reference(enhanced_band, ref_raster)
        hist_matched = True

    return IIRSPreprocessingResult(
        selected_band_index=selected_idx,
        selected_wavelength_um=round(selected_wl, 3),
        processed_2d_raster=enhanced_band,
        histogram_matched=hist_matched,
        status_msg=f"Selected Band {selected_idx} ({selected_wl:.2f} µm). Histogram matched: {hist_matched}."
    )
