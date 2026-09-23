"""
Photometric Normalization Module

Advanced preprocessing for lunar imagery to improve tie-point yield under
solar angle shifts. Implements CLAHE and Lommel-Seeliger correction.
"""

import numpy as np
import cv2


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        img: Input image (float64 or uint8, single channel)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        CLAHE-enhanced image as float64 in [0, 1] range
    """
    img = np.asarray(img, dtype=np.float64)
    
    # Normalize to uint8 for CLAHE
    lo, hi = np.nanpercentile(img, [1, 99])
    if hi <= lo:
        lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
    if hi <= lo:
        return np.zeros(img.shape, np.float64)
    
    img_u8 = np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_clahe = clahe.apply(img_u8)
    
    # Return as float64 normalized to [0, 1]
    return img_clahe.astype(np.float64) / 255.0


def apply_lommel_seeliger(
    img: np.ndarray,
    incidence_angle_deg: float,
    emission_angle_deg: float,
    ref_incidence_deg: float = 60.0,
    ref_emission_deg: float = 0.0
) -> np.ndarray:
    """
    Apply Lommel-Seeliger photometric correction to equalize illumination.
    
    Model: L(i, e) = 2 * cos(i) / (cos(i) + cos(e))
    
    Normalizes image by dividing by the disc function at observed geometry
    and multiplying by reference geometry.
    
    Args:
        img: Input image (float64)
        incidence_angle_deg: Solar incidence angle in degrees
        emission_angle_deg: Emission/viewing angle in degrees
        ref_incidence_deg: Reference incidence angle for normalization
        ref_emission_deg: Reference emission angle for normalization
        
    Returns:
        Photometrically normalized image as float64
    """
    img = np.asarray(img, dtype=np.float64)
    
    i = np.radians(incidence_angle_deg)
    e = np.radians(emission_angle_deg)
    ci, ce = np.cos(i), np.cos(e)
    
    # Lommel-Seeliger at observed geometry
    denom = ci + ce
    if denom <= 1e-9:
        # Near-terminator: return original
        return img
    
    disc_obs = 2.0 * ci / denom
    
    # Lommel-Seeliger at reference geometry
    i_ref = np.radians(ref_incidence_deg)
    e_ref = np.radians(ref_emission_deg)
    ci_ref, ce_ref = np.cos(i_ref), np.cos(e_ref)
    denom_ref = ci_ref + ce_ref
    disc_ref = 2.0 * ci_ref / denom_ref if denom_ref > 1e-9 else 1.0
    
    # Normalize: multiply by ratio of reference to observed
    # This equalizes illumination as if viewed at reference geometry
    return img * (disc_ref / disc_obs)


def apply_lunar_lambert(
    img: np.ndarray,
    incidence_angle_deg: float,
    emission_angle_deg: float,
    ref_incidence_deg: float = 60.0,
    ref_emission_deg: float = 0.0
) -> np.ndarray:
    """
    Apply Lunar-Lambert photometric correction.
    
    Model: (1-L)*cos(i) + 2*L*cos(i)/(cos(i)+cos(e))
    where L = 2 / (1 + cos(e))
    
    Args:
        img: Input image (float64)
        incidence_angle_deg: Solar incidence angle in degrees
        emission_angle_deg: Emission/viewing angle in degrees
        ref_incidence_deg: Reference incidence angle
        ref_emission_deg: Reference emission angle
        
    Returns:
        Photometrically normalized image as float64
    """
    img = np.asarray(img, dtype=np.float64)
    
    i = np.radians(incidence_angle_deg)
    e = np.radians(emission_angle_deg)
    ci, ce = np.cos(i), np.cos(e)
    
    L = 2.0 / (1.0 + ce)
    denom = ci + ce
    ls = 2.0 * ci / denom if denom > 1e-9 else 0.0
    disc_obs = (1.0 - L) * ci + L * ls
    
    # Reference geometry
    i_ref = np.radians(ref_incidence_deg)
    e_ref = np.radians(ref_emission_deg)
    ci_ref, ce_ref = np.cos(i_ref), np.cos(e_ref)
    L_ref = 2.0 / (1.0 + ce_ref)
    denom_ref = ci_ref + ce_ref
    ls_ref = 2.0 * ci_ref / denom_ref if denom_ref > 1e-9 else 0.0
    disc_ref = (1.0 - L_ref) * ci_ref + L_ref * ls_ref
    
    if disc_obs <= 1e-6:
        return img
    
    return img * (disc_ref / disc_obs)


def normalize_photometry(
    img: np.ndarray,
    incidence_angle_deg: float,
    emission_angle_deg: float,
    model: str = "lommel_seeliger",
    enable_clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: tuple = (8, 8),
    ref_incidence_deg: float = 60.0,
    ref_emission_deg: float = 0.0
) -> np.ndarray:
    """
    Full photometric normalization pipeline.
    
    Args:
        img: Input image
        incidence_angle_deg: Solar incidence angle
        emission_angle_deg: Emission angle
        model: "lommel_seeliger" or "lunar_lambert"
        enable_clahe: Whether to apply CLAHE before photometric correction
        clahe_clip_limit: CLAHE clip limit
        clahe_tile_grid: CLAHE tile grid size
        ref_incidence_deg: Reference incidence angle
        ref_emission_deg: Reference emission angle
        
    Returns:
        Normalized image as float64
    """
    img = np.asarray(img, dtype=np.float64)
    
    # Apply CLAHE first if enabled
    if enable_clahe:
        img = apply_clahe(img, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid)
    
    # Apply photometric correction
    if model == "lommel_seeliger":
        return apply_lommel_seeliger(
            img, incidence_angle_deg, emission_angle_deg,
            ref_incidence_deg, ref_emission_deg
        )
    elif model == "lunar_lambert":
        return apply_lunar_lambert(
            img, incidence_angle_deg, emission_angle_deg,
            ref_incidence_deg, ref_emission_deg
        )
    else:
        raise ValueError(f"Unknown photometric model: {model}")