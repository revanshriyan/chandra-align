"""
CHANDRA-ALIGN Preprocessing Package

Unified interface for all preprocessing modules:
- SAR speckle filtering (refined_lee_filter, frost_filter)
- IIRS hyperspectral preprocessing
- General lunar image enhancement (CLAHE, shadow masking, Wallis filter)
"""

from chandra_align.preprocessing.sar import (
    refined_lee_filter,
    frost_filter,
    prepare_sar_intensity,
)

from chandra_align.preprocessing.iirs import (
    find_optimal_reflectance_band,
    histogram_match_to_reference,
    preprocess_iirs_raster,
    IIRSPreprocessingResult,
)

from chandra_align.preprocessing.general import (
    apply_clahe,
    detect_shadows,
    apply_wallis_filter,
    apply_wallis_adaptive,
    preprocess_pipeline,
    suppress_keypoints_in_shadows,
)
from chandra_align.preprocessing.multimodal import (
    gradient_structure,
    gaussian_scale_pyramid,
    preprocess_multimodal_pair,
    resize_to_common_ground_sample,
)

__all__ = [
    # SAR preprocessing
    "refined_lee_filter",
    "frost_filter",
    "prepare_sar_intensity",
    # IIRS preprocessing
    "find_optimal_reflectance_band",
    "histogram_match_to_reference",
    "preprocess_iirs_raster",
    "IIRSPreprocessingResult",
    # General preprocessing
    "apply_clahe",
    "detect_shadows",
    "apply_wallis_filter",
    "apply_wallis_adaptive",
    "preprocess_pipeline",
    "suppress_keypoints_in_shadows",
    "gradient_structure",
    "gaussian_scale_pyramid",
    "preprocess_multimodal_pair",
    "resize_to_common_ground_sample",
]
