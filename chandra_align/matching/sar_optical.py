import numpy as np
import cv2
from typing import Tuple
from chandra_align.matching.deep_matchers import BaseMatcher, MatchResult, ClassicalSIFTMatcher
from chandra_align.preprocessing.sar import refined_lee_filter, prepare_sar_intensity

def compute_structural_gradient(img: np.ndarray) -> np.ndarray:
    """Computes normalized gradient magnitude map invariant to contrast inversions."""
    if img.dtype != np.uint8:
        img_8u = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_8u = img

    grad_x = cv2.Sobel(img_8u, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_8u, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    
    norm_grad = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return norm_grad.astype(np.uint8)

class SAROpticalGradientMatcher(BaseMatcher):
    """
    Cross-modal matcher converting SAR and Optical images into structural gradient maps
    prior to keypoint detection and descriptor matching.
    """
    def __init__(self, max_features: int = 5000):
        super().__init__("SAR_Optical_Gradient")
        self.max_features = max_features
        self.is_available = True

    def match(self, img_sar: np.ndarray, img_opt: np.ndarray) -> MatchResult:
        import time
        start_time = time.perf_counter()

        try:
            # 1. Speckle filter SAR image
            sar_denoised = refined_lee_filter(img_sar, win_size=5)
            sar_prep = prepare_sar_intensity(sar_denoised)

            # 2. Extract structural gradient maps
            sar_grad = compute_structural_gradient(sar_prep)
            opt_grad = compute_structural_gradient(img_opt)

            # 3. Perform descriptor matching on gradient structures
            sift_matcher = ClassicalSIFTMatcher(max_features=self.max_features)
            res = sift_matcher.match(sar_grad, opt_grad)
            res.matcher_name = self.name
            res.execution_time_sec = time.perf_counter() - start_time
            return res

        except Exception as e:
            return MatchResult(
                pts_src=np.empty((0, 2)),
                pts_ref=np.empty((0, 2)),
                confidence=np.empty((0,)),
                matcher_name=self.name,
                execution_time_sec=time.perf_counter() - start_time,
                status="FAILED",
                error_msg=f"SAR-Optical matching failed: {str(e)}"
            )