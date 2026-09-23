import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Check optional deep learning framework availability
HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None

# Check for kornia (required for LoFTR)
HAS_KORNIA = False
try:
    import kornia
    HAS_KORNIA = True
except ImportError:
    kornia = None

# Check for lightglue
HAS_LIGHTGLUE = False
try:
    import lightglue
    HAS_LIGHTGLUE = True
except ImportError:
    lightglue = None


@dataclass
class MatchResult:
    pts_src: np.ndarray  # Shape: (N, 2) [x, y]
    pts_ref: np.ndarray  # Shape: (N, 2) [x, y]
    confidence: np.ndarray  # Shape: (N,)
    matcher_name: str
    execution_time_sec: float
    status: str  # "SUCCESS", "POOR_MATCHES", "FAILED"
    error_msg: str = ""


class BaseMatcher:
    def __init__(self, name: str):
        self.name = name
        self.is_available = False

    def match(self, img_src: np.ndarray, img_ref: np.ndarray) -> MatchResult:
        raise NotImplementedError


def _pad_to_multiple(img: np.ndarray, multiple: int = 8) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Pad image to multiple of 8 (or 16) for dense matchers.
    Returns (padded_img, (pad_h, pad_w), (orig_h, orig_w))
    """
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0), (h, w)
    
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    return padded, (pad_h, pad_w), (h, w)


def _unpad_keypoints(pts: np.ndarray, pad: Tuple[int, int]) -> np.ndarray:
    """Remove padding offset from keypoint coordinates."""
    if pts.size == 0:
        return pts
    pad_h, pad_w = pad
    return pts  # No coordinate shift needed since we padded bottom/right only


class ClassicalSIFTMatcher:
    """Classical SIFT matcher fallback when deep models are unavailable or fail."""
    def __init__(self, max_features: int = 5000):
        self.name = "SIFT"
        self.max_features = max_features
        self.is_available = True

    def match(self, img_src: np.ndarray, img_ref: np.ndarray) -> MatchResult:
        import time
        start_time = time.perf_counter()
        
        try:
            # Convert to grayscale 8-bit if necessary
            if len(img_src.shape) == 3:
                img_src_8u = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)
            elif img_src.dtype != np.uint8:
                img_src_8u = cv2.normalize(img_src, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                img_src_8u = img_src
                
            if len(img_ref.shape) == 3:
                img_ref_8u = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)
            elif img_ref.dtype != np.uint8:
                img_ref_8u = cv2.normalize(img_ref, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                img_ref_8u = img_ref

            sift = cv2.SIFT_create(nfeatures=self.max_features)
            kp1, des1 = sift.detectAndCompute(img_src_8u, None)
            kp2, des2 = sift.detectAndCompute(img_ref_8u, None)

            if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                return MatchResult(
                    pts_src=np.empty((0, 2)),
                    pts_ref=np.empty((0, 2)),
                    confidence=np.empty((0,)),
                    matcher_name=self.name,
                    execution_time_sec=time.perf_counter() - start_time,
                    status="POOR_MATCHES",
                    error_msg="Insufficient keypoints detected"
                )

            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)

            # Lowe's ratio test
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            pts_src = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            pts_ref = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            confidences = np.array([1.0 - (m.distance / 500.0) for m in good_matches], dtype=np.float32)

            status = "SUCCESS" if len(good_matches) >= 15 else "POOR_MATCHES"
            return MatchResult(
                pts_src=pts_src if len(pts_src) > 0 else np.empty((0, 2)),
                pts_ref=pts_ref if len(pts_ref) > 0 else np.empty((0, 2)),
                confidence=confidences if len(confidences) > 0 else np.empty((0,)),
                matcher_name="SIFT",
                execution_time_sec=time.perf_counter() - start_time,
                status=status
            )
        except Exception as e:
            return MatchResult(
                pts_src=np.empty((0, 2)),
                pts_ref=np.empty((0, 2)),
                confidence=np.empty((0,)),
                matcher_name="SIFT",
                execution_time_sec=time.perf_counter() - start_time,
                status="FAILED",
                error_msg=str(e)
            )


class LightGlueALIKEDMatcher:
    """LightGlue + ALIKED matcher (GPU-accelerated)."""
    def __init__(self, max_keypoints: int = 4096):
        self.name = "LightGlue_ALIKED"
        self.max_keypoints = max_keypoints
        self._model = None
        self._device = None
        
        if not HAS_TORCH or not HAS_LIGHTGLUE:
            logger.warning(f"{self.name}: torch or lightglue not available")
            self.is_available = False
        else:
            self.is_available = True

    def _load_model(self):
        if self._model is not None:
            return
            
        try:
            import torch
            from lightglue import ALIKED, LightGlue
            
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"{self.name}: Using device {self._device}")
            
            extractor = ALIKED(max_num_keypoints=self.max_keypoints).eval().to(self._device)
            matcher = LightGlue(features="aliked").eval().to(self._device)
            self._model = (torch, extractor, matcher, self._device)
            self.is_available = True
        except Exception as e:
            logger.warning(f"{self.name}: Failed to load model: {e}")
            self.is_available = False

    def match(self, img_src: np.ndarray, img_ref: np.ndarray) -> MatchResult:
        import time
        start_time = time.time()
        
        if not self.is_available:
            return MatchResult(
                pts_src=np.empty((0, 2)),
                pts_ref=np.empty((0, 2)),
                confidence=np.empty((0,)),
                matcher_name=self.name,
                execution_time_sec=0.0,
                status="FAILED",
                error_msg="Model not available"
            )
        
        try:
            self._load_model()
            if not self.is_available:
                return MatchResult(
                    pts_src=np.empty((0, 2)),
                    pts_ref=np.empty((0, 2)),
                    confidence=np.empty((0,)),
                    matcher_name=self.name,
                    execution_time_sec=0.0,
                    status="FAILED",
                    error_msg="Model load failed"
                )
            
            torch, extractor, matcher, device = self._model
            
            # Convert to tensor
            def img_to_tensor(img):
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = img.astype(np.float32) / 255.0
                return torch.from_numpy(img)[None, None].to(device)
            
            src_tensor = img_to_tensor(img_src)
            ref_tensor = img_to_tensor(img_ref)
            
            # Extract features
            with torch.no_grad():
                feats_src = extractor.extract(src_tensor)
                feats_ref = extractor.extract(ref_tensor)
                
                # Match
                matches = matcher({"image0": feats_src, "image1": feats_ref})
            
            # Extract matches
            matches0 = matches["matches"][0].detach().cpu().numpy()
            kpts0 = feats_src["keypoints"][0].detach().cpu().numpy()
            kpts1 = feats_ref["keypoints"][0].detach().cpu().numpy()
            
            if len(matches0) == 0:
                return MatchResult(
                    pts_src=np.empty((0, 2)),
                    pts_ref=np.empty((0, 2)),
                    confidence=np.empty((0,)),
                    matcher_name=self.name,
                    execution_time_sec=time.time() - start_time,
                    status="POOR_MATCHES",
                    error_msg="No matches found"
                )
            
            pts_src = kpts0[matches0[:, 0]]
            pts_ref = kpts1[matches0[:, 1]]
            
            # Simple confidence based on match index (earlier = better)
            confidences = np.linspace(1.0, 0.5, len(matches0))
            
            status = "SUCCESS" if len(pts_src) >= 15 else "POOR_MATCHES"
            return MatchResult(
                pts_src=pts_src.astype(np.float32),
                pts_ref=pts_ref.astype(np.float32),
                confidence=confidences.astype(np.float32),
                matcher_name=self.name,
                execution_time_sec=time.time() - start_time,
                status=status
            )
            
        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"{self.name}: CUDA OOM, clearing cache and falling back")
            torch.cuda.empty_cache()
            return MatchResult(
                pts_src=np.empty((0, 2)),
                pts_ref=np.empty((0, 2)),
                confidence=np.empty((0,)),
                matcher_name=self.name,
                execution_time_sec=time.time() - start_time,
                status="FAILED",
                error_msg=f"CUDA OOM: {str(e)}"
            )
        except Exception as e:
            return MatchResult(
                pts_src=np.empty((0, 2)),
                pts_ref=np.empty((0, 2)),
                confidence=np.empty((0,)),
                matcher_name=self.name,
                execution_time_sec=time.time() - start_time,
                status="FAILED",
                error_msg=str(e)
            )


class LoFTRMatcher:
    """LoFTR dense matcher (GPU-accelerated)."""
    def __init__(self, config: Dict = None):
        self.name = "LoFTR"
        self.config = config or {}
        self._model = None
        
        if not HAS_TORCH or not HAS_KORNIA:
            logger.warning(f"{self.name}: torch or kornia not available")
            self.is_available = False
        else:
            self.is_available = True

    def _load_model(self):
        if self._model is not None:
            return
            
        try:
            import torch
            import kornia
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"{self.name}: Using device {device}")
            
            # Load LoFTR from kornia
            matcher = kornia.feature.LoFTR(pretrained="outdoor").eval().to(device)
            self._model = (torch, matcher, device)
            self.is_available = True
        except Exception as e:
            logger.warning(f"{self.name}: Failed to load model: {e}")
            self.is_available = False

    def _pad_to_multiple(self, img: np.ndarray, multiple: int = 8) -> Tuple[np.ndarray, Tuple[int, int]]:
        h, w = img.shape[:2]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        
        if pad_h == 0 and pad_w == 0:
            return img, (0, 0)
        
        padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        return padded, (pad_h, pad_w)

    def match(self, img_src: np.ndarray, img_ref: np.ndarray) -> MatchResult:
        import time
        start_time = time.time()
        
        if not self.is_available:
            return MatchResult(
                pts_src=np.empty((0, 2)),
                pts_ref=np.empty((0, 2)),
                confidence=np.empty((0,)),
                matcher_name=self.name,
                execution_time_sec=0.0,
                status="FAILED",
                error_msg="Model not available"
            )
        
        try:
            self._load_model()
            if not self.is_available:
                return MatchResult(
                    pts_src=np.empty((0, 2)),
                    pts_ref=np.empty((0, 2)),
                    confidence=np.empty((0,)),
                    matcher_name=self.name,
                    execution_time_sec=0.0,
                    status="FAILED",
                    error_msg="Model load failed"
                )
            
            torch, matcher, device = self._model
            
            # Convert to grayscale
            if len(img_src.shape) == 3:
                img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)
            if len(img_ref.shape) == 3:
                img_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)
            
            # Pad to multiple of 8 for LoFTR
            src_padded, src_pad = self._pad_to_multiple(img_src, 8)
            ref_padded, ref_pad = self._pad_to_multiple(img_ref, 8)
            
            # Convert to tensors
            src_tensor = torch.from_numpy(src_padded.astype(np.float32) / 255.0)[None, None].to(device)
            ref_tensor = torch.from_numpy(ref_padded.astype(np.float32) / 255.0)[None, None].to(device)
            
            with torch.no_grad():
                data = {"image0": src_tensor, "image1": ref_tensor}
                matches = matcher(data)
            
            # Extract matches
            mkpts0 = matches["keypoints0"].detach().cpu().numpy()
            mkpts1 = matches["keypoints1"].detach().cpu().numpy()
            
            if len(mkpts0) == 0:
                return MatchResult(
                    pts_src=np.empty((0, 2)),
                    pts_ref=np.empty((0, 2)),
                    confidence=np.empty((0,)),
                    matcher_name=self.name,
                    execution_time_sec=time.time() - start_time,
                    status="POOR_MATCHES",
                    error_msg="No matches found"
                )
            
            # Remove padding offset
            pad_h0, pad_w0 = src_pad
            pad_h1, pad_w1 = ref_pad
            mkpts0[:, 0] -= pad_w0
            mkpts0[:, 1] -= pad_h0
            mkpts1[:, 0] -= pad_w1
            mkpts1[:, 1] -= pad_h1
            
            # Filter by confidence
            conf = matches["confidence"].detach().cpu().numpy()
            conf_mask = conf > 0.5
            mkpts0 = mkpts0[conf_mask]
            mkpts1 = mkpts1[conf_mask]
            conf = conf[conf_mask]
            
            status = "SUCCESS" if len(mkpts0) >= 15 else "POOR_MATCHES"
            return MatchResult(
                pts_src=mkpts0.astype(np.float32),
                pts_ref=mkpts1.astype(np.float32),
                confidence=conf.astype(np.float32),
                matcher_name=self.name,
                execution_time_sec=time.time() - start_time,
                status=status
            )
            
        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"{self.name}: CUDA OOM, clearing cache and falling back")
            torch.cuda.empty_cache()
            return MatchResult(
                pts_src=np.empty((0, 2)),
                pts_ref=np.empty((0, 2)),
                confidence=np.empty((0,)),
                matcher_name=self.name,
                execution_time_sec=time.time() - start_time,
                status="FAILED",
                error_msg=f"CUDA OOM: {str(e)}"
            )
        except Exception as e:
            return MatchResult(
                pts_src=np.empty((0, 2)),
                pts_ref=np.empty((0, 2)),
                confidence=np.empty((0,)),
                matcher_name=self.name,
                execution_time_sec=time.time() - start_time,
                status="FAILED",
                error_msg=str(e)
            )


class DeepMatcherChain:
    """Executes a priority chain of feature matchers with safe fallback logic."""
    def __init__(self, matchers: Optional[List] = None):
        if matchers is None:
            # Default priority chain: try deep matchers first, then classical
            self.matchers = []
            
            # Add deep matchers first (if available)
            if HAS_TORCH and HAS_LIGHTGLUE:
                self.matchers.append(LightGlueALIKEDMatcher())
            if HAS_TORCH and HAS_KORNIA:
                self.matchers.append(LoFTRMatcher())
            
            # Always add classical fallback
            self.matchers.append(ClassicalSIFTMatcher())
        else:
            self.matchers = matchers

    def match(self, img_src: np.ndarray, img_ref: np.ndarray, min_matches: int = 15) -> MatchResult:
        last_result = None

        for matcher in self.matchers:
            if not matcher.is_available:
                logger.info(f"Skipping unavailable matcher: {matcher.name}")
                continue

            logger.info(f"Attempting feature matching with {matcher.name}...")
            result = matcher.match(img_src, img_ref)
            last_result = result

            if result.status == "SUCCESS" and len(result.pts_src) >= min_matches:
                logger.info(f"✅ Matcher {matcher.name} succeeded with {len(result.pts_src)} inlier candidates.")
                return result

            logger.warning(f"Matcher {matcher.name} returned status '{result.status}' ({len(result.pts_src)} matches). Trying fallback...")

        if last_result is not None:
            return last_result

        return MatchResult(
            pts_src=np.empty((0, 2)),
            pts_ref=np.empty((0, 2)),
            confidence=np.empty((0,)),
            matcher_name="NONE",
            execution_time_sec=0.0,
            status="FAILED",
            error_msg="All matchers in chain failed or were unavailable."
        )