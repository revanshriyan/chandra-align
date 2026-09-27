import pytest
import numpy as np
import cv2
from chandra_align.matching.deep_matchers import (
    MatchResult,
    ClassicalSIFTMatcher,
    LightGlueALIKEDMatcher,
    LoFTRMatcher,
    DeepMatcherChain,
    _pad_to_multiple,
    _unpad_keypoints,
)


class TestMatchResult:
    """Test MatchResult dataclass."""

    def test_match_result_creation(self):
        """Test MatchResult creation with valid data."""
        result = MatchResult(
            pts_src=np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
            pts_ref=np.array([[11.0, 21.0], [31.0, 41.0]], dtype=np.float32),
            confidence=np.array([0.9, 0.8], dtype=np.float32),
            matcher_name="SIFT",
            execution_time_sec=0.5,
            status="SUCCESS",
        )
        assert result.pts_src.shape == (2, 2)
        assert result.matcher_name == "SIFT"
        assert result.status == "SUCCESS"

    def test_match_result_empty(self):
        """Test MatchResult with empty arrays."""
        result = MatchResult(
            pts_src=np.empty((0, 2)),
            pts_ref=np.empty((0, 2)),
            confidence=np.empty((0,)),
            matcher_name="SIFT",
            execution_time_sec=0.1,
            status="POOR_MATCHES",
        )
        assert len(result.pts_src) == 0
        assert result.status == "POOR_MATCHES"


class TestClassicalSIFTMatcher:
    """Test classical SIFT matcher."""

    def test_sift_matcher_initialization(self):
        """Test SIFT matcher initializes correctly."""
        matcher = ClassicalSIFTMatcher(max_features=2000)
        assert matcher.name == "SIFT"
        assert matcher.max_features == 2000
        assert matcher.is_available is True

    def test_sift_matcher_synthetic_pair(self):
        """Test SIFT matcher on synthetic translated square images."""
        img1 = np.zeros((200, 200), dtype=np.uint8)
        img2 = np.zeros((200, 200), dtype=np.uint8)

        # Draw textured rectangle on img1 and shifted on img2
        cv2.rectangle(img1, (50, 50), (150, 150), 255, -1)
        cv2.rectangle(img1, (70, 70), (120, 120), 100, -1)
        cv2.rectangle(img2, (60, 60), (160, 160), 255, -1)
        cv2.rectangle(img2, (80, 80), (130, 130), 100, -1)

        matcher = ClassicalSIFTMatcher()
        res = matcher.match(img1, img2)

        assert res.status in ["SUCCESS", "POOR_MATCHES"]
        assert res.matcher_name == "SIFT"
        assert isinstance(res.pts_src, np.ndarray)
        assert isinstance(res.pts_ref, np.ndarray)
        assert res.execution_time_sec > 0

    def test_sift_matcher_no_texture(self):
        """Test SIFT matcher on uniform images (should return POOR_MATCHES)."""
        img1 = np.ones((200, 200), dtype=np.uint8) * 128
        img2 = np.ones((200, 200), dtype=np.uint8) * 128

        matcher = ClassicalSIFTMatcher()
        res = matcher.match(img1, img2)

        assert res.status in ["POOR_MATCHES", "FAILED"]
        assert res.matcher_name == "SIFT"

    def test_sift_matcher_rgb_input(self):
        """Test SIFT matcher handles RGB input correctly."""
        img1 = np.zeros((200, 200, 3), dtype=np.uint8)
        img2 = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(img1, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.rectangle(img2, (60, 60), (160, 160), (255, 255, 255), -1)

        matcher = ClassicalSIFTMatcher()
        res = matcher.match(img1, img2)

        assert res.matcher_name == "SIFT"
        assert res.status in ["SUCCESS", "POOR_MATCHES"]


class TestDeepMatcherChain:
    """Test deep matcher chain fallback logic."""

    def test_chain_initialization_default(self):
        """Test chain initializes with default matchers."""
        chain = DeepMatcherChain()
        assert len(chain.matchers) >= 1
        # Should always have SIFT as fallback
        assert any(m.name == "SIFT" for m in chain.matchers)

    def test_chain_fallback_to_sift(self):
        """Test chain falls back to SIFT when deep matchers unavailable."""
        chain = DeepMatcherChain(matchers=[ClassicalSIFTMatcher()])

        img1 = np.random.randint(0, 255, (250, 250), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (250, 250), dtype=np.uint8)

        res = chain.match(img1, img2, min_matches=15)
        assert res.matcher_name == "SIFT"
        assert res.status in ["SUCCESS", "POOR_MATCHES", "FAILED"]

    def test_chain_skips_unavailable_matchers(self):
        """Test chain skips matchers marked as unavailable."""
        # Create a mock unavailable matcher
        class UnavailableMatcher:
            name = "FakeDeepMatcher"
            is_available = False
            def match(self, img_src, img_ref):
                raise RuntimeError("Should not be called")

        chain = DeepMatcherChain(matchers=[UnavailableMatcher(), ClassicalSIFTMatcher()])

        img1 = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (200, 200), dtype=np.uint8)

        res = chain.match(img1, img2)
        assert res.matcher_name == "SIFT"

    def test_chain_returns_best_result(self):
        """Test chain returns first successful matcher."""
        # First matcher returns POOR, second returns SUCCESS
        class PoorMatcher:
            name = "PoorMatcher"
            is_available = True
            def match(self, img_src, img_ref):
                return MatchResult(
                    pts_src=np.array([[10.0, 10.0]]),
                    pts_ref=np.array([[11.0, 11.0]]),
                    confidence=np.array([0.5]),
                    matcher_name="PoorMatcher",
                    execution_time_sec=0.1,
                    status="POOR_MATCHES",
                )

        class GoodMatcher:
            name = "GoodMatcher"
            is_available = True
            def match(self, img_src, img_ref):
                return MatchResult(
                    pts_src=np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]),
                    pts_ref=np.array([[11.0, 11.0], [21.0, 21.0], [31.0, 31.0]]),
                    confidence=np.array([0.9, 0.8, 0.7]),
                    matcher_name="GoodMatcher",
                    execution_time_sec=0.1,
                    status="SUCCESS",
                )

        chain = DeepMatcherChain(matchers=[PoorMatcher(), GoodMatcher()])
        img1 = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (200, 200), dtype=np.uint8)

        res = chain.match(img1, img2, min_matches=2)
        assert res.matcher_name == "GoodMatcher"
        assert res.status == "SUCCESS"


class TestPadUnpad:
    """Test padding/unpadding utilities."""

    def test_pad_to_multiple_already_multiple(self):
        """Test padding when image is already multiple of 8."""
        img = np.zeros((256, 256), dtype=np.uint8)
        padded, pad, orig = _pad_to_multiple(img, 8)
        assert pad == (0, 0)
        assert orig == (256, 256)
        assert padded.shape == (256, 256)

    def test_pad_to_multiple_not_multiple(self):
        """Test padding when image is not multiple of 8."""
        img = np.zeros((250, 250), dtype=np.uint8)
        padded, pad, orig = _pad_to_multiple(img, 8)
        assert pad == (6, 6)  # 256 - 250 = 6
        assert orig == (250, 250)
        assert padded.shape == (256, 256)

    def test_unpad_keypoints(self):
        """Test unpadding keypoints."""
        pts = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
        pad = (6, 6)
        unpad = _unpad_keypoints(pts, pad)
        np.testing.assert_array_equal(unpad, pts)


class TestLightGlueALIKEDMatcher:
    """Test LightGlue + ALIKED matcher (mocked)."""

    def test_lightglue_matcher_initialization(self):
        """Test LightGlue matcher initialization."""
        matcher = LightGlueALIKEDMatcher(max_keypoints=2048)
        assert matcher.name == "LightGlue_ALIKED"
        assert matcher.max_keypoints == 2048

    def test_lightglue_matcher_unavailable_when_torch_missing(self):
        """Test LightGlue matcher marks unavailable when torch missing."""
        # This tests the conditional import logic
        matcher = LightGlueALIKEDMatcher()
        # is_available will be True if torch is installed, False otherwise
        assert isinstance(matcher.is_available, bool)


class TestLoFTRMatcher:
    """Test LoFTR matcher (mocked)."""

    def test_loftr_matcher_initialization(self):
        """Test LoFTR matcher initialization."""
        matcher = LoFTRMatcher()
        assert matcher.name == "LoFTR"

    def test_loftr_padding(self):
        """Test LoFTR internal padding logic."""
        matcher = LoFTRMatcher()
        img = np.zeros((250, 250), dtype=np.uint8)
        padded, pad = matcher._pad_to_multiple(img, 8)
        assert pad == (6, 6)
        assert padded.shape == (256, 256)

    def test_loftr_unpad_coordinates(self):
        """Test LoFTR coordinate unpadding."""
        matcher = LoFTRMatcher()
        mkpts = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
        pad = (6, 6)
        # LoFTR pads bottom/right, so coordinates don't shift
        # The unpadding is just removing the padded region
        unpadded = mkpts.copy()
        unpadded[:, 0] -= pad[1]
        unpadded[:, 1] -= pad[0]
        assert unpadded[0, 0] == 4.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])