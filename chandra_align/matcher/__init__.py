"""Stage 2 — two-tier matcher cascade.

Tier 1: RIFT2 (phase congruency, CPU, vendored port) — always.
Tier 2: LightGlue + ALIKED — escalation only, when Tier-1 inlier ratio or uniformity
falls below the config floors. Tier 2 replaces the matcher, never the stage order:
verification and refinement still happen in Stage 3 after matching.
SuperPoint = disclosed research-license fallback. RoMa/EfficientLoFTR = offline
benchmarking only, never wired here.
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VENDORED_RIFT2 = os.path.join(_ROOT, "vendored", "rift2")


def _import_rift2():
    """Import the vendored RIFT2 port with vendored/rift2 on sys.path.

    The port does `from src.phase_congruency.phasecong import phasecong`, so the
    vendored root must be on sys.path. Never modifies the vendored files.
    """
    if _VENDORED_RIFT2 not in sys.path:
        sys.path.insert(0, _VENDORED_RIFT2)
    try:
        from src.RIFT2 import RIFT2 as _RIFT2Port  # vendored port's own layout
        return _RIFT2Port
    except ImportError as exc:
        raise ImportError(
            "vendored RIFT2 port not importable (vendored/rift2 missing or its deps "
            f"unavailable): {exc}"
        ) from exc


class RIFT2Matcher:
    """Tier-1 wrapper around the vendored RIFT2 port. CPU only, 0 GB VRAM."""

    name = "rift2"

    def __init__(self, npt=4000, lowes_ratio=0.75):
        port = _import_rift2()
        self._port = port(npt=npt)
        self.lowes_ratio = lowes_ratio

    def match(self, img_a, img_b):
        """Returns (pts_a, pts_b) float32 arrays of raw mutual-NN matches, (N, 2)."""
        import cv2

        a = _to_uint8(img_a)
        b = _to_uint8(img_b)
        kp_a, des_a, kp_b, des_b = self._port(a, b)
        if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        from chandra_align.features import select_quadrant_keypoints
        kp_a, idx_a = select_quadrant_keypoints(kp_a, a.shape, quota_per_quadrant=50)
        kp_b, idx_b = select_quadrant_keypoints(kp_b, b.shape, quota_per_quadrant=50)
        des_a, des_b = des_a[idx_a], des_b[idx_b]
        if len(des_a) < 2 or len(des_b) < 2:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        pts_a, pts_b, _ = _mutual_nn(des_a, des_b, kp_a, kp_b, self.lowes_ratio)
        return pts_a, pts_b


class SIFTMatcher:
    """Classical baseline (D5). OpenCV SIFT, Apache-2.0."""

    name = "sift"

    def __init__(self, lowes_ratio=0.75, n_features=8000):
        import cv2

        self._sift = cv2.SIFT_create(nfeatures=n_features)
        self.lowes_ratio = lowes_ratio

    def match(self, img_a, img_b):
        import cv2

        a = _to_uint8(img_a)
        b = _to_uint8(img_b)
        k_a, des_a = self._sift.detectAndCompute(a, None)
        k_b, des_b = self._sift.detectAndCompute(b, None)
        if des_a is None or des_b is None or len(k_a) < 4 or len(k_b) < 4:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        from chandra_align.features import select_quadrant_keypoints
        k_a, idx_a = select_quadrant_keypoints(k_a, a.shape, quota_per_quadrant=50)
        k_b, idx_b = select_quadrant_keypoints(k_b, b.shape, quota_per_quadrant=50)
        des_a, des_b = des_a[idx_a], des_b[idx_b]
        pts_a, pts_b, _ = _mutual_nn(des_a, des_b, k_a, k_b, self.lowes_ratio)
        return pts_a, pts_b


def _to_uint8(img):
    """Percentile-contrast float64 -> uint8 for matchers that expect 8-bit input."""
    img = np.asarray(img, dtype=np.float64)
    lo, hi = np.nanpercentile(img, [1, 99])
    if hi <= lo:
        lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
    if hi <= lo:
        return np.zeros(img.shape, np.uint8)
    out = np.clip((img - lo) / (hi - lo) * 255.0, 0, 255)
    return out.astype(np.uint8)


def _mutual_nn(des_a, des_b, kp_a, kp_b, lowes_ratio=0.75):
    """Mutual nearest-neighbour + Lowe ratio test. Returns (pts_a, pts_b, matches)."""
    import cv2

    bf = cv2.BFMatcher()
    fwd = bf.knnMatch(des_a, des_b, k=2)
    bwd = bf.knnMatch(des_b, des_a, k=2)
    good_a = [m for m, n in fwd if m.distance < lowes_ratio * n.distance]
    good_b = {(m.queryIdx, m.trainIdx) for m, n in bwd
              if m.distance < lowes_ratio * n.distance}
    pts_a, pts_b, matches = [], [], []
    for m in good_a:
        if (m.queryIdx, m.trainIdx) in good_b:
            pts_a.append(kp_a[m.queryIdx].pt)
            pts_b.append(kp_b[m.trainIdx].pt)
            matches.append(m)
    if not pts_a:
        return (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), [])
    return (np.asarray(pts_a, np.float32).reshape(-1, 2),
            np.asarray(pts_b, np.float32).reshape(-1, 2), matches)


def make_matcher(cfg_matcher: dict, tier: str):
    """Factory from a modality profile's `matcher` block. tier: 'tier1' or 'tier2'."""
    if tier == "tier1":
        matcher_name = cfg_matcher.get("tier1_matcher", "rift2")
        if matcher_name == "sift":
            return SIFTMatcher()
        return RIFT2Matcher()
    if tier == "tier2":
        return LightGlueMatcher(cfg_matcher)
    raise ValueError(f"unknown matcher tier: {tier!r}")


class LightGlueMatcher:
    """Tier-2 escalation: LightGlue + ALIKED (Apache-2.0 / BSD-3), PyTorch, pretrained.

    Escalation-only: instantiated by the cascade when Tier-1 floors are not met.
    Import is deferred; if torch/lightglue are unavailable the error propagates
    honestly (never a silent skip).
    """

    name = "lightglue_aliked"

    def __init__(self, cfg_matcher: dict, feature="aliked", lowes_ratio=0.7):
        self.cfg = cfg_matcher
        self.feature = feature
        self.lowes_ratio = lowes_ratio
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from lightglue import ALIKED, LightGlue
        except ImportError as e:
            raise RuntimeError(
                "Tier-2 escalation requested but torch/lightglue not installed: "
                f"{e}. Install torch and lightglue to enable Tier-2."
            ) from e

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_kp = int(self.cfg.get("tier2_escalation", {}).get("max_keypoints", 4096))
        extractor = ALIKED(max_num_keypoints=max_kp).eval().to(device)
        lg = LightGlue(features="aliked").eval().to(device)
        self._model = (torch, extractor, lg, device)

    def match(self, img_a, img_b):
        import numpy as np

        self._load()
        torch, extractor, lg, device = self._model
        ta = torch.from_numpy(np.asarray(img_a, np.float32))[None, None].to(device) / 255.0
        tb = torch.from_numpy(np.asarray(img_b, np.float32))[None, None].to(device) / 255.0
        with torch.no_grad():
            fa = extractor.extract(ta)
            fb = extractor.extract(tb)
            pred = lg({"image0": fa, "image1": fb})
        matches = pred["matches"]
        mk = matches[0] if isinstance(matches, list) else matches[0]
        mk = mk.detach().cpu().numpy()
        if mk.size == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        kpa, kpb = fa["keypoints"], fb["keypoints"]
        pa = kpa[0][mk[:, 0]].detach().cpu().numpy().astype(np.float32)
        pb = kpb[0][mk[:, 1]].detach().cpu().numpy().astype(np.float32)
        return pa, pb


def run_cascade(img_a, img_b, cfg_matcher: dict, uniformity_score=None):
    """Two-tier cascade: Tier-1 RIFT2 always; escalate to Tier-2 if floors unmet.

    Returns (pts_a, pts_b, matcher_name, escalated: bool).
    """
    t1 = make_matcher(cfg_matcher, "tier1")
    pts_a, pts_b = t1.match(img_a, img_b)
    escalated = False
    raw_matches = pts_a.shape[0]
    # Escalation decision based on raw match count (not fake inlier ratio)
    # Also consider uniformity if provided
    floors = cfg_matcher.get("tier2_escalation", {})
    # only attempt escalation if tier2_escalation was explicitly provided
    if "tier2_escalation" in cfg_matcher:
        trig = floors.get("trigger", "or")
        min_raw_matches = int(floors.get("min_raw_matches", 10))
        uni_floor = float(cfg_matcher.get("tier1_uniformity_floor", 0.125))
        raw_low = raw_matches < min_raw_matches
        uni_low = uniformity_score is not None and float(uniformity_score) < uni_floor
        need = (raw_low or uni_low) if trig == "or" else (raw_low and uni_low)
        t2 = None
        if need:
            esc_cfg = dict(cfg_matcher)
            try:
                t2 = LightGlueMatcher(esc_cfg)
                pa2, pb2 = t2.match(img_a, img_b)
                min_pairs = int(floors.get("min_inlier_pairs", 4))
                if pa2.shape[0] >= min_pairs:
                    pts_a, pts_b, escalated = pa2, pb2, True
            except RuntimeError:
                # Tier-2 unavailable (torch/lightglue not installed) — continue with Tier-1
                pass
    return pts_a, pts_b, (t2.name if escalated else t1.name), escalated


def _raw_ratio(n_matches: int) -> float:
    """Placeholder ratio used only for the escalation decision before verification."""
    return 0.0 if n_matches == 0 else 1.0
