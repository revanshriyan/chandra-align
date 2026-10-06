"""Phase 9 — LoFTR candidate matcher arm (detector-free dense matching).

Peer-reviewed motivation: Hou et al., ISPRS Annals XI-2-2026 (Tianwen-1 HiRIC
vs CTX, no-GCP regime): LoFTR gave 3-4x the correspondences of LightGlue
hybrids with roughly halved outlier rates. Our case (small rotation,
cross-modal) favors learned dense methods — but LoFTR is terrestrially
trained, so transfer to lunar imagery is NOT assumed. It runs behind the
same gates and is scored under held-out discipline like every other arm.

Requires torch + kornia (kornia.feature.LoFTR). If unavailable, the arm
reports 'unavailable' instead of failing — the retry harness treats that
as a skipped arm, not an error.
"""

import numpy as np


def loftr_available():
    try:
        import torch  # noqa: F401
        from kornia.feature import LoFTR  # noqa: F401
        return True
    except Exception:
        return False


class LoFTRMatcher:
    """Detector-free dense matcher arm. match(a, b) -> (pts_a, pts_b)."""

    def __init__(self, pretrained="outdoor"):
        self.pretrained = pretrained
        self._matcher = None

    def _load(self):
        import torch
        from kornia.feature import LoFTR
        self._matcher = LoFTR(pretrained=self.pretrained)
        self._matcher.eval()

    @staticmethod
    def _to_tensor(img):
        import torch
        a = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(a, [1, 99])
        a = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
        t = torch.from_numpy(a).unsqueeze(0).unsqueeze(0)
        return t

    def match(self, img_a, img_b):
        """Return (pts_a, pts_b) as float64 (N,2) arrays, possibly empty."""
        if not loftr_available():
            raise RuntimeError(
                "LoFTR arm unavailable: torch + kornia required "
                "(pip install torch kornia)")
        import torch
        if self._matcher is None:
            self._load()
        ta = self._to_tensor(img_a)
        tb = self._to_tensor(img_b)
        with torch.no_grad():
            out = self._matcher({"image0": ta, "image1": tb})
        k0 = out["keypoints0"].cpu().numpy().reshape(-1, 2)
        k1 = out["keypoints1"].cpu().numpy().reshape(-1, 2)
        conf = out.get("confidence", None)
        if conf is not None:
            conf = conf.cpu().numpy().ravel()
            keep = conf > 0.2
            k0, k1 = k0[keep], k1[keep]
        return np.asarray(k0, np.float64), np.asarray(k1, np.float64)
