"""Opt-in AnyMatch matcher front-end (no training, fine-tuned weights).

Uses the AnyMatch fine-tuned LoFTR checkpoint for cross-modal matching.
The checkpoint was fine-tuned on the AnyMatch synthetic dataset (Any-syn)
for cross-modal robustness. Additive and opt-in via CHANDRA_ANYMATCH=1.
With the flag unset, this module is never imported and the pipeline is
bit-identical.

Fail-closed: any error returns (None, None) and the caller falls back
to the default matcher.
"""

import os

import numpy as np

_FLAG = "CHANDRA_ANYMATCH"
_TRUE = frozenset({"1", "true", "yes", "on"})

_AM_ROOT = "/home/hatch/workspace/anymatch/third_party/LoFTR_AnyMatch"
# Pre-extracted plain state dict (from the lightning-saved LoFTR_AnyMatch.ckpt).
_CKPT = "/home/hatch/workspace/anymatch/weights/LoFTR_AnyMatch_sd.pt"

_model = None


def anymatch_enabled():
    """True iff the operator opted in via CHANDRA_ANYMATCH=1."""
    return os.environ.get(_FLAG, "").strip().lower() in _TRUE


def anymatch_available():
    """True iff weights exist and deps import."""
    if not os.path.exists(_CKPT):
        return False
    try:
        import torch  # noqa: F401
        import kornia  # noqa: F401
        return True
    except ImportError:
        return False


def _load_model():
    global _model
    if _model is not None:
        return _model
    import sys
    sys.path.insert(0, _AM_ROOT)
    # Kornia API shim (0.8.x moved create_meshgrid)
    import kornia.utils
    import types
    if "kornia.utils.grid" not in sys.modules:
        _g = types.ModuleType("kornia.utils.grid")
        _g.create_meshgrid = kornia.utils.create_meshgrid
        sys.modules["kornia.utils.grid"] = _g

    import torch
    from src.config.default import get_cfg_defaults
    from src.loftr import LoFTR
    from yacs.config import CfgNode as CN

    cfg = get_cfg_defaults()
    import importlib.util
    # Outdoor LoFTR dense config (same as the validated test).
    config_path = f"{_AM_ROOT}/configs/loftr/outdoor/loftr_ds_dense.py"
    spec = importlib.util.spec_from_file_location("am_cfg", config_path)
    mod = importlib.util.module_from_spec(spec)
    import src.config.default as defmod
    orig = defmod._CN
    defmod._CN = cfg
    try:
        spec.loader.exec_module(mod)
    finally:
        defmod._CN = orig

    def lower_config(yacs_cfg):
        if not isinstance(yacs_cfg, CN):
            return yacs_cfg
        return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}

    loftr_cfg = lower_config(cfg)["loftr"]
    model = LoFTR(loftr_cfg)
    ckpt = torch.load(_CKPT, map_location="cpu", weights_only=True)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    sd2 = {}
    for k, v in sd.items():
        nk = k.replace("matcher.", "") if k.startswith("matcher.") else k
        sd2[nk] = v
    model.load_state_dict(sd2, strict=False)
    model.eval()
    _model = model
    return _model


def _pad_to_mult(img, mult=32):
    """Pad bottom/right with replicate border to a multiple of 32.

    Padding (not resize) preserves coordinates: keypoints in the valid
    region keep their original image coordinates.
    """
    import cv2
    h, w = img.shape[:2]
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    if ph == 0 and pw == 0:
        return img
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REPLICATE)


def match_pair_anymatch(img0, img1):
    """Match two grayscale images with AnyMatch fine-tuned LoFTR.

    Returns (pts0, pts1) as float32 (N,2) arrays, or (None, None) on failure.
    Points are in the input image coordinate frames.
    """
    try:
        import torch
        model = _load_model()
        p0 = _pad_to_mult(img0)
        p1 = _pad_to_mult(img1)

        t0 = torch.from_numpy(p0).float()[None, None] / 255.0
        t1 = torch.from_numpy(p1).float()[None, None] / 255.0

        with torch.no_grad():
            batch = {"image0": t0, "image1": t1}
            model(batch)
            mkpts0 = batch["mkpts0_f"].cpu().numpy()
            mkpts1 = batch["mkpts1_f"].cpu().numpy()

        if len(mkpts0) == 0:
            return None, None
        # Padding is bottom/right only: valid-region coordinates are unchanged.
        return mkpts0.astype(np.float32), mkpts1.astype(np.float32)
    except Exception:
        return None, None
