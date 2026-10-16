"""Opt-in WildMatch matcher front-end (LoFTR fine-tuned on lunar IIRS<->TMC-2).

Colab training 2026-10-15/16 on the 19-pair CrossFeat lunar set (15 train /
2 val W4,N1 / 2 test W5,N13): froze the descriptor backbone, trained only
the LoFTR coarse matching module (5.25M params). Trial 2026-10-16, same
harness as the AnyMatch/XoFTR baselines (frontend -> RANSAC 3.0/2000/0.99 ->
frozen gates):

  W4 (val):  1.1496 px (3411 inliers) vs base AnyMatch 1.2371 (1728)
  N1 (val):  1.3098 px (2985 inliers) vs base 1.3981 (850)
  W5 (test): 1.2568 px (3064 inliers) vs base 1.3895 (1097) -- beats the
             pre-registered 1.31 px IIRS bar and AnyMatch's 1.4531 px
  N13(test): 0.7556 px (2206 inliers) vs base 1.2888 (20)

All COARSE_ADVISORY, 4/4 quadrants, entropy >= 1.83. Wins are fit RMSE on the
trial harness (same basis as the published baselines), not held-out RMSE
and not sub-pixel; frozen gates untouched.

Additive and opt-in via CHANDRA_WILDMATCH=1. With the flag unset, this
module is never imported and the pipeline is bit-identical. Same
architecture and input handling as the AnyMatch arm (LoFTR positional
encoding cap 2016 px, shared downscale, coordinates restored to the
caller's frame); own model cache so it never cross-contaminates the
AnyMatch arm's weights.

Weights (45 MB) live outside the repo at _CKPT, like the XoFTR weights.

Fail-closed: any error returns (None, None) and the caller falls back
to the next arm, then the default matcher.
"""

import os

import numpy as np

_FLAG = "CHANDRA_WILDMATCH"
_TRUE = frozenset({"1", "true", "yes", "on"})

_AM_ROOT = "/home/hatch/workspace/anymatch/third_party/LoFTR_AnyMatch"
# WildMatch-trained LoFTR state dict (Colab 2026-10-16, sha256 recorded in
# the trial notes; reconstructed from the delivered zip).
_CKPT = "/home/hatch/workspace/wildmatch_trained/best_reconstructed.pt"

_PE_LIMIT_PX = 2016

_model = None


def wildmatch_enabled():
    """True iff the operator opted in via CHANDRA_WILDMATCH=1."""
    return os.environ.get(_FLAG, "").strip().lower() in _TRUE


def wildmatch_available():
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
    config_path = f"{_AM_ROOT}/configs/loftr/outdoor/loftr_ds_dense.py"
    spec = importlib.util.spec_from_file_location("wm_cfg", config_path)
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
    """Pad bottom/right with replicate border to a multiple of 32."""
    import cv2
    h, w = img.shape[:2]
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    if ph == 0 and pw == 0:
        return img
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REPLICATE)


def _fit_pe_limit(img0, img1, limit=_PE_LIMIT_PX):
    """Downscale both images by a shared factor so each fits `limit`."""
    import cv2
    m = max(img0.shape[0], img0.shape[1], img1.shape[0], img1.shape[1])
    if m <= limit:
        return img0, img1, 1.0
    scale = limit / m
    out = []
    for img in (img0, img1):
        h, w = img.shape[:2]
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        out.append(cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA))
    return out[0], out[1], scale


def match_pair_wildmatch(img0, img1):
    """Match two grayscale images with WildMatch-trained LoFTR.

    Returns (pts0, pts1) as float32 (N,2) arrays, or (None, None) on failure.
    Points are in the input image coordinate frames.
    """
    try:
        import torch
        model = _load_model()
        s0, s1, scale = _fit_pe_limit(img0, img1)
        p0 = _pad_to_mult(s0)
        p1 = _pad_to_mult(s1)

        t0 = torch.from_numpy(p0).float()[None, None] / 255.0
        t1 = torch.from_numpy(p1).float()[None, None] / 255.0

        with torch.no_grad():
            batch = {"image0": t0, "image1": t1}
            model(batch)
            mkpts0 = batch["mkpts0_f"].cpu().numpy()
            mkpts1 = batch["mkpts1_f"].cpu().numpy()

        if len(mkpts0) == 0:
            return None, None
        if scale != 1.0:
            mkpts0 = mkpts0 / scale
            mkpts1 = mkpts1 / scale
        return mkpts0.astype(np.float32), mkpts1.astype(np.float32)
    except Exception:
        return None, None
