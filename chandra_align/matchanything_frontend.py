"""Opt-in MatchAnything matcher front-end (no training, pretrained weights).

Uses the MatchAnything ELoFTR checkpoint for cross-modal matching.
Additive and opt-in via CHANDRA_MATCHANYTHING=1. With the flag unset,
this module is never imported and the pipeline is bit-identical.

Fail-closed: any error returns (None, None) and the caller falls back
to the default matcher.
"""

import os

import numpy as np

_FLAG = "CHANDRA_MATCHANYTHING"
_TRUE = frozenset({"1", "true", "yes", "on"})

_MA_ROOT = "/home/hatch/workspace/matchanything_hf/imcui/third_party/MatchAnything"
_CKPT = "/home/hatch/workspace/ma_weights/weights/matchanything_eloftr.ckpt"

_model = None
_cfg = None


def matchanything_enabled():
    """True iff the operator opted in via CHANDRA_MATCHANYTHING=1."""
    return os.environ.get(_FLAG, "").strip().lower() in _TRUE


def matchanything_available():
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
    global _model, _cfg
    if _model is not None:
        return _model
    import sys
    sys.path.insert(0, _MA_ROOT)
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
    # Load eloftr model config
    import importlib.util
    config_path = f"{_MA_ROOT}/configs/models/eloftr_model.py"
    spec = importlib.util.spec_from_file_location("eloftr_cfg", config_path)
    mod = importlib.util.module_from_spec(spec)
    import src.config.default as defmod
    orig = defmod._CN
    defmod._CN = cfg
    try:
        spec.loader.exec_module(mod)
    finally:
        defmod._CN = orig
    cfg.LOFTR.MATCH_COARSE.THR = 0.2
    if cfg.LOFTR.COARSE.ROPE:
        cfg.LOFTR.COARSE.NPE = [832, 832, 832, 832]

    def lower_config(yacs_cfg):
        if not isinstance(yacs_cfg, CN):
            return yacs_cfg
        return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}

    loftr_cfg = lower_config(cfg)["loftr"]
    model = LoFTR(loftr_cfg)
    ckpt = torch.load(_CKPT, map_location="cpu", weights_only=False)
    sd = {k.replace("matcher.", ""): v
          for k, v in ckpt["state_dict"].items()
          if k.startswith("matcher.")}
    model.load_state_dict(sd, strict=False)
    model.eval()
    _model = model
    _cfg = cfg
    return _model


def match_pair_matchanything(img0, img1):
    """Match two grayscale images with MatchAnything ELoFTR.

    Returns (pts0, pts1) as float32 (N,2) arrays, or (None, None) on failure.
    Points are in the input image coordinate frames.
    """
    try:
        import cv2
        import torch
        model = _load_model()
        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]

        def prep(img):
            h, w = img.shape[:2]
            hn, wn = (h // 32) * 32, (w // 32) * 32
            if hn == 0 or wn == 0:
                return None, None, None
            rs = cv2.resize(img, (wn, hn))
            t = torch.from_numpy(rs).float()[None, None] / 255.0
            return t, (h, w), (hn, wn)

        t0, orig0, new0 = prep(img0)
        t1, orig1, new1 = prep(img1)
        if t0 is None or t1 is None:
            return None, None

        with torch.no_grad():
            batch = {"image0": t0, "image1": t1}
            model(batch)
            mkpts0 = batch["mkpts0_f"].cpu().numpy()
            mkpts1 = batch["mkpts1_f"].cpu().numpy()

        if len(mkpts0) == 0:
            return None, None
        # Scale back to original image coordinates
        s0x, s0y = orig0[1] / new0[1], orig0[0] / new0[0]
        s1x, s1y = orig1[1] / new1[1], orig1[0] / new1[0]
        pts0 = mkpts0 * np.array([s0x, s0y], dtype=np.float64)
        pts1 = mkpts1 * np.array([s1x, s1y], dtype=np.float64)
        return pts0.astype(np.float32), pts1.astype(np.float32)
    except Exception:
        return None, None
