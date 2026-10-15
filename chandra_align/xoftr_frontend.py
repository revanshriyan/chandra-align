"""Opt-in XoFTR matcher front-end (no training, pretrained weights).

Uses the XoFTR cross-modal LoFTR checkpoint (CVPRW 2024, pretrained for
visible<->thermal matching) for IIRS<->TMC-2 cross-modal matching.
Additive and opt-in via CHANDRA_XOFTR=1. With the flag unset, this module
is never imported and the pipeline is bit-identical.

Trial result (2026-10-15): XoFTR-640 scored 0.9791 px on refined W5
(23 inliers, COARSE, 4/4 quadrants), beating AnyMatch's 1.3324 px on the
same window in a fair apples-to-apples re-run. 640 weights beat 840 on
both bar windows (W5: 0.9791 vs 1.2385; W4: 1.0686 vs 1.1561).

Weights are NOT vendored (43 MB); referenced by absolute path, same as
the AnyMatch front-end. CPU-compatible: the checkpoint was saved on CUDA
so torch.load is shimmed with map_location='cpu', and the DataIOWrapper's
hardcoded .cuda() calls are no-ops on CPU-only torch (validated in trial).

Fail-closed: any error returns (None, None) and the caller falls back
to the default matcher.
"""

import os

import numpy as np

_FLAG = "CHANDRA_XOFTR"
_TRUE = frozenset({"1", "true", "yes", "on"})

_XO_ROOT = "/home/hatch/workspace/xoftr"
# XoFTR-640 checkpoint (proven better than 840 for IIRS<->TMC-2).
_CKPT = "/home/hatch/workspace/xoftr/weights/XoFTR_weights/weights_xoftr_640.ckpt"

# Trial resize: longest side -> 640 px (same as validated trial).
_RESIZE = 640

_model = None


def xoftr_enabled():
    """True iff the operator opted in via CHANDRA_XOFTR=1."""
    return os.environ.get(_FLAG, "").strip().lower() in _TRUE


def xoftr_available():
    """True iff weights exist and deps import."""
    if not os.path.exists(_CKPT):
        return False
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _apply_cpu_shims():
    """CPU-compatibility shims (trial-validated).

    The checkpoint was saved on CUDA; DataIOWrapper also hardcodes
    .cuda() on grayscale tensors. On CPU-only torch these are no-ops.
    """
    import torch
    if torch.cuda.is_available():
        return
    # Patch torch.load once to default to CPU mapping.
    if not getattr(torch.load, "_xoftr_cpu_patched", False):
        _orig_load = torch.load

        def _cpu_load(*args, **kwargs):
            kwargs.setdefault("map_location", torch.device("cpu"))
            return _orig_load(*args, **kwargs)

        _cpu_load._xoftr_cpu_patched = True
        torch.load = _cpu_load
    # No-op .cuda() on tensors (DataIOWrapper hardcodes it).
    if not getattr(torch.Tensor.cuda, "_xoftr_noop", False):
        _noop = lambda self, *a, **k: self  # noqa: E731
        _noop._xoftr_noop = True
        torch.Tensor.cuda = _noop


def _load_model():
    global _model
    if _model is not None:
        return _model
    import sys
    sys.path.insert(0, _XO_ROOT)

    import torch  # noqa: F401
    _apply_cpu_shims()

    from src.xoftr import XoFTR
    from src.config.default import get_cfg_defaults
    from src.utils.data_io import DataIOWrapper, lower_config

    # Same config as the validated trial.
    config = get_cfg_defaults(inference=True)
    config = lower_config(config)
    config["xoftr"]["match_coarse"]["thr"] = 0.3
    config["xoftr"]["fine"]["thr"] = 0.1
    config["xoftr"]["fine"]["denser"] = False
    matcher = XoFTR(config=config["xoftr"])
    config["test"]["img0_resize"] = _RESIZE
    config["test"]["img1_resize"] = _RESIZE
    _model = DataIOWrapper(matcher, config=config["test"], ckpt=_CKPT)
    return _model


def match_pair_xoftr(img0, img1):
    """Match two grayscale images with XoFTR-640.

    Returns (pts0, pts1) as float32 (N,2) arrays, or (None, None) on failure.
    Points are in the input image coordinate frames (DataIOWrapper rescales
    model outputs back via the preprocess scale factors).
    """
    try:
        model = _load_model()
        out = model.from_cv_imgs(img0, img1)
        mkpts0 = np.asarray(out["mkpts0"], dtype=np.float32)
        mkpts1 = np.asarray(out["mkpts1"], dtype=np.float32)
        if len(mkpts0) == 0:
            return None, None
        return mkpts0, mkpts1
    except Exception:
        return None, None
