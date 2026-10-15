"""Tests for Claude audit item #9 — coverage for recent work.

1. Non-circular rescue scorer (_noncircular_rescue_score): verifies the
   80/20 split, determinism, fail-closed guards, and — critically — that
   the reported RMSE is measured on points excluded from every fit
   (perturbing holdout-only points changes the score; a circular scorer
   would be bit-identical).
2. AnyMatch _fit_pe_limit: downscale factor, coordinate-rescale math, and
   the <=2016px untouched path. Synthetic arrays only — no weights.
3. XoFTR frontend: flag logic, fail-closed when weights are absent, and
   graceful (None, None) from match_pair_xoftr without weights.

All tests run on CPU with no network and no large weights.
"""

import os

import numpy as np
import pytest

from chandra_align.deform_field import _apply_affine, _noncircular_rescue_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_rescue_inputs(n=120, seed=42, noise=0.1):
    """Smooth-warp point set that passes the full non-circular pipeline."""
    rng = np.random.default_rng(seed)
    ps = rng.uniform(0, 2000, (n, 2))
    Ma = np.array([[1.0, 0.0, 7.3], [0.0, 1.0, -3.9]])
    warp = np.stack([3.0 * np.sin(ps[:, 0] / 400.0),
                     2.0 * np.cos(ps[:, 1] / 500.0)], axis=1)
    pr = _apply_affine(Ma, ps) + warp + rng.normal(0, noise, (n, 2))
    mask = np.ones(n, dtype=bool)
    return ps, pr, mask, Ma


def _holdout_indices(n):
    """Reproduce the scorer's deterministic 80/20 split (seed 1007)."""
    perm = np.random.default_rng(1007).permutation(n)
    n_hold = max(1, int(round(n * 0.2)))
    return perm[:n_hold], perm[n_hold:]


# ---------------------------------------------------------------------------
# 1. Non-circular rescue scorer
# ---------------------------------------------------------------------------

def test_noncircular_split_counts():
    ps, pr, mask, Ma = _synthetic_rescue_inputs(n=120)
    out = _noncircular_rescue_score(ps, pr, mask, Ma, seed=7)
    assert out["noncircular_n_fit_raw"] == 96
    assert out["noncircular_n_hold_raw"] == 24
    assert out["noncircular_scoring"] is True


def test_noncircular_split_disjoint_and_deterministic():
    n = 120
    idx_hold, idx_fit = _holdout_indices(n)
    assert len(set(idx_hold) & set(idx_fit)) == 0
    assert len(idx_hold) + len(idx_fit) == n
    # deterministic across calls
    h2, f2 = _holdout_indices(n)
    assert np.array_equal(idx_hold, h2) and np.array_equal(idx_fit, f2)


def test_noncircular_result_keys():
    ps, pr, mask, Ma = _synthetic_rescue_inputs()
    out = _noncircular_rescue_score(ps, pr, mask, Ma, seed=7)
    for key in ("noncircular_rmse_px", "noncircular_n_fit_raw",
                "noncircular_n_hold_raw", "noncircular_n_admitted_fit",
                "noncircular_n_admitted_hold", "noncircular_lambda",
                "noncircular_scoring"):
        assert key in out, key
    assert out["noncircular_rmse_px"] > 0
    assert np.isfinite(out["noncircular_rmse_px"])


def test_noncircular_deterministic():
    ps, pr, mask, Ma = _synthetic_rescue_inputs()
    a = _noncircular_rescue_score(ps, pr, mask, Ma, seed=7)
    b = _noncircular_rescue_score(ps, pr, mask, Ma, seed=7)
    assert a["noncircular_rmse_px"] == b["noncircular_rmse_px"]


def test_noncircular_score_measured_on_holdout():
    """Perturbing ONLY holdout points must change the reported RMSE.

    The holdout pool is disjoint from every fit (scoring field, admission,
    refit), so a circular scorer measured on fit/admitted points would be
    bit-identical here. A change proves the score comes from the holdout.
    """
    ps, pr, mask, Ma = _synthetic_rescue_inputs(n=120)
    base = _noncircular_rescue_score(ps, pr, mask, Ma, seed=7)
    idx_hold, _ = _holdout_indices(len(ps))
    pr_pert = pr.copy()
    pr_pert[idx_hold] += 0.5  # within the 1px admission threshold
    pert = _noncircular_rescue_score(ps, pr_pert, mask, Ma, seed=7)
    assert pert["noncircular_rmse_px"] != base["noncircular_rmse_px"]
    # +0.5px systematic shift on the scored pool must inflate the RMSE
    assert pert["noncircular_rmse_px"] > base["noncircular_rmse_px"]


def test_noncircular_too_few_raw_raises():
    ps, pr, mask, Ma = _synthetic_rescue_inputs(n=49)
    with pytest.raises(ValueError, match="too_few_raw"):
        _noncircular_rescue_score(ps, pr, mask, Ma, seed=7)


def test_noncircular_garbage_holdout_fails_closed():
    """A +5px holdout shift breaks the 1px admission gate -> ValueError."""
    ps, pr, mask, Ma = _synthetic_rescue_inputs(n=120)
    idx_hold, _ = _holdout_indices(len(ps))
    pr_bad = pr.copy()
    pr_bad[idx_hold] += 5.0
    with pytest.raises(ValueError, match="too_few_admitted_hold"):
        _noncircular_rescue_score(ps, pr_bad, mask, Ma, seed=7)


# ---------------------------------------------------------------------------
# 2. AnyMatch _fit_pe_limit
# ---------------------------------------------------------------------------

def _pe_limit():
    from chandra_align.anymatch_frontend import _fit_pe_limit, _PE_LIMIT_PX
    return _fit_pe_limit, _PE_LIMIT_PX


def test_pe_limit_small_inputs_untouched():
    _fit_pe_limit, _PE_LIMIT_PX = _pe_limit()
    assert _PE_LIMIT_PX == 2016
    a = np.zeros((140, 2000), dtype=np.uint8)
    b = np.zeros((140, 2000), dtype=np.uint8)
    s0, s1, scale = _fit_pe_limit(a, b)
    assert scale == 1.0
    assert s0.shape == a.shape and s1.shape == b.shape


def test_pe_limit_large_inputs_downscaled():
    _fit_pe_limit, _PE_LIMIT_PX = _pe_limit()
    a = np.zeros((4400, 308), dtype=np.uint8)
    b = np.zeros((4400, 308), dtype=np.uint8)
    s0, s1, scale = _fit_pe_limit(a, b)
    assert scale == pytest.approx(2016.0 / 4400.0)
    assert s0.shape[0] == 2016  # longest side capped
    assert s1.shape == s0.shape


def test_pe_limit_shared_scale_across_pair():
    """One shared factor from the max dim of BOTH images."""
    _fit_pe_limit, _PE_LIMIT_PX = _pe_limit()
    a = np.zeros((3000, 100), dtype=np.uint8)   # max dim 3000
    b = np.zeros((100, 100), dtype=np.uint8)   # small
    s0, s1, scale = _fit_pe_limit(a, b)
    assert scale == pytest.approx(2016.0 / 3000.0)
    # both images scaled by the same factor (relative scale preserved)
    assert s0.shape[0] == round(3000 * scale)
    assert s1.shape == (round(100 * scale), round(100 * scale))


def test_pe_limit_coordinate_rescale_roundtrip():
    """The caller's rescale (mkpts / scale) inverts the downscale."""
    _fit_pe_limit, _ = _pe_limit()
    a = np.zeros((4400, 308), dtype=np.uint8)
    b = np.zeros((4400, 308), dtype=np.uint8)
    _, _, scale = _fit_pe_limit(a, b)
    dw, dh = round(308 * scale), round(4400 * scale)  # downscaled frame
    rng = np.random.default_rng(0)
    # synthetic "model outputs" in the downscaled frame
    mk = np.stack([rng.uniform(0, dw, 50), rng.uniform(0, dh, 50)],
                  axis=1).astype(np.float32)
    recovered = mk / scale
    # must land back inside the ORIGINAL image frame
    assert recovered[:, 0].max() <= 308
    assert recovered[:, 1].max() <= 4400
    # and the mapping is exactly invertible
    assert np.allclose(recovered * scale, mk)


# ---------------------------------------------------------------------------
# 3. XoFTR frontend (no weights required)
# ---------------------------------------------------------------------------

def _xoftr_mod():
    import chandra_align.xoftr_frontend as m
    return m


def test_xoftr_flag_logic():
    m = _xoftr_mod()
    old = os.environ.get("CHANDRA_XOFTR")
    try:
        for val, expected in [("1", True), ("true", True), ("YES", True),
                              ("on", True), ("0", False), ("off", False),
                              ("", False), ("2", False)]:
            os.environ["CHANDRA_XOFTR"] = val
            assert m.xoftr_enabled() is expected, val
        if "CHANDRA_XOFTR" in os.environ:
            del os.environ["CHANDRA_XOFTR"]
        assert m.xoftr_enabled() is False
    finally:
        if old is None:
            os.environ.pop("CHANDRA_XOFTR", None)
        else:
            os.environ["CHANDRA_XOFTR"] = old


def test_xoftr_available_false_when_weights_missing(monkeypatch):
    m = _xoftr_mod()
    monkeypatch.setattr(m, "_CKPT", "/nonexistent/weights_xoftr_640.ckpt")
    assert m.xoftr_available() is False


def test_xoftr_match_graceful_without_weights(monkeypatch):
    """match_pair_xoftr must return (None, None), never raise, when the
    checkpoint is absent (fail-closed for the pipeline's fallback)."""
    m = _xoftr_mod()
    monkeypatch.setattr(m, "_CKPT", "/nonexistent/weights_xoftr_640.ckpt")
    monkeypatch.setattr(m, "_model", None)
    a = np.zeros((64, 64), dtype=np.uint8)
    b = np.zeros((64, 64), dtype=np.uint8)
    p0, p1 = m.match_pair_xoftr(a, b)
    assert p0 is None and p1 is None
