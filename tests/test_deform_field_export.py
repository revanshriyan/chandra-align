"""Tests for the opt-in dense-remap export (CHANDRA_DEFORM_FIELD=1).

When the deform-field stage applies, the exported warped_sec carries the
fitted field (M_hook + TPS residual) via cv2.remap instead of the
affine-only warp. Fail-closed: any field/map failure falls back to the
affine warp, bit-identical to the pre-change behavior.

Unit tests on build_field_remap_maps run everywhere (numpy + cv2 only).
App-level tests need the Gradio runtime and skip locally, running in CI --
the same pattern as tests/test_deform_field_stage.py.
"""

import os

import numpy as np
import pytest
import cv2

from chandra_align.deform_field import (
    _eval_residual_field,
    _fit_residual_field,
    apply_deform_field_stage,
    build_field_remap_maps,
)


# ---------------------------------------------------------------------------
# Unit tests (no app import needed).
# ---------------------------------------------------------------------------

def _synthetic_field(seed=7):
    """Smooth sinusoidal displacement field on a 256px grid; returns (M, field)."""
    theta = np.deg2rad(2.0)
    s = 1.01
    M = np.array([[s * np.cos(theta), -s * np.sin(theta), 15.0],
                  [s * np.sin(theta), s * np.cos(theta), -8.0]])
    gy, gx = np.mgrid[0:256:32, 0:256:32]
    p_ctrl = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(float)
    d = np.stack([3.0 * np.sin(p_ctrl[:, 0] / 40.0),
                  2.0 * np.cos(p_ctrl[:, 1] / 50.0)], axis=1)
    return M, _fit_residual_field(p_ctrl, d, 1.0)


def _zero_field():
    gy, gx = np.mgrid[0:256:32, 0:256:32]
    px = gx.ravel() / 1000.0
    py = gy.ravel() / 1000.0
    n = len(px)
    return (np.zeros(n), np.zeros(3), np.zeros(n), np.zeros(3), px, py)


def _bilin(img, xs, ys):
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = np.clip(x0 + 1, 0, img.shape[1] - 1)
    y1 = np.clip(y0 + 1, 0, img.shape[0] - 1)
    x0 = np.clip(x0, 0, img.shape[1] - 1)
    y0 = np.clip(y0, 0, img.shape[0] - 1)
    fx = np.clip(xs - np.floor(xs), 0, 1)
    fy = np.clip(ys - np.floor(ys), 0, 1)
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def test_zero_field_reproduces_warp_affine():
    """Sampling-map convention check: zero field -> remap == warpAffine.

    This empirically pins the convention S(q) = A^-1 (q - t); a mismatch
    here would mean the field remap does not reproduce the pipeline's
    validated warpAffine behavior.
    """
    rng = np.random.default_rng(0)
    img = (rng.random((256, 256)) * 255).astype(np.uint8)
    M, _ = _synthetic_field()
    M = M  # only the matrix is needed here
    out_warp = cv2.warpAffine(img, M, (256, 256))
    mx, my = build_field_remap_maps((256, 256), (256, 256), M, _zero_field())
    out_remap = cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    assert np.abs(out_warp.astype(int) - out_remap.astype(int)).max() < 2


def test_fixed_point_inversion_accuracy():
    """The built maps invert F(p) = M_hook @ p + d(p) to << 1 px.

    For random secondary points p, q = F(p); bilinearly sampling the maps
    at q must recover p (inversion residual, no pixel-quantization noise).
    """
    rng = np.random.default_rng(11)
    M, field = _synthetic_field()
    mx, my = build_field_remap_maps((256, 256), (256, 256), M, field)
    p = rng.uniform(20, 236, (60, 2))
    A = M[:, :2]
    t = M[:, 2]
    q = (p @ A.T + t) + _eval_residual_field(field, p)
    inside = (q[:, 0] > 2) & (q[:, 0] < 253) & (q[:, 1] > 2) & (q[:, 1] < 253)
    p, q = p[inside], q[inside]
    assert len(p) > 30
    p_samp = np.stack([_bilin(mx, q[:, 0], q[:, 1]),
                       _bilin(my, q[:, 0], q[:, 1])], axis=1)
    resid = np.linalg.norm(p_samp - p, axis=1)
    assert resid.max() < 0.05, float(resid.max())


def test_build_maps_degenerate_inputs_raise():
    """Fail-closed: every degenerate input raises ValueError for the caller."""
    M, field = _synthetic_field()
    with pytest.raises(ValueError):
        build_field_remap_maps((256, 256), (256, 256), M, ("bad",))
    bad_field = list(field)
    bad_field[0] = np.full_like(np.asarray(field[0]), np.nan)
    with pytest.raises(ValueError):
        build_field_remap_maps((256, 256), (256, 256), M, tuple(bad_field))
    with pytest.raises(ValueError):  # singular linear part
        build_field_remap_maps((256, 256), (256, 256),
                               np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0]]), field)
    with pytest.raises(ValueError):  # non-positive shape
        build_field_remap_maps((0, 256), (256, 256), M, field)
    with pytest.raises(ValueError):  # wrong matrix shape
        build_field_remap_maps((256, 256), (256, 256), np.eye(3), field)


def test_build_maps_deterministic():
    M, field = _synthetic_field()
    a = build_field_remap_maps((256, 256), (256, 256), M, field)
    b = build_field_remap_maps((256, 256), (256, 256), M, field)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_fast_path_bounded_vs_exact():
    """Shipped path (float32, coarse grid) vs exact path: < 0.05 px map diff."""
    M, field = _synthetic_field()
    exact = build_field_remap_maps((256, 256), (256, 256), M, field, _exact=True)
    fast = build_field_remap_maps((256, 256), (256, 256), M, field)
    dx = np.abs(exact[0].astype(float) - fast[0].astype(float))
    dy = np.abs(exact[1].astype(float) - fast[1].astype(float))
    assert dx.max() < 0.05 and dy.max() < 0.05, (float(dx.max()), float(dy.max()))


def test_stage_returns_field_and_mhook():
    """apply_deform_field_stage carries the fitted forward map for export."""
    rng = np.random.default_rng(7)
    n = 120
    p1 = rng.uniform(0, 2000, (n, 2))
    M = np.array([[1.0, 0.0, 7.3], [0.0, 1.0, -3.9]])
    warp = np.stack([8.0 * np.sin(p1[:, 0] / 400.0),
                     6.0 * np.cos(p1[:, 1] / 500.0)], axis=1)
    p2 = (p1 @ M[:, :2].T + M[:, 2]) + warp + rng.normal(0, 0.2, (n, 2))
    out = apply_deform_field_stage(p1, p2, M)
    assert out["applied"] is True
    assert out["M_hook"].shape == (2, 3)
    wx, ax, wy, ay, px, py = out["field"]
    assert len(px) == n and len(wx) == n


# ---------------------------------------------------------------------------
# App-level tests (require the Gradio runtime; skip locally, run in CI).
# ---------------------------------------------------------------------------

def _require_app():
    pytest.importorskip("gradio", reason="application tests require the pinned Gradio runtime")
    pytest.importorskip("multipart", reason="application tests require python-multipart")
    import app as app_module
    return app_module


def _real_pair():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref = cv2.imread(os.path.join(root, "data", "benchmark_crops",
                                  "ohrc_01_reference.png"), cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(os.path.join(root, "data", "benchmark_crops",
                                  "ohrc_01_source.png"), cv2.IMREAD_GRAYSCALE)
    assert ref is not None and sec is not None
    return ref, sec


def _set_flag(value):
    old = os.environ.get("CHANDRA_DEFORM_FIELD")
    if value is None:
        os.environ.pop("CHANDRA_DEFORM_FIELD", None)
    else:
        os.environ["CHANDRA_DEFORM_FIELD"] = value
    return old


def _restore_flag(old):
    if old is None:
        os.environ.pop("CHANDRA_DEFORM_FIELD", None)
    else:
        os.environ["CHANDRA_DEFORM_FIELD"] = old


def test_flag_off_export_untouched():
    """Flag off: the new code is never reached; export bit-identical."""
    app_module = _require_app()
    import chandra_align.deform_field as df_mod
    ref, sec = _real_pair()
    old = _set_flag(None)
    try:
        out_plain = app_module._align_core(ref, sec)
        orig = df_mod.build_field_remap_maps
        def _raiser(*a, **k):
            raise AssertionError("flag-off path must not build field maps")
        df_mod.build_field_remap_maps = _raiser
        try:
            out_patched = app_module._align_core(ref, sec)
        finally:
            df_mod.build_field_remap_maps = orig
    finally:
        _restore_flag(old)
    assert out_plain["judge_metrics"]["warp_export_kind"] == "affine"
    assert out_patched["judge_metrics"]["warp_export_kind"] == "affine"
    assert np.array_equal(out_plain["warped_sec"], out_patched["warped_sec"])


def test_flag_on_field_remap_exported():
    """Flag on + stage applied: export carries the field; differs from affine."""
    app_module = _require_app()
    import chandra_align.deform_field as df_mod
    ref, sec = _real_pair()
    old = _set_flag("1")
    try:
        out = app_module._align_core(ref, sec)
        assert out["judge_metrics"]["deform_field_stage"]["applied"] is True
        assert out["judge_metrics"]["warp_export_kind"] == "field_remap"
        w = out["warped_sec"]
        assert np.all(np.isfinite(w))
        orig = df_mod.build_field_remap_maps
        def _raiser(*a, **k):
            raise RuntimeError("sabotaged for fallback test")
        df_mod.build_field_remap_maps = _raiser
        try:
            out_fb = app_module._align_core(ref, sec)
        finally:
            df_mod.build_field_remap_maps = orig
    finally:
        _restore_flag(old)
    assert out_fb["judge_metrics"]["warp_export_kind"].startswith(
        "affine (field remap failed")
    assert np.all(np.isfinite(out_fb["warped_sec"]))
    # The field actually shipped: the remap differs from the affine fallback.
    assert not np.array_equal(w, out_fb["warped_sec"])
    assert np.abs(w.astype(float) - out_fb["warped_sec"].astype(float)).max() > 0


def test_flag_on_export_deterministic():
    """Two flag-on runs produce byte-identical exported rasters."""
    app_module = _require_app()
    ref, sec = _real_pair()
    old = _set_flag("1")
    try:
        a = app_module._align_core(ref, sec)["warped_sec"]
        b = app_module._align_core(ref, sec)["warped_sec"]
    finally:
        _restore_flag(old)
    assert np.array_equal(a, b)
