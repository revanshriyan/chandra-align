"""Phase 9 cross-modal toolkit tests — synthetic only, no torch."""

import numpy as np
import pytest

from chandra_align.testing import make_pair_shift
from chandra_align.xmodal import funnel, goa_nmi, gsd, sift_rescue, tps, wallis


def _rng_scene(h=128, w=128, seed=11):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(float)
    img = (np.sin(0.11 * x + 0.05 * y) + 0.6 * np.sin(0.031 * y - 0.07 * x)
           + 0.25 * rng.normal(0, 1, (h, w)))
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    return img


# ---------------- wallis ----------------

def test_wallis_moves_std_toward_target():
    rng = np.random.default_rng(3)
    low = 100.0 + 5.0 * rng.normal(0, 1, (128, 128))  # std ~5, far from 40
    out = wallis.wallis_normalize(low, window=21, target_mean=127.0, target_std=40.0)
    assert out.shape == low.shape
    assert out.min() >= 0 and out.max() <= 255
    assert abs(out.std() - 40.0) < abs(low.std() - 40.0)


def test_wallis_rejects_bad_window():
    with pytest.raises(ValueError):
        wallis.wallis_normalize(np.zeros((16, 16)), window=1)


# ---------------- gsd ----------------

def test_gsd_raises_on_unknown():
    img = np.zeros((32, 32))
    with pytest.raises(ValueError, match="GSD unknown"):
        gsd.to_common_gsd(img, None, 2.0)
    with pytest.raises(ValueError, match="GSD unknown"):
        gsd.to_common_gsd(img, 4.0, None)
    with pytest.raises(ValueError, match="GSD unknown"):
        gsd.to_common_gsd(img, 0.0, 2.0)
    with pytest.raises(ValueError, match="GSD unknown"):
        gsd.to_common_gsd(img, 4.0, -1.0)


def test_gsd_downsample_shape_and_scale():
    img = _rng_scene(100, 100)
    out, scale = gsd.to_common_gsd(img, 4.0, 8.0)  # coarser target: shrink
    assert scale == pytest.approx(0.5)
    assert out.shape == (50, 50)


def test_gsd_upsample_shape_and_scale():
    img = _rng_scene(64, 64)
    out, scale = gsd.to_common_gsd(img, 8.0, 4.0)  # finer target: grow
    assert scale == pytest.approx(2.0)
    assert out.shape == (128, 128)


# ---------------- tps ----------------

def test_tps_exact_interpolation_lmbda0():
    rng = np.random.default_rng(5)
    pa = rng.uniform(0, 100, (8, 2))
    # known smooth warp: affine + mild quadratic bend
    pb = pa @ np.array([[1.1, 0.05], [-0.03, 0.9]]) + np.array([3.0, -2.0])
    pb = pb + 0.002 * (pa ** 2).sum(axis=1, keepdims=True)
    fit = tps.fit_tps(pa, pb, lmbda=0.0)
    resid, rmse = tps.tps_residual(pa, pb, fit["warp"])
    assert rmse < 1e-6, f"lmbda=0 must interpolate, rmse={rmse}"
    assert fit["n_points"] == 8


def test_tps_smoothing_reduces_bending_energy():
    rng = np.random.default_rng(9)
    pa = rng.uniform(0, 100, (10, 2))
    pb = pa + rng.normal(0, 2.0, (10, 2))  # noisy: smoothing should help
    e0 = tps.fit_tps(pa, pb, lmbda=0.0)["bending_energy"]
    e1 = tps.fit_tps(pa, pb, lmbda=0.05)["bending_energy"]
    assert e1 <= e0


def test_tps_warp_callable_shape():
    rng = np.random.default_rng(13)
    pa = rng.uniform(0, 50, (6, 2))
    pb = pa + np.array([5.0, -3.0])
    fit = tps.fit_tps(pa, pb, lmbda=0.05)
    q = rng.uniform(0, 50, (20, 2))
    assert fit["warp"](q).shape == (20, 2)


# ---------------- goa_nmi ----------------

def test_nmi_identical_patches_near_one():
    patch = _rng_scene(32, 32, seed=21)
    assert goa_nmi.nmi_score(patch, patch) == pytest.approx(1.0, abs=1e-6)


def test_nmi_unrelated_patches_low():
    rng = np.random.default_rng(23)
    a = rng.uniform(0, 255, (32, 32))
    b = rng.uniform(0, 255, (32, 32))
    assert goa_nmi.nmi_score(a, b) < 0.5


def test_nmi_constant_patch_zero():
    a = np.full((16, 16), 128.0)
    b = _rng_scene(16, 16, seed=27)
    assert goa_nmi.nmi_score(a, b) == 0.0


def test_goa_points_within_image():
    img = _rng_scene(128, 128, seed=31)
    pts = goa_nmi.gradient_magnitude_points(img, n=100)
    assert pts.ndim == 2 and pts.shape[1] == 2
    assert len(pts) <= 100
    assert len(pts) > 0
    assert np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] <= 127)
    assert np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] <= 127)


def test_nmi_rerank_prefers_true_pair():
    img = _rng_scene(128, 128, seed=37)
    # pair 0: same location -> NMI ~1; pair 1: far offset -> low NMI
    cands_a = np.array([[64.0, 64.0], [64.0, 64.0]])
    cands_b = np.array([[64.0, 64.0], [10.0, 100.0]])
    order, scores = goa_nmi.nmi_rerank(cands_a, cands_b, img, img, window=21)
    assert order[0] == 0
    assert scores[0] > scores[1]


# ---------------- funnel ----------------

def test_funnel_returns_all_stage_counts():
    def detect_fn(img):
        pts = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
        return {"points": pts, "descriptors": np.zeros((3, 8))}

    def describe_match_fn(det_a, det_b):
        return {"pairs": np.array([[0, 0], [1, 1], [2, 2]]), "scores": np.array([0.5, 0.6, 0.7])}

    def ransac_fn(pa, pb):
        return {"inliers": np.array([True, True, False]),
                "model": np.eye(2, 3), "inlier_rank": {"note": "stub"}}

    img = np.zeros((64, 64))
    out = funnel.diagnose_funnel(detect_fn, describe_match_fn, ransac_fn, img, img)
    assert out["n_detected_a"] == 3
    assert out["n_detected_b"] == 3
    assert out["n_descriptors_a"] == 3
    assert out["n_descriptors_b"] == 3
    assert out["n_raw_matches"] == 3
    assert out["n_ransac_inliers"] == 2
    assert out["inlier_rank"] == {"note": "stub"}


def test_funnel_empty_detection_stays_zero():
    def detect_fn(img):
        return {"points": np.zeros((0, 2))}
    out = funnel.diagnose_funnel(
        detect_fn, lambda a, b: {"pairs": []}, lambda pa, pb: {}, np.zeros((32, 32)), np.zeros((32, 32)))
    assert out["n_detected_a"] == 0
    assert out["n_raw_matches"] == 0
    assert out["n_ransac_inliers"] == 0
    assert out["inlier_rank"] is None


# ---------------- sift_rescue ----------------

def test_sift_rescue_returns_verdict_dict():
    ref, mov, _ = make_pair_shift(shape=(256, 256), dx=7.3, dy=-3.9, seed=7)
    out = sift_rescue.sift_rescue_tps(ref, mov, np.zeros((0, 2)), np.zeros((0, 2)))
    assert isinstance(out, dict)
    assert out["verdict"] in ("TPS_RESCUED", "RESCUE_FAILED")
    assert "n_inliers" in out and "tps" in out and "gate3" in out
    if out["verdict"] == "TPS_RESCUED":
        assert out["tps"] is not None and out["n_inliers"] >= 13
        warped = out["tps"]["warp"](np.array([[10.0, 10.0]]))
        assert warped.shape == (1, 2) and np.all(np.isfinite(warped))
    else:
        assert out["tps"] is None


def test_sift_rescue_fails_closed_on_blank():
    blank = np.full((128, 128), 128.0)
    out = sift_rescue.sift_rescue_tps(blank, blank, np.zeros((0, 2)), np.zeros((0, 2)))
    assert out["verdict"] == "RESCUE_FAILED"
    assert out["tps"] is None
