"""No-crash regression tests for DEGENERATE / fail-closed pipeline paths.

Standing rule under test: the pipeline must never fail catastrophically and
must never fabricate success. Adversarial or unsupported inputs must produce
an honest fail-closed verdict (DEGENERATE_FAILURE or an ABSTAIN code), never
an exception escaping the public entry point and never a fabricated
ACCEPT/SUCCESS. On rejection, transform telemetry is masked (no garbage
transform is presented as real).

Entry points exercised:
- chandra_align.matcher.SIFTMatcher / run_cascade (matching stage)
- chandra_align.refine.verify_magsac / verify_guarded (verification stage)
- chandra_align.metrics.conditioning.classify_prefit_abstain /
  check_transform_conditioning (fail-closed codes)
- chandra_align.metrics.validate_registration_gate (frozen three-tier gate)
- an in-memory mirror of scripts/run_pipeline.py stages
  (match -> verify -> refine -> quadrant metrics -> gate)
"""

import numpy as np
import pytest
import cv2

cv2.setRNGSeed(42)

from chandra_align.alignment import decompose_partial_affine
from chandra_align.matcher import SIFTMatcher, run_cascade
from chandra_align.metrics import apply_transform, validate_registration_gate
from chandra_align.metrics.conditioning import (
    check_transform_conditioning,
    classify_prefit_abstain,
)
from chandra_align.metrics.quadrant import compute_quadrant_metrics
from chandra_align.refine import (
    refine_subpixel_ncc,
    uniformity_score,
    verify_guarded,
    verify_magsac,
)
from chandra_align.trust import evaluate as trust_evaluate, trust_flag
from chandra_align.metrics import rmse_heldout

# Frozen gate constants (mirrors scripts/run_scale_ratio.py and the gate code).
_MIN_INLIERS = 8
_CFG_VERIFY = {"ransac_reproj_threshold": 3.0, "max_iters": 2000, "confidence": 0.99}
_CFG_MATCHER = {
    "tier1_matcher": "sift",
    "tier2_escalation": {"matcher": "sift", "min_raw_matches": 4,
                         "min_inlier_pairs": 4},
}

ACCEPT_CODES = {"SUCCESS_SUBPIXEL", "COARSE_ADVISORY"}
REJECT_CODES = {"DEGENERATE_FAILURE"}


# ---------------------------------------------------------------------------
# Adversarial image builders (deterministic)
# ---------------------------------------------------------------------------

def _blank(shape=(256, 256)):
    return np.zeros(shape, np.uint8)


def _constant(shape=(256, 256), value=128):
    return np.full(shape, value, np.uint8)


def _noise(shape=(256, 256), seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, shape).astype(np.uint8)


def _textured(shape=(256, 256), seed=7):
    """Crater-like blob texture spanning the whole frame."""
    rng = np.random.default_rng(seed)
    img = np.zeros(shape, np.float32)
    h, w = shape
    for scale, count, amp in [(24, 30, 0.5), (10, 120, 0.7), (4, 400, 1.0)]:
        ys = rng.integers(0, h, count)
        xs = rng.integers(0, w, count)
        yy, xx = np.ogrid[-scale:scale + 1, -scale:scale + 1]
        disc = (xx ** 2 + yy ** 2 <= scale ** 2).astype(np.float32)
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - scale), min(h, y + scale + 1)
            x0, x1 = max(0, x - scale), min(w, x + scale + 1)
            img[y0:y1, x0:x1] += amp * disc[y0 - (y - scale):y1 - (y - scale),
                                            x0 - (x - scale):x1 - (x - scale)]
    img += rng.normal(0, 0.05, shape).astype(np.float32)
    img = np.clip(img, 0, None)
    return (255 * img / img.max()).astype(np.uint8)


def _single_quadrant_texture(shape=(256, 256), seed=7):
    """Texture confined to the top-left quadrant; everything else blank."""
    img = _textured(shape, seed=seed)
    h, w = shape
    img[h // 2:, :] = 0
    img[:, w // 2:] = 0
    return img


def _shift(img, dx, dy):
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], np.float32)
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------------------
# Pipeline helper: in-memory mirror of scripts/run_pipeline.py stages
# ---------------------------------------------------------------------------

def _run_pipeline(img_a, img_b):
    """Match -> RANSAC verify -> NCC refine -> quadrant metrics -> frozen gate.

    Returns a dict with status_code, status_message, rmse_px (the gate basis;
    None/masked on rejection), n_inliers, model (None on rejection), and
    telemetry (None on rejection — never a garbage transform).
    """
    result = {
        "n_raw": 0, "n_inliers": 0, "model": None, "telemetry": None,
        "status_code": "DEGENERATE_FAILURE",
        "status_message": "REJECTED (uninitialised)",
        "rmse_px": None,
    }
    pts_a, pts_b = SIFTMatcher().match(img_a, img_b)
    result["n_raw"] = int(pts_a.shape[0])
    inl_a, inl_b, M, _ratio = verify_magsac(pts_a, pts_b, _CFG_VERIFY)
    if M is None:
        result["status_message"] = "REJECTED (no verified model)"
        return result
    ref_a, ref_b, _stats = refine_subpixel_ncc(
        np.asarray(img_a), np.asarray(img_b), inl_a, inl_b, ncc_window=11)
    if len(ref_a) == 0:
        result["status_message"] = "REJECTED (no refined correspondences)"
        return result
    result["n_inliers"] = int(len(ref_a))
    residuals = np.hypot(*(apply_transform(M, ref_a) - ref_b).T)
    quad_metrics, entropy = compute_quadrant_metrics(
        ref_a, residuals, np.asarray(img_a).shape[:2])
    rmse = float(np.sqrt((residuals ** 2).mean()))
    msg, code = validate_registration_gate(
        rmse, len(ref_a), _MIN_INLIERS, entropy, quad_metrics, model=M)
    result["status_message"] = msg
    result["status_code"] = code
    if code == "DEGENERATE_FAILURE":
        # Telemetry stays masked: no garbage transform leaves this function.
        return result
    result["rmse_px"] = rmse
    result["model"] = M
    result["telemetry"] = decompose_partial_affine(M)
    return result


def _assert_honest(result):
    """A verdict is honest iff an ACCEPT is fully supported by measurement and
    a REJECT carries no transform telemetry."""
    code = result["status_code"]
    if code in ACCEPT_CODES:
        limit = 0.50 if code == "SUCCESS_SUBPIXEL" else 2.50
        assert result["rmse_px"] is not None, "ACCEPT with masked RMSE"
        assert result["rmse_px"] <= limit, (
            f"fabricated {code}: rmse {result['rmse_px']} exceeds {limit}")
        assert result["n_inliers"] >= _MIN_INLIERS, "ACCEPT below inlier floor"
        assert result["model"] is not None and np.all(np.isfinite(result["model"]))
        assert result["telemetry"] is not None
        assert all(np.isfinite(v) for v in result["telemetry"].values())
    elif code in REJECT_CODES:
        assert result["model"] is None, "reject leaks a garbage transform"
        assert result["telemetry"] is None, "reject leaks transform telemetry"
        assert result["rmse_px"] is None, "reject presents an RMSE as success"
    else:
        pytest.fail(f"unknown status code {code!r}")


# ---------------------------------------------------------------------------
# End-to-end adversarial image pairs
# ---------------------------------------------------------------------------

class TestAdversarialPairs:
    def test_blank_pair_rejects_honestly(self):
        r = _run_pipeline(_blank(), _blank())
        assert r["status_code"] == "DEGENERATE_FAILURE"
        _assert_honest(r)

    def test_constant_pair_rejects_honestly(self):
        r = _run_pipeline(_constant(value=128), _constant(value=200))
        assert r["status_code"] == "DEGENERATE_FAILURE"
        _assert_honest(r)

    def test_pure_noise_pair_rejects_honestly(self):
        r = _run_pipeline(_noise(seed=1), _noise(seed=2))
        assert r["status_code"] == "DEGENERATE_FAILURE"
        _assert_honest(r)

    def test_single_quadrant_texture_rejects_honestly(self):
        tex = _single_quadrant_texture()
        r = _run_pipeline(tex, _shift(tex, 2.0, -1.0))
        # A one-quadrant cluster can never satisfy the spatial gates.
        assert r["status_code"] == "DEGENERATE_FAILURE"
        _assert_honest(r)

    def test_tiny_blank_pair_no_crash(self):
        r = _run_pipeline(_blank((32, 32)), _blank((32, 32)))
        assert r["status_code"] == "DEGENERATE_FAILURE"
        _assert_honest(r)

    def test_tiny_textured_pair_no_crash_honest_verdict(self):
        tex = _textured((32, 32), seed=11)
        r = _run_pipeline(tex, _shift(tex, 1.5, -1.0))
        _assert_honest(r)

    def test_empty_images_no_crash(self):
        e = np.zeros((0, 0), np.uint8)
        r = _run_pipeline(e, e)
        assert r["status_code"] == "DEGENERATE_FAILURE"
        _assert_honest(r)

    def test_nan_inf_images_no_crash(self):
        nan = np.full((64, 64), np.nan)
        inf = np.full((64, 64), np.inf)
        for a, b in [(nan, nan), (inf, inf), (nan, inf)]:
            r = _run_pipeline(a, b)
            assert r["status_code"] == "DEGENERATE_FAILURE"
            _assert_honest(r)

    def test_no_overlap_shift_rejects_honestly(self):
        tex = _textured()
        r = _run_pipeline(tex, _shift(tex, 500.0, 500.0))
        assert r["status_code"] == "DEGENERATE_FAILURE"
        _assert_honest(r)

    def test_identical_pair_honest_accept(self):
        from chandra_align.testing import make_pair_shift
        tex, _, _ = make_pair_shift(shape=(256, 256), seed=7)
        r = _run_pipeline(tex, tex.copy())
        # A genuinely perfect alignment may honestly ACCEPT; the RMSE must
        # actually support the tier (no fabricated success).
        assert r["status_code"] == "SUCCESS_SUBPIXEL"
        assert r["rmse_px"] <= 0.50
        _assert_honest(r)

    def test_valid_shifted_pair_control(self):
        from chandra_align.testing import make_pair_shift
        tex, mov, _M = make_pair_shift(shape=(512, 512), dx=7.3, dy=-3.9,
                                       angle_deg=0.4, seed=7)
        r = _run_pipeline(tex, mov)
        _assert_honest(r)
        # Control sanity: a real textured pair with a small shift must not
        # degenerate — if it does, the harness itself is broken.
        assert r["status_code"] in ACCEPT_CODES
        assert r["model"] is not None

    def test_reject_masks_transform_telemetry(self):
        r = _run_pipeline(_blank(), _blank())
        assert r["status_code"] == "DEGENERATE_FAILURE"
        assert r["model"] is None
        assert r["telemetry"] is None
        assert r["rmse_px"] is None


# ---------------------------------------------------------------------------
# Matcher-stage no-crash
# ---------------------------------------------------------------------------

class TestMatcherNoCrash:
    @pytest.mark.parametrize("img", [
        np.zeros((256, 256), np.uint8),                    # blank
        np.full((256, 256), 128, np.uint8),                # constant
        np.zeros((0, 0), np.uint8),                        # empty
        np.zeros((1, 1), np.uint8),                        # 1x1
        np.zeros((1, 64), np.uint8),                       # degenerate row
        np.full((64, 64), np.nan),                         # all-NaN
        np.full((64, 64), np.inf),                         # all-Inf
        np.full((64, 64, 3), 200, np.uint8),               # 3-channel constant
    ], ids=["blank", "constant", "empty", "1x1", "row",
            "nan", "inf", "rgb-constant"])
    def test_sift_matcher_never_raises(self, img):
        pa, pb = SIFTMatcher().match(img, img)
        assert pa.shape == (0, 2) and pb.shape == (0, 2)

    def test_run_cascade_adversarial_no_crash(self):
        for a, b in [(_blank(), _blank()), (_noise(seed=1), _noise(seed=2)),
                     (_blank((32, 32)), _blank((32, 32)))]:
            pa, pb, name, escalated = run_cascade(a, b, _CFG_MATCHER)
            assert pa.shape[0] == pb.shape[0]
            assert isinstance(name, str)


# ---------------------------------------------------------------------------
# Verification-stage fail-closed codes
# ---------------------------------------------------------------------------

class TestVerifyFailClosed:
    def test_verify_magsac_empty_inputs(self):
        e = np.zeros((0, 2), np.float32)
        inl_a, inl_b, M, ratio = verify_magsac(e, e, _CFG_VERIFY)
        assert M is None and ratio == 0.0
        assert len(inl_a) == 0 and len(inl_b) == 0

    def test_verify_magsac_too_few_points(self):
        for n in (1, 2):
            pa = np.zeros((n, 2), np.float32)
            _a, _b, M, ratio = verify_magsac(pa, pa, _CFG_VERIFY)
            assert M is None and ratio == 0.0

    def test_verify_magsac_none_inputs_fail_closed(self):
        _a, _b, M, ratio = verify_magsac(None, None, _CFG_VERIFY)
        assert M is None and ratio == 0.0

    def test_verify_guarded_none_inputs_fail_closed(self):
        out = verify_guarded(None, None, _CFG_VERIFY)
        assert out["abstain_code"] == "ZERO_CANDIDATES"
        assert out["model"] is None and out["ok"] is False

    def test_verify_guarded_zero_candidates(self):
        e = np.zeros((0, 2), np.float32)
        out = verify_guarded(e, e, _CFG_VERIFY)
        assert out["abstain_code"] == "ZERO_CANDIDATES"
        assert out["model"] is None and out["ok"] is False

    def test_verify_guarded_mismatched_lengths(self):
        out = verify_guarded(np.zeros((3, 2), np.float32),
                             np.zeros((5, 2), np.float32), _CFG_VERIFY)
        assert out["abstain_code"] == "ZERO_CANDIDATES"
        assert out["model"] is None

    def test_verify_guarded_insufficient_unique(self):
        dup = np.tile(np.array([[5.0, 5.0]], np.float32), (8, 1))
        out = verify_guarded(dup, dup, _CFG_VERIFY)
        assert out["abstain_code"] == "INSUFFICIENT_UNIQUE"
        assert out["model"] is None and out["ok"] is False

    def test_verify_guarded_collinear(self):
        rng = np.random.default_rng(3)
        line = np.column_stack([
            np.linspace(0, 100, 20),
            np.linspace(0, 100, 20) + rng.normal(0, 1e-9, 20),
        ]).astype(np.float32)
        out = verify_guarded(line, line, _CFG_VERIFY)
        assert out["abstain_code"] == "COLLINEAR"
        assert out["model"] is None and out["ok"] is False

    def test_verify_guarded_ill_conditioned(self):
        rng = np.random.default_rng(5)
        pts = (1e7 + rng.uniform(0, 5, (10, 2))).astype(np.float32)
        out = verify_guarded(pts, pts, _CFG_VERIFY)
        assert out["abstain_code"] == "ILL_CONDITIONED"
        assert out["model"] is None and out["ok"] is False

    def test_verify_guarded_garbage_correspondences_never_accept(self):
        rng = np.random.default_rng(9)
        pa = rng.uniform(0, 256, (40, 2)).astype(np.float32)
        pb = rng.uniform(0, 256, (40, 2)).astype(np.float32)
        out = verify_guarded(pa, pb, _CFG_VERIFY)
        if out["ok"]:
            # A fit was attempted: the gate must still reject it downstream.
            msg, code = validate_registration_gate(
                99.0, out["n_unique"], _MIN_INLIERS, 0.0, {},
                model=out["model"])
            assert code == "DEGENERATE_FAILURE"
        else:
            assert out["abstain_code"] in {
                "ZERO_CANDIDATES", "INSUFFICIENT_UNIQUE", "COLLINEAR",
                "ILL_CONDITIONED", "NO_VALID_MODEL"}


# ---------------------------------------------------------------------------
# Prefit abstain classifier codes (unit level)
# ---------------------------------------------------------------------------

class TestPrefitAbstainCodes:
    def test_none_is_zero_candidates(self):
        assert classify_prefit_abstain(None, None) == "ZERO_CANDIDATES"

    def test_empty_is_zero_candidates(self):
        e = np.zeros((0, 2))
        assert classify_prefit_abstain(e, e) == "ZERO_CANDIDATES"

    def test_mismatched_is_zero_candidates(self):
        assert classify_prefit_abstain(
            np.zeros((3, 2)), np.zeros((5, 2))) == "ZERO_CANDIDATES"

    def test_duplicates_are_insufficient_unique(self):
        dup = np.tile(np.array([[5.0, 5.0]]), (8, 1))
        assert classify_prefit_abstain(dup, dup) == "INSUFFICIENT_UNIQUE"

    def test_collinear(self):
        rng = np.random.default_rng(3)
        line = np.column_stack([
            np.linspace(0, 100, 20),
            np.linspace(0, 100, 20) + rng.normal(0, 1e-9, 20)])
        assert classify_prefit_abstain(line, line) == "COLLINEAR"

    def test_ill_conditioned(self):
        rng = np.random.default_rng(5)
        pts = 1e7 + rng.uniform(0, 5, (10, 2))
        assert classify_prefit_abstain(pts, pts) == "ILL_CONDITIONED"

    def test_healthy_points_pass(self):
        rng = np.random.default_rng(6)
        pts = rng.uniform(0, 256, (20, 2))
        assert classify_prefit_abstain(pts, pts) is None


# ---------------------------------------------------------------------------
# Gate honesty: never fabricate, Gate 3 backstop
# ---------------------------------------------------------------------------

class TestGateHonesty:
    @pytest.mark.parametrize("rmse", ["UNMEASURED", None, float("nan"),
                                      float("inf"), 99.0])
    def test_garbage_rmse_always_rejects(self, rmse):
        msg, code = validate_registration_gate(rmse, 0, _MIN_INLIERS, 0.0, {})
        assert code == "DEGENERATE_FAILURE"
        assert "REJECTED" in msg

    def test_zero_inliers_never_accepts(self):
        msg, code = validate_registration_gate(
            0.01, 0, _MIN_INLIERS, 1.9,
            {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0})
        assert code == "DEGENERATE_FAILURE"

    def test_gate3_rejects_nan_transform(self):
        msg, code = validate_registration_gate(
            0.1, 100, _MIN_INLIERS, 1.9,
            {"Q1": 25, "Q2": 25, "Q3": 25, "Q4": 25},
            model=np.full((2, 3), np.nan))
        assert code == "DEGENERATE_FAILURE"

    def test_gate3_rejects_scale_collapse(self):
        msg, code = validate_registration_gate(
            0.1, 100, _MIN_INLIERS, 1.9,
            {"Q1": 25, "Q2": 25, "Q3": 25, "Q4": 25},
            model=np.array([[1e9, 0, 0], [0, 1e-9, 0]]))
        assert code == "DEGENERATE_FAILURE"

    def test_gate3_rejects_projective_transform(self):
        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.1, 0.0, 1.0]])
        msg, code = validate_registration_gate(
            0.1, 100, _MIN_INLIERS, 1.9,
            {"Q1": 25, "Q2": 25, "Q3": 25, "Q4": 25}, model=M)
        assert code == "DEGENERATE_FAILURE"

    def test_gate3_rejects_malformed_transform(self):
        msg, code = validate_registration_gate(
            0.1, 100, _MIN_INLIERS, 1.9,
            {"Q1": 25, "Q2": 25, "Q3": 25, "Q4": 25},
            model=np.eye(2))
        assert code == "DEGENERATE_FAILURE"

    def test_gate3_keeps_honest_accept(self):
        msg, code = validate_registration_gate(
            0.1, 100, _MIN_INLIERS, 1.9,
            {"Q1": 25, "Q2": 25, "Q3": 25, "Q4": 25},
            model=np.eye(2, 3))
        assert code == "SUCCESS_SUBPIXEL"

    def test_conditioning_fail_closed(self):
        ok, _ = check_transform_conditioning(None)
        assert ok is False
        ok, _ = check_transform_conditioning(np.full((2, 3), np.nan))
        assert ok is False
        ok, _ = check_transform_conditioning(np.full((2, 3), np.inf))
        assert ok is False
        ok, _ = check_transform_conditioning(np.eye(2, 3))
        assert ok is True

    def test_decompose_rejects_garbage_instead_of_fabricating(self):
        with pytest.raises(ValueError):
            decompose_partial_affine(np.full((2, 3), np.nan))


# ---------------------------------------------------------------------------
# Trust-layer guardrails
# ---------------------------------------------------------------------------

class TestTrustGuardrails:
    def test_evaluate_empty_is_unmeasured_not_crash(self):
        e = np.zeros((0, 2), np.float32)
        M, rm = trust_evaluate(None, e, e)
        assert M is None
        assert rm["rmse_px"] == "UNMEASURED"

    def test_rmse_heldout_refuses_empty_set(self):
        with pytest.raises(ValueError):
            rmse_heldout(np.eye(2, 3), np.zeros((0, 2)), np.zeros((0, 2)))

    def test_trust_flag_unmeasured_never_trusted(self):
        assert trust_flag("UNMEASURED", 0.9, 0.9, {}) == "Not Trusted"
        assert trust_flag(None, 0.9, 0.9, {}) == "Not Trusted"

    def test_uniformity_empty_is_zero(self):
        assert uniformity_score(np.zeros((0, 2), np.float32), (256, 256)) == 0.0
