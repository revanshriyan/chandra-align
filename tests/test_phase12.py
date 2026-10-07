"""Phase 12 tests — confidence calibration (deterministic, no sklearn)."""

import os
import numpy as np
import pytest

from chandra_align.trust.calibration import (
    MIN_INLIERS_DEFAULT,
    brier_score,
    confidence_features,
    fit_logistic_regression,
    format_upper_bound,
    is_calibrated,
    predict_proba,
    reliability_diagram,
)


def test_features_match_heuristic_formulas():
    # inliers=16 (>= 2x min 8) -> support 1.0; ratio 16/32=0.5;
    # entropy 1.5 -> spread 0.75; rmse 1.0 -> quality 0.6
    f = confidence_features(16, 32, 1.5, 1.0)
    assert f == pytest.approx((1.0, 0.5, 0.75, 0.6))


def test_features_degenerate_inputs():
    f = confidence_features(0, 0, 0.0, None)
    assert f == (0.0, 0.0, 0.0, 0.0)
    f = confidence_features("bad", "bad", "bad", "bad")
    assert f == (0.0, 0.0, 0.0, 0.0)


def test_irls_deterministic():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (40, 4))
    y = (X[:, 0] + X[:, 3] > 1.0).astype(float)
    a = fit_logistic_regression(X, y)
    b = fit_logistic_regression(X, y)
    assert a["weights"] == pytest.approx(b["weights"])
    assert a["intercept"] == pytest.approx(b["intercept"])


def test_irls_perfect_separation():
    # two well-separated clusters -> confident, correct probabilities
    # (ridge l2=1.0 shrinks weights, so 0.8/0.2 not 0.9/0.1 — documented)
    X = np.array([[0.9, 0.9, 0.9, 0.9]] * 10 + [[0.1, 0.1, 0.1, 0.1]] * 10)
    y = np.array([1.0] * 10 + [0.0] * 10)
    fit = fit_logistic_regression(X, y)
    p = predict_proba(X, fit["weights"], fit["intercept"])
    assert np.all(p[:10] > 0.8)
    assert np.all(p[10:] < 0.2)


def test_brier_bounds():
    assert brier_score([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2]) == pytest.approx(0.025)
    assert 0.0 <= brier_score([0, 1], [0.5, 0.5]) <= 1.0
    with pytest.raises(ValueError):
        brier_score([0, 1], [1.5, 0.5])


def test_reliability_diagram_empty_bins_honest():
    y = np.array([1.0, 0.0])
    p = np.array([0.05, 0.95])
    bins = reliability_diagram(y, p, n_bins=10)
    assert len(bins) == 10
    empty = [b for b in bins if b["n"] == 0]
    assert len(empty) == 8
    assert all(b["observed_rate"] is None for b in empty)


def test_is_calibrated_flags_miscalibration():
    y = np.array([1.0] * 10 + [0.0] * 10)
    p = np.array([0.1] * 10 + [0.9] * 10)  # inverted: badly miscalibrated
    bins = reliability_diagram(y, p, n_bins=2)
    ok, detail = is_calibrated(bins)
    assert ok is False
    assert "worst" in detail


def test_upper_bound_framing():
    s = format_upper_bound(0.3840)
    assert s == "true error <= 1.38 px (includes 1 px marking noise)"
    assert format_upper_bound(None) is None
    assert format_upper_bound(float("nan")) is None


def test_gates_do_not_import_calibration():
    # the calibration module must never influence gate decisions:
    # no gate/metrics path may import it
    import subprocess
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(
        ["grep", "-rn", "trust.calibration\\|trust\\.calibration\\|from .calibration\\|from chandra_align.trust.calibration",
         "chandra_align/metrics", "chandra_align/refine", "--include=*.py"],
        capture_output=True, text=True, cwd=repo_root)
    assert out.stdout.strip() == "", f"gate path imports calibration: {out.stdout}"
