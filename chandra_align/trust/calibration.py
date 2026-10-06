"""Phase 12 — confidence calibration.

The 0-100 confidence heuristic in chandra_align/metrics/reporting.py is an
evidence index with guessed weights (30/25/20/25), explicitly documented as
NOT a probability. This module calibrates it: fit a logistic map from the
same four features to measured ACCEPT outcomes, so the output means
P(SUCCESS_SUBPIXEL | features).

Hard rule preserved: calibration NEVER influences gate decisions. The gates
in chandra_align/metrics decide; this module only scores reported outcomes
after the fact. Nothing here is imported by any gate path.

Fit method: logistic regression via IRLS (iteratively reweighted least
squares) in numpy — deterministic, no sklearn dependency. Equivalent to
Platt scaling on the four confidence features.
"""

import numpy as np

MIN_INLIERS_DEFAULT = 8
GATE_RMSE_REF_PX = 2.5  # same reference as build_confidence_assessment


def confidence_features(inliers, correspondences, entropy, rmse_px,
                        min_inliers=MIN_INLIERS_DEFAULT):
    """The four confidence features, identical formulas to the heuristic.

    Returns (support, inlier_ratio, spread, residual_quality), each in [0, 1].
    """
    try:
        inl = max(0, int(inliers))
    except (TypeError, ValueError):
        inl = 0
    try:
        corr = max(0, int(correspondences))
    except (TypeError, ValueError):
        corr = 0
    try:
        ent = float(entropy)
        if not np.isfinite(ent):
            ent = 0.0
    except (TypeError, ValueError):
        ent = 0.0
    try:
        rmse = float(rmse_px)
        if not np.isfinite(rmse) or rmse < 0:
            rmse = None
    except (TypeError, ValueError):
        rmse = None

    support = min(inl / max(int(min_inliers or 1), 1), 1.0)
    inlier_ratio = inl / corr if corr else 0.0
    spread = min(max(ent / 2.0, 0.0), 1.0)
    residual_quality = max(0.0, 1.0 - rmse / GATE_RMSE_REF_PX) if rmse is not None else 0.0
    return (float(support), float(inlier_ratio), float(spread),
            float(residual_quality))


def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def fit_logistic_regression(X, y, l2=1.0, max_iter=200, tol=1e-8):
    """Fit logistic regression p = sigmoid(X @ w + b) by IRLS.

    Deterministic: zero init, fixed iteration count, no randomness.
    l2 ridge stabilizes the fit on small/imbalanced calibration sets.
    Returns dict with weights, intercept, n_iter, converged.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2 or len(y) != len(X) or len(y) == 0:
        raise ValueError("X must be (n, k) and y length-n, non-empty")
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("y must be binary 0/1")
    n, k = X.shape
    Xa = np.column_stack([X, np.ones(n)])  # last col = intercept
    wb = np.zeros(k + 1)
    converged = False
    for it in range(max_iter):
        p = _sigmoid(Xa @ wb)
        w_ = p * (1.0 - p)
        w_ = np.clip(w_, 1e-10, None)
        # IRLS update with ridge (not on intercept)
        XtW = Xa.T * w_
        H = XtW @ Xa
        H[np.diag_indices(k + 1)] += l2 * np.r_[np.ones(k), 0.0]
        g = Xa.T @ (p - y)
        g[:k] += l2 * wb[:k]
        step = np.linalg.solve(H, g)
        wb_new = wb - step
        if float(np.max(np.abs(wb_new - wb))) < tol:
            wb = wb_new
            converged = True
            break
        wb = wb_new
    return {"weights": [float(v) for v in wb[:k]],
            "intercept": float(wb[k]),
            "n_iter": it + 1, "converged": bool(converged),
            "l2": float(l2), "n_samples": int(n), "n_features": int(k)}


def predict_proba(X, weights, intercept):
    """P(ACCEPT | features) under the fitted logistic map."""
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    z = X @ w + float(intercept)
    return _sigmoid(z)


def brier_score(y_true, y_prob):
    """Mean squared error between predicted probabilities and 0/1 outcomes."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    if len(y_true) != len(y_prob) or len(y_true) == 0:
        raise ValueError("need non-empty, equal-length inputs")
    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_diagram(y_true, y_prob, n_bins=10):
    """Bin predicted probabilities; compare mean predicted vs observed rate.

    Returns list of dicts: bin edges, n, mean_predicted, observed_rate,
    abs_diff. Bins are equal-width on [0, 1]; empty bins report n=0 and
    None rates (honest, not interpolated).
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            sel = (y_prob >= lo) & (y_prob <= hi)
        else:
            sel = (y_prob >= lo) & (y_prob < hi)
        n = int(sel.sum())
        if n == 0:
            bins.append({"bin_lo": lo, "bin_hi": hi, "n": 0,
                         "mean_predicted": None, "observed_rate": None,
                         "abs_diff": None})
        else:
            mp = float(y_prob[sel].mean())
            obs = float(y_true[sel].mean())
            bins.append({"bin_lo": lo, "bin_hi": hi, "n": n,
                         "mean_predicted": mp, "observed_rate": obs,
                         "abs_diff": float(abs(mp - obs))})
    return bins


def is_calibrated(bins, tol=0.15, min_n=5):
    """Calibrated iff every bin with >= min_n samples has |pred-obs| < tol.

    Bins with too few samples are reported, not counted — small bins cannot
    confirm or refute calibration.
    """
    checked = [b for b in bins if b["n"] >= min_n]
    if not checked:
        return False, "no bin has enough samples to judge"
    worst = max(checked, key=lambda b: b["abs_diff"])
    ok = all(b["abs_diff"] < tol for b in checked)
    detail = (f"{len(checked)} bins checked; worst |pred-obs| = "
              f"{worst['abs_diff']:.3f} in bin [{worst['bin_lo']:.1f}, "
              f"{worst['bin_hi']:.1f}] (n={worst['n']})")
    return bool(ok), detail


def format_upper_bound(rmse_px, marking_noise_px=1.0):
    """Upper-bound error framing: 'true error <= X px (includes marking noise)'.

    marking_noise_px is an assumed human-marking noise floor (default 1.0 px),
    documented here rather than hidden. Returns None when rmse is unmeasured.
    """
    try:
        rmse = float(rmse_px)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(rmse) or rmse < 0:
        return None
    return (f"true error <= {rmse + marking_noise_px:.2f} px "
            f"(includes {marking_noise_px:g} px marking noise)")
