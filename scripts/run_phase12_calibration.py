"""Phase 12: calibrate confidence -> P(SUCCESS_SUBPIXEL | features).

Builds a calibration set from:
  (a) synthetic pairs run through the REAL pipeline (SIFT -> Lowe ->
      verify_guarded -> quadrant metrics -> gate), spanning a difficulty
      gradient so outcomes cover ACCEPT and non-ACCEPT;
  (b) the complete-feature rows of results/table_canonical_v1.csv.

Fits logistic regression (numpy IRLS, deterministic) mapping the four
confidence features to P(ACCEPT). Writes results/phase12_calibration.json
with fitted parameters, reliability-diagram bins, Brier score, and an
honest calibrated / NOT CALIBRATED verdict.

The gates are untouched: this scores outcomes after the fact, never during.
"""
import csv
import json

import cv2
import numpy as np

from chandra_align.metrics import compute_quadrant_metrics, validate_registration_gate
from chandra_align.refine import verify_guarded
from chandra_align.testing import (make_pair_degenerate, make_pair_illumination,
                                   make_pair_scale, make_pair_shift)
from chandra_align.trust.calibration import (brier_score, confidence_features,
                                             fit_logistic_regression,
                                             is_calibrated, predict_proba,
                                             reliability_diagram)

MIN_INLIERS = 8


def _u8(img):
    a = np.asarray(img, dtype=np.float64)
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) * (255.0 / (hi - lo + 1e-9)), 0, 255).astype(np.uint8)


def run_pipeline_case(img_ref, img_mov):
    """Run SIFT->Lowe->guarded RANSAC->gate. Returns (features, label, info)."""
    a8, b8 = _u8(img_mov), _u8(img_ref)  # mov is "source", ref is "reference"
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.01, edgeThreshold=10)
    ka, da = sift.detectAndCompute(a8, None)
    kb, db = sift.detectAndCompute(b8, None)
    n_corr = 0
    if da is not None and db is not None and len(da) >= 2 and len(db) >= 2:
        knn = cv2.BFMatcher().knnMatch(da, db, k=2)
        good = [m for m, n in knn if m.distance < 0.75 * n.distance]
        n_corr = len(good)
        pa = np.float64([[ka[m.queryIdx].pt[0], ka[m.queryIdx].pt[1]] for m in good])
        pb = np.float64([[kb[m.trainIdx].pt[0], kb[m.trainIdx].pt[1]] for m in good])
    else:
        pa = np.zeros((0, 2)); pb = np.zeros((0, 2))
    cfg = {"ransac_reproj_threshold": 3.0, "max_iters": 5000, "confidence": 0.99}
    g = verify_guarded(pa, pb, cfg, image_shape=a8.shape)
    n_inl, rmse, entropy, verdict = 0, None, 0.0, "ABSTAIN"
    if g["ok"] and g["model"] is not None:
        M = np.asarray(g["model"], np.float64)
        ia, ib = g["inliers_a"], g["inliers_b"]
        proj = ia @ M[:, :2].T + M[:, 2]
        resid = np.hypot(proj[:, 0] - ib[:, 0], proj[:, 1] - ib[:, 1])
        rmse = float(resid.mean())
        qm_counts, qm_entropy = compute_quadrant_metrics(ia, resid, a8.shape)
        entropy = float(qm_entropy)
        _msg, verdict = validate_registration_gate(rmse, len(ia), MIN_INLIERS,
                                                   entropy, qm_counts, model=M)
        n_inl = int(len(ia))
    feats = confidence_features(n_inl, n_corr, entropy, rmse, MIN_INLIERS)
    label = 1 if verdict == "SUCCESS_SUBPIXEL" else 0
    return feats, label, {"verdict": verdict, "n_inliers": n_inl,
                         "n_corr": n_corr, "rmse": rmse, "entropy": entropy}


def synthetic_cases():
    rng = np.random.default_rng(20261007)
    cases = []
    # easy shifts: small translation, no rotation -> mostly ACCEPT
    for i in range(24):
        dx, dy = rng.uniform(-8, 8, 2)
        cases.append(("synth_easy_shift", dict(shape=(512, 512), dx=float(dx),
                                               dy=float(dy), angle_deg=0.0,
                                               seed=1000 + i),
                      make_pair_shift))
    # rotated shifts: 1-5 deg -> mixed
    for i in range(16):
        dx, dy = rng.uniform(-10, 10, 2)
        ang = float(rng.uniform(1.0, 5.0)) * (1 if i % 2 == 0 else -1)
        cases.append(("synth_rotated", dict(shape=(512, 512), dx=float(dx),
                                            dy=float(dy), angle_deg=ang,
                                            seed=2000 + i),
                      make_pair_shift))
    # illumination: gamma/gain stress -> mixed to fail
    for i in range(20):
        gamma = float(rng.uniform(1.3, 2.2))
        gain = float(rng.uniform(0.8, 1.4))
        cases.append(("synth_illum", dict(shape=(512, 512), dx=5.0, dy=2.0,
                                          gamma=gamma, gain=gain, seed=3000 + i),
                      make_pair_illumination))
    # scale: 1.5x-4x -> mostly fail
    for i in range(12):
        ratio = float(rng.uniform(1.5, 4.0))
        cases.append(("synth_scale", dict(shape=(512, 512), ratio=ratio,
                                          dx=3.0, dy=3.0, seed=4000 + i),
                      make_pair_scale))
    # degenerate: featureless -> must fail
    for i in range(6):
        cases.append(("synth_degenerate", dict(shape=(512, 512), seed=5000 + i),
                      make_pair_degenerate))
    return cases


def canonical_rows():
    out = []
    with open("results/table_canonical_v1.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                corr = int(float(r["correspondences"])); inl = int(float(r["inliers"]))
                rmse = float(r["rmse_px"]); ent = float(r["entropy"])
            except (ValueError, TypeError):
                continue
            v = (r["verdict"] or "").upper()
            label = 1 if ("ACCEPTED" in v or "SUCCESS_SUBPIXEL" in v) else 0
            feats = confidence_features(inl, corr, ent, rmse, MIN_INLIERS)
            out.append((f"canonical:{r['benchmark']}/{r['pair']}/{r['matcher']}",
                        feats, label,
                        {"verdict": r["verdict"], "n_inliers": inl,
                         "n_corr": corr, "rmse": rmse, "entropy": ent}))
    return out


def main():
    samples = []
    for name, kwargs, maker in synthetic_cases():
        ref, mov, _M = maker(**kwargs)
        feats, label, info = run_pipeline_case(ref, mov)
        samples.append({"source": name, "features": list(feats),
                        "label": label, **info})
    for source, feats, label, info in canonical_rows():
        samples.append({"source": source, "features": list(feats),
                        "label": label, **info})

    X = np.array([s["features"] for s in samples])
    y = np.array([s["label"] for s in samples])
    n_pos = int(y.sum())
    print(f"calibration set: n={len(y)} accept={n_pos} non-accept={len(y) - n_pos}")

    fit = fit_logistic_regression(X, y)
    proba = predict_proba(X, fit["weights"], fit["intercept"])
    for s, p in zip(samples, proba):
        s["p_accept"] = float(p)

    bins = reliability_diagram(y, proba, n_bins=10)
    brier = brier_score(y, proba)
    cal_ok, cal_detail = is_calibrated(bins)

    # reference: uncalibrated heuristic score / 100 as a "probability"
    heur = np.array([0.30 * f[0] + 0.25 * f[1] + 0.20 * f[2] + 0.25 * f[3]
                     for f in X])
    brier_heur = brier_score(y, heur)

    result = {
        "n_samples": len(y), "n_accept": n_pos,
        "feature_names": ["support", "inlier_ratio", "spread", "residual_quality"],
        "fit": fit,
        "brier_score": brier,
        "brier_heuristic_baseline": brier_heur,
        "reliability_bins": bins,
        "calibrated": bool(cal_ok),
        "calibration_detail": cal_detail,
        "samples": samples,
        "notes": ("Logistic map p = sigmoid(w . features + b), numpy IRLS, "
                  "deterministic. Gates untouched: scoring only, post-hoc. "
                  "Brier heuristic baseline = raw 30/25/20/25 score / 100."),
    }
    with open("results/phase12_calibration.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print("wrote results/phase12_calibration.json")
    print(f"Brier: fitted={brier:.4f} heuristic-baseline={brier_heur:.4f}")
    print("weights:", [f"{w:+.3f}" for w in fit["weights"]],
          f"b={fit['intercept']:+.3f}")
    print("per-bin |pred-obs|:",
          [f"{b['abs_diff']:.2f}" if b["abs_diff"] is not None else "empty"
           for b in bins])
    print("CALIBRATED" if cal_ok else "NOT CALIBRATED", "-", cal_detail)


if __name__ == "__main__":
    main()
