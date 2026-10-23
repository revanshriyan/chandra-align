"""Arm-vs-fallback point-set arbitration (opt-in: CHANDRA_ARBITRATE=1).

When an opt-in matcher arm fires, the pipeline historically used its
points unconditionally. This selector scores the arm's point set against
the default path's point set with one shared evidence protocol and picks
a winner; the winner then flows through the unchanged downstream (fit,
deform stage, frozen gate). Arbitration chooses input points, never
verdicts.

Protocol (docs/arm-arbitration-prereg-2026-10-23.md): for each candidate
set, over 3 fixed seeds, split the correspondences 80/20, fit an affine
with cv2.estimateAffinePartial2D (RANSAC, 3.0 px) on the fit split, and
measure RMSE on the check split. Score is the median check RMSE. A set
with fewer than MIN_POINTS points, or one that cannot be fit on any
seed, scores +inf and cannot win unless both sets are +inf (then the
arm keeps its points, matching the historical behavior).
"""

from __future__ import annotations

import cv2
import numpy as np

SEEDS = (7, 77, 777)
MIN_POINTS = 8
TIE_TOLERANCE_PX = 0.05


def _score_point_set(pts_ref, pts_sec, seeds=SEEDS):
    """Median held-out RMSE (px) of a correspondence set; +inf if unusable."""
    pts_ref = np.asarray(pts_ref, dtype=np.float64).reshape(-1, 2)
    pts_sec = np.asarray(pts_sec, dtype=np.float64).reshape(-1, 2)
    n = len(pts_ref)
    if n < MIN_POINTS or len(pts_sec) != n:
        return float("inf")
    scores = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_fit = max(3, int(round(0.8 * n)))
        fit_idx, check_idx = idx[:n_fit], idx[n_fit:]
        if len(check_idx) < 2:
            continue
        cv2.setRNGSeed(int(seed))
        matrix, _ = cv2.estimateAffinePartial2D(
            pts_sec[fit_idx].astype(np.float32),
            pts_ref[fit_idx].astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if matrix is None:
            continue
        predicted = pts_sec[check_idx] @ matrix[:, :2].T + matrix[:, 2]
        errors = np.linalg.norm(predicted - pts_ref[check_idx], axis=1)
        scores.append(float(np.sqrt(np.mean(errors**2))))
    if not scores:
        return float("inf")
    return float(np.median(scores))


def arbitrate_point_sets(arm_pts, fallback_pts):
    """Pick "arm" or "fallback" for two (pts_ref, pts_sec) candidates.

    Returns (winner, info) where info records both scores and point
    counts for telemetry. Ties within TIE_TOLERANCE_PX go to the set
    with more correspondences; if both sets are unusable the arm wins
    (historical behavior, fail-closed).
    """
    arm_ref, arm_sec = arm_pts
    fb_ref, fb_sec = fallback_pts
    arm_score = _score_point_set(arm_ref, arm_sec)
    fb_score = _score_point_set(fb_ref, fb_sec)
    info = {
        "arm_score_px": arm_score,
        "fallback_score_px": fb_score,
        "arm_points": int(len(arm_ref)),
        "fallback_points": int(len(fb_ref)),
    }
    if np.isinf(arm_score) and np.isinf(fb_score):
        winner = "arm"
    elif fb_score < arm_score - TIE_TOLERANCE_PX:
        winner = "fallback"
    elif arm_score < fb_score - TIE_TOLERANCE_PX:
        winner = "arm"
    else:
        winner = "fallback" if len(fb_ref) > len(arm_ref) else "arm"
    info["winner"] = winner
    return winner, info
