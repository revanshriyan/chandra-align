"""Phase 8 — Gate 3 transform conditioning + pre-fit ABSTAIN codes.

Gate 3 (numerically verified 2026-10-05) rejects degenerate transforms by
math, not by threshold-tuning:

- cond(A) < 1e7            (SVD condition number of the linear part)
- |det(A)| > 1e-4          (no area collapse)
- scale_ratio < 20         (max/min singular value; no extreme anisotropy)
- projectivity < 0.05      (affine stays affine; homography stays near-affine)
- rmse_px <= 5             (gross misfit backstop)

Applied inside RANSAC acceptance AND as a final backstop, fail closed.

Pre-fit ABSTAIN codes are structurally distinct from post-fit REJECT:
a fit that never should have been attempted is not the same as a fit
that failed. Codes: ZERO_CANDIDATES, INSUFFICIENT_UNIQUE, COLLINEAR,
ILL_CONDITIONED, NO_VALID_MODEL.
"""

import numpy as np

# Gate 3 thresholds (frozen; verified 2026-10-05 — do not tune for optics).
GATE3_COND_MAX = 1e7
GATE3_DET_MIN = 1e-4
GATE3_SCALE_RATIO_MAX = 20.0
GATE3_PROJECTIVITY_MAX = 0.05
GATE3_RMSE_MAX_PX = 5.0

# Pre-fit ABSTAIN thresholds.
ABSTAIN_UNIQUE_MIN = 6
ABSTAIN_DESIGN_COND_MAX = 1e6


def _linear_part(M):
    M = np.asarray(M, dtype=np.float64)
    if M.shape == (2, 3):
        return M[:, :2], 0.0
    if M.shape == (3, 3):
        # projectivity: distance of the last row from [0, 0, 1]
        proj = float(np.hypot(M[2, 0], M[2, 1]) + abs(M[2, 2] - 1.0))
        return M[:2, :2] / M[2, 2] if abs(M[2, 2]) > 1e-12 else M[:2, :2], proj
    raise ValueError(f"expected 2x3 or 3x3 transform, got {M.shape}")


def check_transform_conditioning(M, rmse_px=None):
    """Apply Gate 3 to a fitted transform. Returns (ok, report).

    ok is True only when every check passes. report carries the measured
    values and per-check pass/fail so rejections stay explainable.
    Non-finite inputs fail closed.
    """
    report = {"ok": False, "checks": {}}
    try:
        A, projectivity = _linear_part(M)
    except (ValueError, TypeError):
        report["error"] = "unparseable transform"
        return False, report
    if not np.all(np.isfinite(A)):
        report["error"] = "non-finite linear part"
        return False, report
    try:
        sv = np.linalg.svd(A, compute_uv=False)
    except np.linalg.LinAlgError:
        report["error"] = "svd failed"
        return False, report
    s_max, s_min = float(sv[0]), float(sv[-1])
    cond = s_max / s_min if s_min > 0 else float("inf")
    det = abs(float(np.linalg.det(A)))
    scale_ratio = s_max / s_min if s_min > 0 else float("inf")
    checks = {
        "cond_lt_1e7": bool(cond < GATE3_COND_MAX),
        "det_gt_1e-4": bool(det > GATE3_DET_MIN),
        "scale_ratio_lt_20": bool(scale_ratio < GATE3_SCALE_RATIO_MAX),
        "projectivity_lt_0.05": bool(projectivity < GATE3_PROJECTIVITY_MAX),
    }
    if rmse_px is not None:
        try:
            rmse_v = float(rmse_px)
        except (TypeError, ValueError):
            rmse_v = float("inf")
        checks["rmse_le_5px"] = bool(np.isfinite(rmse_v) and rmse_v <= GATE3_RMSE_MAX_PX)
    report.update({
        "cond": float(cond),
        "det": float(det),
        "scale_ratio": float(scale_ratio),
        "projectivity": float(projectivity),
        "checks": checks,
    })
    ok = all(checks.values())
    report["ok"] = ok
    return ok, report


def dedup_correspondences(pts_a, pts_b, radius=2.0):
    """Merge correspondences closer than `radius` px (unique-inlier dedup).

    Greedy: iterate in input order, keep a pair only when its pts_a is
    farther than `radius` from every kept pts_a. Returns
    (kept_a, kept_b, kept_idx, n_raw, n_unique).
    """
    pa = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
    pb = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)
    n = len(pa)
    if n == 0:
        z = np.zeros((0, 2), np.float64)
        return z, z, np.zeros(0, dtype=int), 0, 0
    kept = []
    for i in range(n):
        if all(np.hypot(*(pa[i] - pa[j])) >= radius for j in kept):
            kept.append(i)
    kept = np.asarray(kept, dtype=int)
    return pa[kept], pb[kept], kept, n, len(kept)


def classify_prefit_abstain(pts_a, pts_b):
    """Pre-fit structural check. Returns an ABSTAIN code or None.

    Codes (distinct from post-fit REJECT):
    - ZERO_CANDIDATES: no correspondences at all.
    - INSUFFICIENT_UNIQUE: fewer than 6 unique points after 1 px dedup.
    - COLLINEAR: points are (near-)collinear — no 2D constraint.
    - ILL_CONDITIONED: affine design-matrix SVD cond > 1e6.
    None means "a fit may be attempted" (it can still fail -> NO_VALID_MODEL).
    """
    pa = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
    pb = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)
    if len(pa) == 0 or len(pb) == 0 or len(pa) != len(pb):
        return "ZERO_CANDIDATES"
    _, _, _, n_raw, n_unique = dedup_correspondences(pa, pb, radius=1.0)
    if n_unique < ABSTAIN_UNIQUE_MIN:
        return "INSUFFICIENT_UNIQUE"
    # Collinearity: centred points must span 2D.
    c = pa - pa.mean(axis=0)
    sv = np.linalg.svd(c, compute_uv=False)
    if sv[0] <= 0 or (sv[1] / sv[0]) < 1e-6:
        return "COLLINEAR"
    # Design-matrix conditioning for the affine fit [x y 1].
    X = np.column_stack([pa, np.ones(len(pa))])
    try:
        dsv = np.linalg.svd(X, compute_uv=False)
    except np.linalg.LinAlgError:
        return "ILL_CONDITIONED"
    cond = float(dsv[0] / dsv[-1]) if dsv[-1] > 0 else float("inf")
    if not np.isfinite(cond) or cond > ABSTAIN_DESIGN_COND_MAX:
        return "ILL_CONDITIONED"
    return None
