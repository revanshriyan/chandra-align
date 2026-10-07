"""Opt-in deformation-field refinement stage: affine -> field -> re-gate.

Provenance
----------
The thin-plate-spline residual-field formulation below is ported verbatim
from scripts/deformation_field_experiment.py (2026-10-08), which broke the
1.6 px floor on the real OHRC pair: a regularized TPS field fit on
affine-residuals predicted a held-out quadrant at 1.02 px RMSE where the
global partial-affine managed 19.33 px, and cut spatial held-out error from
6.47/5.50 px to 3.88/1.99 px on both splits. The lambda grid, the /1000
coordinate normalization, and the lstsq-based solver below carry the
validated semantics of that experiment; do not "simplify" them without
re-running that validation.

Pipeline contract
-----------------
* ADDITIVE and OPT-IN. The caller (app._align_core) invokes
  apply_deform_field_stage only when CHANDRA_DEFORM_FIELD=1. With the flag
  unset the stage is never called and the pipeline is bit-identical.
* The stage NEVER changes gate thresholds. It returns field-corrected
  residuals; the caller re-runs the SAME frozen validate_registration_gate
  on them (ACCEPT: RMSE<=0.50 px, >=8 inliers, entropy>=0.75, >=3 quads;
  COARSE: RMSE<=2.50 px, >=8 inliers, entropy>=0.50, >=2 quads).
* FAIL-CLOSED. Degenerate input, fit failure, folding detection, or "no
  lambda beats affine on internal held-out" returns {"applied": False, ...}
  and the caller keeps the affine verdict. Never raises into the pipeline
  for expected degenerate inputs (the caller additionally guards the call).
"""

import os

import cv2
import numpy as np

from chandra_align.trust import split_fit_holdout

# Environment flag enabling the stage. Default OFF.
_DEFORM_FIELD_ENV = "CHANDRA_DEFORM_FIELD"
_FLAG_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

# Validated in the deformation-field experiment (docs/deformation-field-2026-10-08.md).
TPS_SCALE = 1000.0
LAMBDA_GRID = (0.1, 1.0, 10.0, 100.0)
MIN_FIELD_POINTS = 6          # below this the field is not attempted
MIN_JACOBIAN_DET = 0.5        # folding fuse: min det(J) must stay above this
MAX_DISPLACEMENT_PX = 500.0   # numerical-blowup fuse (validated fields: <=35 px)


def deform_field_enabled():
    """True iff the operator opted in via CHANDRA_DEFORM_FIELD=1."""
    return os.environ.get(_DEFORM_FIELD_ENV, "").strip().lower() in _FLAG_TRUE_VALUES


# ---------------------------------------------------------------------------
# Thin-plate spline on residuals (ported verbatim from
# scripts/deformation_field_experiment.py).
# ---------------------------------------------------------------------------

def _tps_kernel(d2):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d2 > 0, d2 * np.log(d2), 0.0)


def _tps_fit(px, py, v, lmbda):
    """Thin-plate spline f(px,py)=v. Coords must be pre-normalized."""
    n = len(px)
    d2 = (px[:, None] - px[None, :]) ** 2 + (py[:, None] - py[None, :]) ** 2
    K = _tps_kernel(d2)
    A = np.zeros((n + 3, n + 3))
    A[:n, :n] = K + lmbda * np.eye(n)
    A[:n, n] = 1.0
    A[:n, n + 1] = px
    A[:n, n + 2] = py
    A[n, :n] = 1.0
    A[n + 1, :n] = px
    A[n + 2, :n] = py
    b = np.zeros(n + 3)
    b[:n] = v
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return sol[:n], sol[n:]  # w, a


def _tps_eval(w, a, px, py, qx, qy):
    d2 = (qx[:, None] - px[None, :]) ** 2 + (qy[:, None] - py[None, :]) ** 2
    return _tps_kernel(d2) @ w + a[0] + a[1] * qx + a[2] * qy


def _fit_residual_field(p_src, residuals, lmbda):
    """Fit TPS displacement field p_src -> residuals. Returns field tuple."""
    px, py = p_src[:, 0] / TPS_SCALE, p_src[:, 1] / TPS_SCALE
    wx, ax = _tps_fit(px, py, residuals[:, 0], lmbda)
    wy, ay = _tps_fit(px, py, residuals[:, 1], lmbda)
    return (wx, ax, wy, ay, px, py)


def _eval_residual_field(field, p_src):
    """Evaluate displacement field at query points. Returns (M, 2)."""
    wx, ax, wy, ay, px, py = field
    qx, qy = p_src[:, 0] / TPS_SCALE, p_src[:, 1] / TPS_SCALE
    d = np.stack([_tps_eval(wx, ax, px, py, qx, qy),
                  _tps_eval(wy, ay, px, py, qx, qy)], axis=1)
    return d


def _apply_affine(M, p):
    """Apply 2x3 partial-affine matrix M to (N,2) points."""
    p = np.asarray(p, dtype=np.float64)
    return p @ np.asarray(M, dtype=np.float64)[:, :2].T + np.asarray(M, dtype=np.float64)[:, 2]


def _jacobian_min_det(M, field, p_src, h=1.0, max_points=200):
    """Minimum Jacobian determinant of the full map p -> M@p + d(p).

    J(p) = A + grad d(p), with A the 2x2 linear part of M and grad d via
    central differences. A folding (or near-folding) field has det <= 0;
    validated smooth fields stay >= 0.93.
    """
    p_src = np.asarray(p_src, dtype=np.float64)
    n = p_src.shape[0]
    if n == 0:
        return float("nan")
    idx = np.arange(n)
    if n > max_points:
        idx = idx[:max_points]  # deterministic subsample: first points
    q = p_src[idx]
    A = np.asarray(M, dtype=np.float64)[:, :2]
    ex = np.array([h, 0.0])
    ey = np.array([0.0, h])
    ddx = (_eval_residual_field(field, q + ex) - _eval_residual_field(field, q - ex)) / (2.0 * h)
    ddy = (_eval_residual_field(field, q + ey) - _eval_residual_field(field, q - ey)) / (2.0 * h)
    # J_i = A + [[ddx_x, ddy_x], [ddx_y, ddy_y]]
    j11 = A[0, 0] + ddx[:, 0]
    j12 = A[0, 1] + ddy[:, 0]
    j21 = A[1, 0] + ddx[:, 1]
    j22 = A[1, 1] + ddy[:, 1]
    dets = j11 * j22 - j12 * j21
    return float(np.min(dets)) if dets.size else float("nan")


# ---------------------------------------------------------------------------
# The stage.
# ---------------------------------------------------------------------------

def _as_points(a):
    try:
        arr = np.asarray(a, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def apply_deform_field_stage(p_src, p_ref, M, lambda_grid=LAMBDA_GRID, seed=7):
    """Fit a regularized TPS residual field on affine inliers (opt-in stage).

    Parameters
    ----------
    p_src, p_ref : (N, 2) array-like
        Inlier correspondences (secondary -> reference frame).
    M : (2, 3) array-like
        The RANSAC partial-affine matrix mapping p_src -> p_ref.
    lambda_grid : iterable of float
        Smoothing strengths swept; the winner is chosen by INTERNAL
        held-out, never by inlier fit. The held-out uses the pipeline's own
        stride-based 80/20 split (chandra_align.trust.split_fit_holdout) --
        the same generalization methodology the gate applies to the affine
        model -- so the comparison changes only the model, never the
        measurement. Note its limit: the stride split is interleaved
        (interpolation-like); the harder spatial-band extrapolation score
        from docs/deformation-field-2026-10-08.md (3.88/1.99 px) is the
        stricter benchmark and is reported there, not here.
    seed : int
        Accepted for interface stability; the stride split itself is
        deterministic, so runs are reproducible regardless.

    Returns
    -------
    dict with at least ``applied`` (bool) and ``reason`` (str|None).
    When applied, also carries ``residuals_vec`` (N,2), ``residuals_mag``
    (N,), ``lambda_chosen``, ``rmse_before_px`` / ``rmse_after_px``
    (in-sample), ``heldout_rmse_affine_px`` / ``heldout_rmse_px``
    (internal stride 80/20 check), ``max_displacement_px``, ``min_jacobian_det``,
    ``lambda_sweep`` ([[lambda, check_rmse], ...]) and ``n_points``.
    All other outcomes are fail-closed: {"applied": False, "reason": ...}.
    """
    out = {"applied": False, "reason": None, "n_points": 0}

    ps = _as_points(p_src)
    pr = _as_points(p_ref)
    if ps is None or pr is None or ps.shape[0] != pr.shape[0]:
        out["reason"] = "bad_points"
        return out
    n = ps.shape[0]
    out["n_points"] = int(n)
    if n < MIN_FIELD_POINTS:
        out["reason"] = "insufficient_points"
        return out
    try:
        Ma = np.asarray(M, dtype=np.float64)
    except (TypeError, ValueError):
        out["reason"] = "bad_matrix"
        return out
    if Ma.shape != (2, 3) or not np.all(np.isfinite(Ma)):
        out["reason"] = "bad_matrix"
        return out

    grid = tuple(float(v) for v in lambda_grid) if lambda_grid else ()
    if not grid or any(not np.isfinite(v) or v < 0 for v in grid):
        out["reason"] = "bad_lambda_grid"
        return out

    # Affine residuals: the field models what the global transform misses.
    r = pr - _apply_affine(Ma, ps)
    if not np.all(np.isfinite(r)):
        out["reason"] = "nonfinite_residuals"
        return out
    rmse_before = float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))

    # Internal held-out for lambda selection: the pipeline's own stride-based
    # 80/20 split (deterministic, interleaved). Same split the gate uses for
    # the affine model, so lambda selection compares models, not methodologies.
    fit_a, fit_b, hold_a, hold_b = split_fit_holdout(ps, pr, holdout_fraction=0.2, seed=seed)
    if len(hold_a) == 0 or len(fit_a) < MIN_FIELD_POINTS:
        out["reason"] = "insufficient_points_for_split"
        return out
    r_fit = fit_b - _apply_affine(Ma, fit_a)
    r_hold = hold_b - _apply_affine(Ma, hold_a)
    if not (np.all(np.isfinite(r_fit)) and np.all(np.isfinite(r_hold))):
        out["reason"] = "nonfinite_residuals"
        return out

    check_rmse_affine = float(np.sqrt(np.mean(np.sum(r_hold ** 2, axis=1))))
    sweep = []
    best = None  # (check_rmse, lmbda)
    for lmbda in grid:
        try:
            field = _fit_residual_field(fit_a, r_fit, lmbda)
            d_hold = _eval_residual_field(field, hold_a)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
        if not np.all(np.isfinite(d_hold)):
            continue
        check_rmse = float(np.sqrt(np.mean(np.sum((r_hold - d_hold) ** 2, axis=1))))
        sweep.append([lmbda, check_rmse])
        if best is None or check_rmse < best[0]:
            best = (check_rmse, lmbda)
    out["lambda_sweep"] = sweep
    out["heldout_rmse_affine_px"] = check_rmse_affine
    out["heldout_n_check"] = int(len(hold_a))

    # Honest behavior: no lambda beating affine on held-out -> keep affine.
    # The win must be MEANINGFUL, not float dust: on near-perfect data the
    # stiffest lambda can "beat" affine by <1% by fitting noise. The fuse
    # requires >=5% relative and >=0.02 px absolute improvement.
    out["improvement_required_px"] = None
    if best is None:
        out["reason"] = "no_improvement"
        return out
    improvement = check_rmse_affine - best[0]
    required = max(0.05 * check_rmse_affine, 0.02)
    out["improvement_px"] = float(improvement)
    out["improvement_required_px"] = float(required)
    if not (improvement >= required):
        out["reason"] = "no_improvement"
        return out
    lmbda_star = best[1]
    out["lambda_chosen"] = float(lmbda_star)
    out["heldout_rmse_px"] = float(best[0])

    # Refit the winning lambda on ALL inliers.
    try:
        field = _fit_residual_field(ps, r, lmbda_star)
        d_all = _eval_residual_field(field, ps)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        out["reason"] = "refit_failed"
        return out
    if not np.all(np.isfinite(d_all)):
        out["reason"] = "nonfinite_field"
        return out

    max_disp = float(np.max(np.linalg.norm(d_all, axis=1)))
    out["max_displacement_px"] = max_disp
    if max_disp > MAX_DISPLACEMENT_PX:
        out["reason"] = "displacement_blowup"
        return out

    min_det = _jacobian_min_det(Ma, field, ps)
    out["min_jacobian_det"] = min_det
    if not np.isfinite(min_det) or min_det < MIN_JACOBIAN_DET:
        out["reason"] = "folding_detected"
        return out

    r_corr = r - d_all
    rmse_after = float(np.sqrt(np.mean(np.sum(r_corr ** 2, axis=1))))

    out.update({
        "applied": True,
        "reason": None,
        "rmse_before_px": rmse_before,
        "rmse_after_px": rmse_after,
        "residuals_vec": r_corr,
        "residuals_mag": np.linalg.norm(r_corr, axis=1),
        # The fitted forward map, for the opt-in dense-remap export:
        # reference coords = M_hook @ p_sec + d(p_sec). Carried so the
        # exported raster can reproduce the scored geometry instead of the
        # affine-only warp. Telemetry sanitizers must exclude these keys.
        "field": field,          # (wx, ax, wy, ay, px, py) TPS tuple
        "M_hook": Ma.copy(),     # (2, 3) partial-affine the field corrects
    })
    return out


# ---------------------------------------------------------------------------
# Dense-remap export (opt-in): evaluate the fitted forward map on the grid.
# ---------------------------------------------------------------------------

def _check_export_field(field):
    """Validate the fitted field tuple; return normalized components."""
    try:
        wx, ax, wy, ay, px, py = field
        wx = np.asarray(wx, dtype=np.float64).reshape(-1)
        wy = np.asarray(wy, dtype=np.float64).reshape(-1)
        ax = np.asarray(ax, dtype=np.float64).reshape(3)
        ay = np.asarray(ay, dtype=np.float64).reshape(3)
        px = np.asarray(px, dtype=np.float64).reshape(-1)
        py = np.asarray(py, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        raise ValueError("malformed field tuple")
    n = px.shape[0]
    if n == 0 or not (wx.shape[0] == n and wy.shape[0] == n and py.shape[0] == n):
        raise ValueError("field control points inconsistent")
    for a in (wx, ax, wy, ay, px, py):
        if not np.all(np.isfinite(a)):
            raise ValueError("non-finite field coefficients")
    return (wx, ax, wy, ay, px, py)


def _tps_eval_both(w_x, a_x, w_y, a_y, px, py, qx, qy, dtype=np.float64):
    """TPS evaluation of both displacement components with a SHARED kernel.

    Same math as _tps_eval (which builds the (T, n) kernel twice, once per
    component); the kernel depends only on geometry, so one build serves
    both. Exact when dtype=float64; the export path uses float32, whose
    rounding is bounded end-to-end by unit test (< 0.05 px map diff).
    """
    dt = np.float32 if dtype is np.float32 else np.float64
    px = np.asarray(px, dtype=dt).reshape(-1)
    py = np.asarray(py, dtype=dt).reshape(-1)
    qx = np.asarray(qx, dtype=dt).reshape(-1)
    qy = np.asarray(qy, dtype=dt).reshape(-1)
    w_x = np.asarray(w_x, dtype=dt).reshape(-1)
    w_y = np.asarray(w_y, dtype=dt).reshape(-1)
    a_x = np.asarray(a_x, dtype=dt).reshape(3)
    a_y = np.asarray(a_y, dtype=dt).reshape(3)
    d2 = (qx[:, None] - px[None, :]) ** 2 + (qy[:, None] - py[None, :]) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(d2 > 0, d2 * np.log(d2), dt(0.0))
    dx = K @ w_x + a_x[0] + a_x[1] * qx + a_x[2] * qy
    dy = K @ w_y + a_y[0] + a_y[1] * qx + a_y[2] * qy
    return np.stack([dx, dy], axis=1)


def _eval_field_tiled(field, q, tile_points=16384):
    """Evaluate the residual field at query points, tiled for memory.

    _tps_eval materializes a (T, n_ctrl) float64 kernel; tiling keeps the
    transient under ~256 MB even for thousands of control points.
    Deterministic: tiles are processed in row order, no randomness.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1, 2)
    if q.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    parts = []
    for s in range(0, q.shape[0], tile_points):
        d = _eval_residual_field(field, q[s:s + tile_points])
        if not np.all(np.isfinite(d)):
            raise ValueError("non-finite field evaluation")
        parts.append(d)
    return np.vstack(parts)


def _eval_field_tiled_fast(field, q, tile_points=65536):
    """Export-path field evaluation: shared float32 kernel, larger tiles.

    Same TPS math as _eval_residual_field; float32 rounding is bounded
    end-to-end (unit test: < 0.05 px map diff vs the exact path).
    """
    wx, ax, wy, ay, px, py = field
    q = np.asarray(q, dtype=np.float64).reshape(-1, 2)
    if q.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    parts = []
    for s in range(0, q.shape[0], tile_points):
        qq = q[s:s + tile_points] / TPS_SCALE  # _tps_eval_both takes normalized coords
        d = _tps_eval_both(wx, ax, wy, ay, px, py, qq[:, 0], qq[:, 1],
                           dtype=np.float32)
        if not np.all(np.isfinite(d)):
            raise ValueError("non-finite field evaluation")
        parts.append(np.asarray(d, dtype=np.float64))
    return np.vstack(parts)


def build_field_remap_maps(sec_shape, ref_shape, M_hook, field,
                           n_iter=5, coarse_factor=2, _exact=False):
    """Build cv2.remap sampling maps reproducing the scored field geometry.

    The stage scores the forward map F(p) = M_hook @ p + d(p) (secondary ->
    reference coordinates). The exported raster must place secondary content
    p at reference position F(p); i.e. for each output pixel q we need the
    secondary coordinate p with F(p) = q. That inverse is obtained by
    fixed-point iteration on the affine sampling map S(q) = A^-1 (q - t):

        p_{k+1} = S(q - d(p_k)),   p_0 = S(q),

    which converges geometrically because the validated field is smooth and
    fold-free (min Jacobian det >= 0.5, so ||A^-1 grad d|| << 1). The
    displacement field d(.) is evaluated ONCE on the secondary grid (tiled);
    iterations only bilinearly resample it, so this is cheap and exact to
    ~1e-4 px after a few iterations -- no new fitting, no new validation.

    Performance: d(.) is smooth (TPS, regularized), so it is evaluated on a
    coarse_factor-decimated grid and bilinearly upsampled (cv2.resize), and
    the kernel is shared between components in float32. The total
    approximation error vs the exact path (_exact=True: float64, full grid)
    is bounded by unit test (< 0.05 px map diff).

    Returns (map_x, map_y) float32 arrays of shape ref_shape for
    cv2.remap(sec, map_x, map_y, INTER_LINEAR). Raises ValueError on any
    degenerate input; the caller must fail closed to the affine warp.
    """
    try:
        hs, ws = int(sec_shape[0]), int(sec_shape[1])
        hr, wr = int(ref_shape[0]), int(ref_shape[1])
    except (TypeError, ValueError, IndexError):
        raise ValueError("bad image shapes")
    if min(hs, ws, hr, wr) <= 0:
        raise ValueError("non-positive image shape")
    Ma = np.asarray(M_hook, dtype=np.float64)
    if Ma.shape != (2, 3) or not np.all(np.isfinite(Ma)):
        raise ValueError("bad M_hook")
    A = Ma[:, :2]
    t = Ma[:, 2]
    det = float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
    if not np.isfinite(det) or abs(det) < 1e-9:
        raise ValueError("singular M_hook linear part")
    A_inv = np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]]) / det
    field = _check_export_field(field)

    # Forward displacement on the secondary grid, tiled. The field is
    # smooth (regularized TPS), so it is evaluated on a decimated grid and
    # bilinearly upsampled; the error is bounded by unit test (< 0.05 px
    # end-to-end vs _exact=True). _exact=True forces the slow exact path
    # (float64, full grid) for validation only.
    cf = 1 if _exact else max(int(coarse_factor), 1)
    ys, xs = np.mgrid[0:hs:cf, 0:ws:cf].astype(np.float64)
    p_sec = np.stack([xs.ravel(), ys.ravel()], axis=1)
    if _exact:
        d_coarse = _eval_field_tiled(field, p_sec)
    else:
        d_coarse = _eval_field_tiled_fast(field, p_sec)
    hc, wc = ys.shape
    d_coarse = d_coarse.reshape(hc, wc, 2).astype(np.float32)
    if cf == 1:
        d_sec_f32 = d_coarse
    else:
        d_sec_f32 = cv2.resize(d_coarse, (ws, hs),
                               interpolation=cv2.INTER_LINEAR)
    if not np.all(np.isfinite(d_sec_f32)):
        raise ValueError("non-finite displacement grid")

    # Output pixel coordinates (reference frame), kept as 2D grids so
    # cv2.remap can resample the displacement field directly.
    yr, xr = np.mgrid[0:hr, 0:wr].astype(np.float64)
    q = np.stack([xr, yr], axis=-1)  # (hr, wr, 2)

    def _affine_sample(qq):
        return (qq - t) @ A_inv.T

    p = _affine_sample(q)
    for _ in range(max(int(n_iter), 0)):
        dk = cv2.remap(d_sec_f32,
                       p[..., 0].astype(np.float32), p[..., 1].astype(np.float32),
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
        p = _affine_sample(q - dk.astype(np.float64))
        if not np.all(np.isfinite(p)):
            raise ValueError("non-finite fixed-point iterate")
    map_x = p[..., 0].astype(np.float32)
    map_y = p[..., 1].astype(np.float32)
    if not (np.all(np.isfinite(map_x)) and np.all(np.isfinite(map_y))):
        raise ValueError("non-finite remap maps")
    return map_x, map_y
