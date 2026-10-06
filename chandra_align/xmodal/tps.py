"""Smoothing thin-plate-spline warp (Phase 9 rescue branch), numpy-only.

Fits a 2-D thin-plate spline mapping source points `pa` to target points
`pb`. With lmbda=0 the spline exactly interpolates the control points;
with lmbda>0 a ridge penalty on the non-affine weights trades exactness
for smoothness (bending energy decreases as lmbda grows).

What it does NOT do: it does not validate the correspondences it is
given — garbage in, smooth garbage out. Callers must gate the input
points (RANSAC inliers, Gate 3 conditioning) before fitting.
"""

import numpy as np


def _tps_kernel(r):
    """U(r) = r^2 * log(r^2), with U(0) = 0 (continuous limit)."""
    r = np.asarray(r, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(r > 0, (r ** 2) * np.log(r ** 2), 0.0)
    return out


def fit_tps(pa, pb, lmbda=0.05):
    """Fit a smoothing thin-plate spline pa -> pb.

    Solves the TPS linear system with numpy only:

        [ K + lmbda*I   P ] [w]   [pb]
        [ P^T           0 ] [a] = [0 ]

    where K[i,j] = U(||pa_i - pa_j||), P = [1, x, y], w are the
    non-affine weights and a the affine coefficients. The ridge term
    applies to the non-affine block only, so the global affine trend is
    never penalized.

    Parameters
    ----------
    pa, pb : (N, 2) array-like, N >= 3 non-collinear points.
    lmbda : float >= 0. Smoothing strength; 0 = exact interpolation.

    Returns
    -------
    dict with keys:
        'warp'           callable mapping (M, 2) -> (M, 2)
        'bending_energy' float, trace(w^T K w) — 0 for a pure affine fit
        'n_points'       int, number of control points
        'lmbda'          the smoothing used
    """
    pa = np.asarray(pa, dtype=np.float64)
    pb = np.asarray(pb, dtype=np.float64)
    if pa.ndim != 2 or pa.shape[1] != 2 or pb.shape != pa.shape:
        raise ValueError(f"pa and pb must both be (N,2); got {pa.shape}, {pb.shape}")
    n = pa.shape[0]
    if n < 3:
        raise ValueError(f"TPS needs >= 3 points, got {n}")
    if lmbda < 0:
        raise ValueError(f"lmbda must be >= 0, got {lmbda}")

    dists = np.linalg.norm(pa[:, None, :] - pa[None, :, :], axis=-1)
    K = _tps_kernel(dists)
    P = np.hstack([np.ones((n, 1)), pa])

    top = np.hstack([K + lmbda * np.eye(n), P])
    bottom = np.hstack([P.T, np.zeros((3, 3))])
    L = np.vstack([top, bottom])
    Y = np.vstack([pb, np.zeros((3, 2))])

    try:
        sol = np.linalg.solve(L, Y)
    except np.linalg.LinAlgError:
        # Degenerate control geometry: least-squares fallback rather than
        # an exception mid-pipeline. Callers still gate the result.
        sol, *_ = np.linalg.lstsq(L, Y, rcond=None)
    w = sol[:n]      # (N, 2) non-affine weights
    a = sol[n:]      # (3, 2) affine coefficients [t; ax; ay]

    bending_energy = float(np.trace(w.T @ K @ w))

    def warp(q):
        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 2:
            raise ValueError(f"warp expects (M,2), got {q.shape}")
        d = np.linalg.norm(q[:, None, :] - pa[None, :, :], axis=-1)
        Kq = _tps_kernel(d)                      # (M, N)
        aff = a[0] + q[:, [0]] * a[1] + q[:, [1]] * a[2]  # (M, 2)
        return aff + Kq @ w

    return {"warp": warp, "bending_energy": bending_energy,
            "n_points": int(n), "lmbda": float(lmbda)}


def tps_residual(pa, pb, warp):
    """Per-point Euclidean residual of `warp` on control points.

    Returns (N,) residuals and the RMSE. With lmbda=0 on non-degenerate
    control points the residuals are ~0 by construction — that measures
    the solver, not the registration; score on held-out points instead.
    """
    pa = np.asarray(pa, dtype=np.float64)
    pb = np.asarray(pb, dtype=np.float64)
    pred = np.asarray(warp(pa), dtype=np.float64)
    resid = np.linalg.norm(pred - pb, axis=1)
    rmse = float(np.sqrt(np.mean(resid ** 2))) if resid.size else float("nan")
    return resid, rmse
