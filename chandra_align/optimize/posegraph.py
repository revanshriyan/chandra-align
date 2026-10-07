"""Phase 15 — joint 2D pose-graph optimization.

Promotes the Phase 10 triplet *check* into an *optimizer*: all images'
2D affines are solved simultaneously so the whole graph is consistent,
instead of trusting each pairwise fit in isolation.

Model
-----
Each image i has an absolute pose T_i (2x3 affine, warp-style: maps image-i
pixel coordinates into a common reference frame). Image 0 is pinned to the
identity and is never optimized. A measured pairwise edge (i -> j) with
affine M_ij (maps image-i coords to image-j coords, same convention as
everywhere else in this repo) should satisfy::

    T_j ~= M_ij o T_i

Residuals are the 6 entries of (M_ij @ T_i^h - T_j^h), weighted per edge.
Solved with scipy.optimize.least_squares, trust-region reflective + Huber
loss (robust to one bad-but-gated edge). Note: scipy's Levenberg-Marquardt
implementation only supports a linear loss, so TRF+Huber is the correct
robust combination; the plan's "LM + Huber" is honored in spirit (robust
least squares) via the only scipy path that supports both.

Safety
------
Gates run BEFORE the optimizer. Every edge must carry a verdict in
GATED_VERDICTS (SUCCESS_SUBPIXEL or COARSE_ADVISORY, the two passing codes
of validate_registration_gate). Anything else — DEGENERATE_FAILURE,
ABSTAIN codes, a missing matrix — is refused with ValueError. Garbage in,
jointly optimized garbage out is not acceptable: the optimizer never sees
ungated pairs.
"""

import numpy as np
from scipy.optimize import least_squares

from ..trust.loop_closure import loop_closure

# The two passing verdicts of validate_registration_gate
# (chandra_align/metrics/quadrant.py). Frozen; the optimizer does not
# define its own acceptance criteria.
GATED_VERDICTS = {"SUCCESS_SUBPIXEL", "COARSE_ADVISORY"}


def _as_h(M):
    """2x3 affine -> 3x3 homogeneous."""
    H = np.eye(3)
    H[:2, :] = np.asarray(M, np.float64).reshape(2, 3)
    return H


def _as_23(H):
    """3x3 homogeneous -> 2x3 affine."""
    return np.asarray(H, np.float64).reshape(3, 3)[:2, :]


def affine_inverse(M):
    """Inverse of a 2x3 affine (raises LinAlgError if singular)."""
    return _as_23(np.linalg.inv(_as_h(M)))


def check_gated(edges):
    """Refuse any edge that did not pass the frozen gates.

    Each edge dict needs: i, j, M (2x3, not None), verdict in GATED_VERDICTS.
    Raises ValueError naming the offending edge.
    """
    for k, e in enumerate(edges):
        tag = f"edge[{k}] ({e.get('i')}->{e.get('j')})"
        if e.get("M") is None:
            raise ValueError(f"{tag}: refused — missing transform (None)")
        if e.get("verdict") not in GATED_VERDICTS:
            raise ValueError(
                f"{tag}: refused — verdict {e.get('verdict')!r} is not gated "
                f"(need one of {sorted(GATED_VERDICTS)})"
            )
    return True


def _chain_init(edges, n_images):
    """Deterministic initialization: breadth-first chaining from image 0.

    Neighbors are visited in (weight desc, index asc) order so the result
    does not depend on input edge order. Undirected traversal; reversed
    edges use the affine inverse.
    """
    adj = {i: [] for i in range(n_images)}
    for e in edges:
        i, j = int(e["i"]), int(e["j"])
        w = float(e.get("weight", 1.0))
        adj[i].append((j, _as_h(e["M"]), w))
        adj[j].append((i, np.linalg.inv(_as_h(e["M"])), w))
    for i in adj:
        adj[i].sort(key=lambda t: (-t[2], t[0]))
    T = [np.eye(3)] + [None] * (n_images - 1)
    seen = {0}
    queue = [0]
    while queue:
        u = queue.pop(0)
        for v, H_uv, _w in adj[u]:
            if v not in seen:
                # T_v ~= M_uv o T_u
                T[v] = H_uv @ T[u]
                seen.add(v)
                queue.append(v)
    for i in range(1, n_images):
        if T[i] is None:  # disconnected node: identity start, optimizer moves it
            T[i] = np.eye(3)
    return T


def optimize_posegraph(edges, n_images, huber_scale=1.0, cycle=None,
                       probe_points=None):
    """Jointly optimize absolute 2D affines for a gated edge set.

    Parameters
    ----------
    edges : list of dict
        Each: {"i": int, "j": int, "M": 2x3 (i->j), "weight": float,
        "verdict": str}. All must be gated (see check_gated).
    n_images : int
        Number of images. Image 0 is pinned to identity.
    huber_scale : float
        Huber f_scale for least_squares (robust to bad-but-gated edges).
        Units are the residual vector's units (translation entries are px):
        set it a few times above the inlier RMSE but well below the size
        of corruption you want downweighted. If it sits below the inlier
        noise, even good edges are linearized and a bad edge can win a
        tug-of-war — the test suite pins this behavior.
    cycle : list of (i, j) tuples, optional
        Directed cycle used to report loop misclosure before/after
        (default [(0,1),(1,2),(2,0)] when n_images == 3).
    probe_points : (N,2) array, optional
        Probe grid for misclosure / per-edge RMSE in px. Defaults to a
        8x8 grid over [-512, 512]^2.

    Returns
    -------
    dict with: T (list of 2x3, T[0] is identity), per_edge_rmse_px,
    misclosure_before_px, misclosure_after_px, nfev, cost, status.
    """
    check_gated(edges)
    if n_images < 2:
        raise ValueError("need at least 2 images")
    if probe_points is None:
        g = np.linspace(-512, 512, 8)
        yy, xx = np.meshgrid(g, g, indexing="ij")
        probe_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    else:
        probe_points = np.asarray(probe_points, np.float64).reshape(-1, 2)

    T0 = _chain_init(edges, n_images)
    x0 = np.concatenate([_as_23(T0[i]).ravel() for i in range(1, n_images)])

    H_meas = [(_as_h(e["M"]), float(e.get("weight", 1.0)),
               int(e["i"]), int(e["j"])) for e in edges]

    def residuals(x):
        T = [np.eye(3)]
        for i in range(1, n_images):
            H = np.eye(3)
            H[:2, :] = x[(i - 1) * 6:i * 6].reshape(2, 3)
            T.append(H)
        out = []
        for Hm, w, i, j in H_meas:
            R = (Hm @ T[i] - T[j])[:2, :].ravel()  # 6 entries, px-ish units
            out.append(w * R)
        return np.concatenate(out)

    sol = least_squares(residuals, x0, method="trf", loss="huber",
                        f_scale=float(huber_scale))
    T_opt = [np.eye(3)]
    for i in range(1, n_images):
        H = np.eye(3)
        H[:2, :] = sol.x[(i - 1) * 6:i * 6].reshape(2, 3)
        T_opt.append(H)
    T_23 = [_as_23(H) for H in T_opt]

    # Per-edge RMSE in px over the probe grid: warp probes by (M_ij o T_i)
    # versus T_j.
    per_edge_rmse = []
    for Hm, _w, i, j in H_meas:
        A = (Hm @ T_opt[i])[:2, :]
        B = T_opt[j][:2, :]
        pa = probe_points @ A[:, :2].T + A[:, 2]
        pb = probe_points @ B[:, :2].T + B[:, 2]
        per_edge_rmse.append(float(np.hypot(*(pa - pb).T).mean()))

    # Loop misclosure before/after on the requested cycle.
    mis_before = mis_after = None
    if cycle is None and n_images == 3:
        cycle = [(0, 1), (1, 2), (2, 0)]
    if cycle:
        def edge_M(ii, jj):
            for Hm, _w, i, j in H_meas:
                if (i, j) == (ii, jj):
                    return _as_23(Hm)
                if (i, j) == (jj, ii):
                    return affine_inverse(_as_23(Hm))
            return None
        Mab = [edge_M(a, b) for a, b in cycle]
        if all(m is not None for m in Mab):
            mis_before = loop_closure(Mab[0], Mab[1], Mab[2],
                                      probe_points)["cyclic_rmse_px"]
            # Optimized pairwise affines from absolute poses:
            # M_ij^opt = T_j o T_i^{-1}.
            Mopt = [_as_23(T_opt[b] @ np.linalg.inv(T_opt[a]))
                    for a, b in cycle]
            mis_after = loop_closure(Mopt[0], Mopt[1], Mopt[2],
                                     probe_points)["cyclic_rmse_px"]

    return {
        "T": [m.tolist() for m in T_23],
        "per_edge_rmse_px": per_edge_rmse,
        "misclosure_before_px": mis_before,
        "misclosure_after_px": mis_after,
        "nfev": int(sol.nfev),
        "cost": float(sol.cost),
        "status": "OPTIMIZED",
    }
