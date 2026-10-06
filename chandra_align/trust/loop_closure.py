"""Phase 10 — triplet loop-closure: pipeline-independent geometric evidence.

Given three pairwise transforms A->B, B->C, C->A (each estimated
independently), compose them: a point should return to itself. The cyclic
RMSE measures global consistency WITHOUT using any single pair's inliers
as truth.

cyclic_rmse < 5 px closes the loop. A failed leg poisons honestly with +inf
(a missing transform is not a zero error).
"""

import numpy as np


def compose_affine(M2, M1):
    """Return M2 o M1 for 2x3 affines (apply M1 first, then M2)."""
    A = np.vstack([np.asarray(M1, np.float64).reshape(2, 3), [0, 0, 1]])
    B = np.vstack([np.asarray(M2, np.float64).reshape(2, 3), [0, 0, 1]])
    return (B @ A)[:2, :]


def loop_closure(M_ab, M_bc, M_ca, points):
    """Triplet closure A->B->C->A.

    Each M may be None (failed leg -> +inf poisoning). points: (N,2) probe
    points in A's frame. Returns dict with cyclic_rmse_px (+inf if any leg
    failed), closed (bool, < 5 px), per-leg status.
    """
    legs = {"A->B": M_ab, "B->C": M_bc, "C->A": M_ca}
    failed = [k for k, M in legs.items() if M is None]
    if failed:
        return {"cyclic_rmse_px": float("inf"), "closed": False,
                "failed_legs": failed, "n_points": 0}
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    M_cycle = compose_affine(M_ca, compose_affine(M_bc, M_ab))
    back = pts @ M_cycle[:, :2].T + M_cycle[:, 2]
    rmse = float(np.hypot(back[:, 0] - pts[:, 0], back[:, 1] - pts[:, 1]).mean())
    return {"cyclic_rmse_px": rmse, "closed": bool(rmse < 5.0),
            "failed_legs": [], "n_points": int(len(pts)),
            "cycle_matrix": M_cycle.tolist()}
