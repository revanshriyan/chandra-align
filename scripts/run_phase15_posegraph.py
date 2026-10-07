"""Phase 15 driver — joint 2D pose-graph optimization.

1. Looks for a REAL image triplet (three images with gated pairwise
   affines on all three legs). None exists in this repo today (Phase 9
   produced pairwise IIRS<->TMC-2 legs only; Phase 13 produced pairs, not
   triplets) -> records an ABSTAIN for real data with the reason. Nothing
   is fabricated.
2. Runs a SYNTHETIC demo: known ground-truth absolute affines, noisy
   synthetic correspondences per leg, each leg fitted and passed through
   the real frozen gate (validate_registration_gate); gated legs go to
   the optimizer; loop misclosure is reported before/after.
3. Writes results/phase15_posegraph.json.

Run: PYTHONPATH=. python3 scripts/run_phase15_posegraph.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chandra_align.metrics.quadrant import validate_registration_gate
from chandra_align.optimize.posegraph import optimize_posegraph, GATED_VERDICTS

OUT = os.path.join(os.path.dirname(__file__), "..", "results",
                   "phase15_posegraph.json")
RNG = np.random.default_rng(1507)


def fit_affine(src, dst):
    A = np.column_stack([src, np.ones(len(src))])      # (N,3)
    coef, *_ = np.linalg.lstsq(A, dst, rcond=None)      # (3,2)
    M = coef.T                                          # (2,3)
    pred = src @ M[:, :2].T + M[:, 2]
    rmse = float(np.hypot(*(pred - dst).T).mean())
    return M, rmse


def quad_stats(pts):
    """Quadrant counts + entropy for gate input (frame split at median)."""
    mx, my = np.median(pts[:, 0]), np.median(pts[:, 1])
    counts = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    for x, y in pts:
        if x >= mx and y >= my:
            counts["Q1"] += 1
        elif x < mx and y >= my:
            counts["Q2"] += 1
        elif x < mx and y < my:
            counts["Q3"] += 1
        else:
            counts["Q4"] += 1
    n = max(1, len(pts))
    ent = -sum((c / n) * np.log2(c / n) for c in counts.values() if c > 0)
    return counts, float(ent)


def gate_leg(M, rmse, pts):
    counts, ent = quad_stats(pts)
    _label, verdict = validate_registration_gate(
        rmse, len(pts), 8, ent, counts)
    return verdict, {"rmse": rmse, "n": len(pts), "entropy": ent,
                     "quads": counts}


def real_triplet_search():
    """Do three real images with gated pairwise affines exist? (No.)"""
    # Phase 9: pairwise IIRS<->TMC-2 only. Phase 13: pairs (OHRC stereo,
    # TMC-2 fore/nadir), never three images with all three gated legs.
    # Checked results/phase9_iirs_retry.json, results/table_phase13_windows.csv.
    return {
        "status": "ABSTAIN",
        "reason": ("No real-data image triplet exists: the repo holds gated "
                   "PAIRWISE registrations (Phase 9 IIRS<->TMC-2, Phase 13 "
                   "OHRC stereo and TMC-2 fore/nadir pairs) but no set of "
                   "three images with gated affines on all three legs "
                   "(A->B, B->C, C->A). Optimizing a fabricated triplet "
                   "would be dishonest; skipping real data."),
    }


def synthetic_demo():
    # Ground-truth absolute poses (image i -> reference). T0 pinned.
    def rt(deg, tx, ty):
        t = np.deg2rad(deg)
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, tx], [s, c, ty]])
    T_true = [np.eye(2, 3), rt(1.5, 40.0, -25.0), rt(-2.0, -60.0, 80.0)]

    def true_M(i, j):
        Hi = np.eye(3); Hi[:2, :] = T_true[i]
        Hj = np.eye(3); Hj[:2, :] = T_true[j]
        return (Hj @ np.linalg.inv(Hi))[:2, :]

    legs = [(0, 1, 0.35), (1, 2, 1.10), (2, 0, 0.45)]  # (i, j, noise_sigma)
    edges, leg_info = [], []
    for i, j, sig in legs:
        Mij = true_M(i, j)
        g = np.linspace(-400, 400, 9)
        yy, xx = np.meshgrid(g, g, indexing="ij")
        src = np.stack([xx.ravel(), yy.ravel()], axis=1)
        dst = src @ Mij[:, :2].T + Mij[:, 2]
        dst = dst + RNG.normal(0, sig, dst.shape)   # measurement noise
        M_fit, rmse = fit_affine(src, dst)
        verdict, stats = gate_leg(M_fit, rmse, dst)
        leg_info.append({"pair": f"{i}->{j}", "verdict": verdict,
                         "rmse_px": round(rmse, 4),
                         "n_inliers": stats["n"],
                         "noise_sigma": sig})
        if verdict not in GATED_VERDICTS:
            leg_info[-1]["note"] = "leg rejected by gate; excluded"
            continue
        edges.append({"i": i, "j": j, "M": M_fit,
                      "weight": float(stats["n"]) / 81.0, "verdict": verdict})

    if len(edges) < 3:
        return {"status": "ABSTAIN",
                "reason": "fewer than 3 gated synthetic legs; cannot close loop",
                "legs": leg_info}

    g = np.linspace(-512, 512, 8)
    yy, xx = np.meshgrid(g, g, indexing="ij")
    probes = np.stack([xx.ravel(), yy.ravel()], axis=1)
    res = optimize_posegraph(edges, 3, probe_points=probes)
    before, after = res["misclosure_before_px"], res["misclosure_after_px"]
    accepted = (after is not None and before is not None
                and after < 0.5 * before and after < 1.0)
    return {
        "status": "OPTIMIZED",
        "legs": leg_info,
        "misclosure_before_px": before,
        "misclosure_after_px": after,
        "per_edge_rmse_px": [round(v, 4) for v in res["per_edge_rmse_px"]],
        "nfev": res["nfev"],
        "acceptance": ("met (synthetic-only)" if accepted
                       else "NOT met — see misclosure values"),
        "note": ("Ground-truth affines known; each leg independently fitted "
                 "from noisy synthetic correspondences and passed through "
                 "the real frozen gate before optimization."),
    }


def main():
    out = {
        "real_triplet": real_triplet_search(),
        "synthetic": synthetic_demo(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("real triplet:", out["real_triplet"]["status"])
    s = out["synthetic"]
    print("synthetic:", s["status"], "| acceptance:", s.get("acceptance"))
    if s["status"] == "OPTIMIZED":
        print(f"  misclosure {s['misclosure_before_px']:.4f} -> "
              f"{s['misclosure_after_px']:.4f} px")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
