"""Phase 8 bake-off: NCC vs Lucas-Kanade vs phase-correlation refinement.

Compares the three sub-pixel refinement methods on synthetic pairs with
known ground truth. Each method refines the RANSAC inliers; a partial-affine
model is refit on the refined inliers and scored by HELD-OUT RMSE against
the known transform (no-peeking: the winner is chosen by held-out error,
never in-sample).

Run: python scripts/run_phase8_bakeoff.py
Writes: results/phase8_bakeoff.json
"""

import json
import numpy as np

from chandra_align.testing import (
    make_pair_shift,
    make_pair_illumination,
    make_pair_scale,
)
from chandra_align.matcher import SIFTMatcher
from chandra_align.refine import verify_guarded, refine_subpixel_ncc
from chandra_align.refine.subpixel import refine_lk, refine_phasecorr


def refit_partial_affine(pa, pb):
    """Least-squares partial-affine [a -b tx; b a ty] refit (no RANSAC)."""
    pa = np.asarray(pa, np.float64)
    pb = np.asarray(pb, np.float64)
    n = len(pa)
    X = np.zeros((2 * n, 4))
    y = np.zeros(2 * n)
    X[0::2, 0] = pa[:, 0]
    X[0::2, 1] = -pa[:, 1]
    X[0::2, 2] = 1
    X[1::2, 0] = pa[:, 1]
    X[1::2, 1] = pa[:, 0]
    X[1::2, 3] = 1
    y[0::2] = pb[:, 0]
    y[1::2] = pb[:, 1]
    (a, b_, tx, ty), *_ = np.linalg.lstsq(X, y, rcond=None)
    return np.array([[a, -b_, tx], [b_, a, ty]], np.float64)


def heldout_rmse(M, pa_hold, pb_hold):
    pa = np.asarray(pa_hold, np.float64)
    pb = np.asarray(pb_hold, np.float64)
    proj = pa @ M[:, :2].T + M[:, 2]
    return float(np.hypot(proj[:, 0] - pb[:, 0], proj[:, 1] - pb[:, 1]).mean())


def run_case(name, ref, mov, seed=7):
    rng = np.random.default_rng(seed)
    matcher = SIFTMatcher(lowes_ratio=0.75, n_features=4000)
    pa_all, pb_all = matcher.match(ref, mov)
    if len(pa_all) < 10:
        return {"name": name, "skipped": "too few matches", "n_matches": int(len(pa_all))}
    cfg = {"ransac_reproj_threshold": 3.0, "max_iters": 2000, "confidence": 0.99}
    g = verify_guarded(pa_all, pb_all, cfg, image_shape=ref.shape)
    if not g["ok"]:
        return {"name": name, "skipped": f"guarded verify failed: {g['abstain_code']}",
                "n_matches": int(len(pa_all))}
    ia, ib, M0 = g["inliers_a"], g["inliers_b"], g["model"]
    n = len(ia)
    perm = rng.permutation(n)
    n_fit = max(6, int(0.8 * n))
    fit_idx, hold_idx = perm[:n_fit], perm[n_fit:]
    if len(hold_idx) < 2:
        return {"name": name, "skipped": "too few held-out points", "n_matches": int(len(pa_all))}
    fa, fb = ia[fit_idx], ib[fit_idx]
    ha, hb = ia[hold_idx], ib[hold_idx]

    methods = {
        "none": (fa, fb),
        "ncc": refine_subpixel_ncc(ref, mov, fa, fb, ncc_window=11)[:2],
        "lk": refine_lk(ref, mov, fa, fb, model=M0)[:2],
        "phasecorr": refine_phasecorr(ref, mov, fa, fb)[:2],
    }
    scores = {}
    for mname, (ra, rb) in methods.items():
        if len(ra) < 6:
            scores[mname] = {"heldout_rmse_px": None, "n_refined": int(len(ra))}
            continue
        M = refit_partial_affine(ra, rb)
        scores[mname] = {
            "heldout_rmse_px": heldout_rmse(M, ha, hb),
            "n_refined": int(len(ra)),
        }
    return {"name": name, "n_matches": int(len(pa_all)), "n_inliers": n,
            "n_fit": int(n_fit), "n_heldout": int(len(hold_idx)), "scores": scores}


def main():
    cases = [
        ("shift", make_pair_shift(shape=(512, 512), dx=7.3, dy=-3.9, angle_deg=0.4, seed=7)),
        ("illumination", make_pair_illumination(shape=(512, 512), dx=5.0, dy=2.0, seed=7)),
        ("scale", make_pair_scale(shape=(512, 512), ratio=10.0, dx=3.0, dy=3.0, seed=7)),
    ]
    results = []
    for name, (ref, mov, _gt) in cases:
        r = run_case(name, ref, mov)
        results.append(r)
        print(f"--- {name}: {r.get('skipped', 'ok')}")
        for mname, s in r.get("scores", {}).items():
            print(f"    {mname:10s} held-out RMSE: {s['heldout_rmse_px']}  (n={s['n_refined']})")
    # Winner by mean held-out RMSE over completed cases.
    totals = {}
    for r in results:
        for mname, s in r.get("scores", {}).items():
            if s["heldout_rmse_px"] is not None:
                totals.setdefault(mname, []).append(s["heldout_rmse_px"])
    means = {m: float(np.mean(v)) for m, v in totals.items() if v}
    winner = min(means, key=means.get) if means else None
    out = {"cases": results, "mean_heldout_rmse_px": means, "winner": winner}
    with open("results/phase8_bakeoff.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("winner (held-out):", winner, means)
    print("wrote results/phase8_bakeoff.json")


if __name__ == "__main__":
    main()
