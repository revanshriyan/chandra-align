"""Phase 10: run the independent verification layer on synthetic pairs.

Covers the three trust legs:
1. Pixel area-check on synthetic pairs (correct transform -> verified,
   wrong transform -> not verified).
2. Triplet loop-closure on a synthetic A->B->C->A chain.
3. Threshold calibration report (thresholds measured, not hand-waved).

Writes results/phase10_verification.json
"""
import json
import numpy as np

from chandra_align.testing import make_pair_shift
from chandra_align.trust.verify import verify_pair, calibrate_thresholds
from chandra_align.trust.loop_closure import loop_closure


def main():
    out = {"pairs": [], "triplet": None, "calibration": None}

    # --- leg 1: area-check on synthetic pairs ---
    for seed, (dx, dy) in enumerate([(5.0, -2.0), (-8.0, 3.0), (0.0, 0.0)]):
        ref, mov, M_gt = make_pair_shift(shape=(256, 256), dx=dx, dy=dy, seed=seed)
        r = verify_pair(mov, ref, M_gt, pair_id=f"synth_shift_{seed}")
        out["pairs"].append({k: r[k] for k in
                             ("pair_id", "overall", "n_cells", "n_verified",
                              "n_weak", "n_no_evidence", "verified_fraction",
                              "summary")})
        # wrong-transform control
        M_bad = M_gt.copy()
        M_bad[:, 2] += 25.0
        rb = verify_pair(mov, ref, M_bad, pair_id=f"synth_shift_{seed}_wrong")
        out["pairs"].append({k: rb[k] for k in
                             ("pair_id", "overall", "n_cells", "n_verified",
                              "n_weak", "n_no_evidence", "verified_fraction",
                              "summary")})

    # --- leg 2: triplet loop closure (synthetic chain) ---
    M_ab = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 2.0]])
    M_bc = np.array([[1.0, 0.0, -3.0], [0.0, 1.0, 1.0]])
    M_ca = np.array([[1.0, 0.0, -2.0], [0.0, 1.0, -3.0]])
    pts = np.array([[10.0, 10.0], [50.0, 80.0], [100.0, 20.0], [200.0, 150.0]])
    out["triplet"] = loop_closure(M_ab, M_bc, M_ca, pts)

    # --- leg 3: calibration ---
    out["calibration"] = calibrate_thresholds()

    with open("results/phase10_verification.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote results/phase10_verification.json")
    for p in out["pairs"]:
        print(" ", p["summary"])
    print("  triplet:", "CLOSED" if out["triplet"]["closed"] else "OPEN",
          f"({out['triplet']['cyclic_rmse_px']:.2e} px)")
    print("  calibration separates:", out["calibration"]["separates"])


if __name__ == "__main__":
    main()
