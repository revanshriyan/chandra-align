"""Phase 11 — robustness battery runner.

Runs the CURRENT default pipeline (measured, never modified) over a
deterministic perturbation battery on seeded synthetic pairs.

Battery (perturbation seed = SEED_BASE + run index; base-pair seeds fixed):
  identity, gaussian_noise{2,5,10}, brightness_shift{-30,+30},
  contrast_gain{0.8,1.25}, gamma_curve{0.7,1.4}, small_rotation{0.5,2.0},
  jpeg_artifacts{50,25}, sun_flip
Base pairs: shift / small-rotation / illumination (seeded generators).

Writes results/phase11_robustness.json — no wall-clock fields, so a re-run
reproduces it byte-for-byte. See chandra_align/eval/robustness.py for the
perturbation contracts and the withheld-accuracy rule.

Run: PYTHONPATH=. python3 scripts/run_phase11_robustness.py
"""

import json

import numpy as np

from chandra_align.eval.robustness import (
    BATTERY_VERSION,
    PERTURBATIONS,
    compose_gt_with_warp,
    measure_pair,
)
from chandra_align.testing import (
    make_pair_illumination,
    make_pair_shift,
)

SEED_BASE = 1000
SHAPE = (512, 512)

BASE_PAIRS = [
    ("shift", lambda: make_pair_shift(shape=SHAPE, dx=7.3, dy=-3.9, seed=7)),
    ("rot1.5", lambda: make_pair_shift(shape=SHAPE, dx=3.0, dy=-2.0,
                                       angle_deg=1.5, seed=11)),
    ("illum", lambda: make_pair_illumination(shape=SHAPE, dx=5.0, dy=2.0, seed=13)),
]

BATTERY = [
    ("identity", {}),
    ("gaussian_noise", {"sigma": 2.0}),
    ("gaussian_noise", {"sigma": 5.0}),
    ("gaussian_noise", {"sigma": 10.0}),
    ("brightness_shift", {"delta": -30.0}),
    ("brightness_shift", {"delta": 30.0}),
    ("contrast_gain", {"gain": 0.8}),
    ("contrast_gain", {"gain": 1.25}),
    ("gamma_curve", {"gamma": 0.7}),
    ("gamma_curve", {"gamma": 1.4}),
    ("small_rotation", {"angle_deg": 0.5}),
    ("small_rotation", {"angle_deg": 2.0}),
    ("jpeg_artifacts", {"quality": 50}),
    ("jpeg_artifacts", {"quality": 25}),
    ("sun_flip", {}),
]


def main():
    runs = []
    idx = 0
    for pair_name, gen in BASE_PAIRS:
        ref, mov, M_gt = gen()
        for pname, params in BATTERY:
            seed = SEED_BASE + idx
            idx += 1
            fn = PERTURBATIONS[pname]
            mov_p, info = fn(mov, seed=seed, **params)
            M_run = M_gt
            if "dM" in info:  # small_rotation: compose the truth through the warp
                M_run = compose_gt_with_warp(M_gt, np.asarray(info["dM"]))
            rec = measure_pair(ref, mov_p, M_run, image_shape=SHAPE)
            rec.update({
                "base_pair": pair_name,
                "perturbation": pname,
                "params": {k: v for k, v in params.items()},
                "seed": seed,
            })
            runs.append(rec)
            print(f"  {pair_name:7s} {pname:16s} {str(params):28s} -> "
                  f"{rec['verdict']:18s} inl={rec['inliers']:4d} "
                  f"rmse={rec['rmse_px']}")
    out = {
        "battery_version": BATTERY_VERSION,
        "checkpoint": {
            "base_pair_seeds": {"shift": 7, "rot1.5": 11, "illum": 13},
            "perturbation_seed_base": SEED_BASE,
            "perturbation_seed_rule": "SEED_BASE + run_index (documented order)",
            "pipeline": "SIFT(nfeatures=10000, cT=0.005, eT=15) + BFMatcher kNN k=2 "
                        "+ Lowe 0.75 -> verify_guarded(ransac=3.0px, 2000 iters, "
                        "conf 0.99) -> quadrant metrics -> validate_registration_gate "
                        "(min_inliers=8). Frozen; measured only.",
            "reproducibility": "no wall-clock or RNG-state fields; cv2 RANSAC "
                               "verified deterministic across processes; re-run "
                               "reproduces this file byte-for-byte.",
        },
        "runs": runs,
    }
    with open("results/phase11_robustness.json", "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("wrote results/phase11_robustness.json")


if __name__ == "__main__":
    main()
