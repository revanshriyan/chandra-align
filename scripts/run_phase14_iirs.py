"""Phase 14: run the phase-congruency arm on the real IIRS<->TMC-2 pair.

IIRS 2852nm band (crop [0:4634, 0:218], per-column-median destriped) vs
TMC-2 common-GSD. Correspondences go through verify_guarded (Phase 8) and
Gate 3 transform conditioning. Thresholds are NOT tuned: Lowe 0.75,
RANSAC defaults from the Phase 9 harness, frozen Gate 3 limits.
"""
import numpy as np

from chandra_align.xmodal import pc_arm
from chandra_align.refine import verify_guarded
from chandra_align.metrics.conditioning import check_transform_conditioning
from chandra_align.metrics import compute_quadrant_metrics, validate_registration_gate

IIRS_PATH = "/home/hatch/workspace/qa_lunar/pradan_iirs2/bands/iirs2_b128_2852.6nm.npy"
TMC_PATH = "/home/hatch/workspace/phase9_tmc_common_gsd.npy"


def main():
    raw = np.asarray(np.load(IIRS_PATH, mmap_mode="r")[0:4634, 0:218]).astype(np.float64)
    col_med = np.median(raw, axis=0)
    iirs = raw - (col_med - np.median(col_med))[None, :]
    tmc = np.load(TMC_PATH)
    H, W = iirs.shape
    print(f"IIRS {iirs.shape} (destriped) vs TMC-2 {tmc.shape}", flush=True)

    print("computing PC maps + matching ...", flush=True)
    pa, pb = pc_arm.match(iirs, tmc)
    print(f"Lowe-good correspondences: {len(pa)}", flush=True)

    cfg = {"ransac_reproj_threshold": 3.0, "max_iters": 5000, "confidence": 0.99}
    g = verify_guarded(pa, pb, cfg, image_shape=(H, W))
    print(f"verify_guarded: ok={g['ok']} abstain={g['abstain_code']} "
          f"raw={g['n_raw']} unique={g['n_unique']}")
    if not g["ok"]:
        print("verdict: NO-FIT (abstain / gate rejection before scoring)")
        return
    M = g["model"]
    ia, ib = g["inliers_a"], g["inliers_b"]
    proj = ia @ M[:, :2].T + M[:, 2]
    resid = np.hypot(proj[:, 0] - ib[:, 0], proj[:, 1] - ib[:, 1])
    rmse = float(resid.mean())
    ok3, rep3 = check_transform_conditioning(M, rmse_px=rmse)
    counts, entropy = compute_quadrant_metrics(ia, resid, (H, W))
    msg, code = validate_registration_gate(rmse, len(ia), 8, entropy, counts, model=M)
    print(f"RANSAC inliers: {len(ia)}  rmse={rmse:.4f}px  entropy={entropy:.3f}")
    print(f"Gate3: {'PASS' if ok3 else 'FAIL'}  gate verdict: {code}")
    print(f"M =\n{np.array2string(M, precision=3, suppress_small=True)}")
    print(f"message: {msg}")


if __name__ == "__main__":
    main()
