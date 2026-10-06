"""Phase 9 — IIRS retry on the verified 2023-12-25 crop with the full toolkit.

Arms (each scored in IIRS working pixels, behind the Phase 8 gates):
  sift_raw      SIFT + Lowe + RANSAC on destriped pair
  sift_wallis   SIFT + Lowe + RANSAC on Wallis-normalized pair (A/B)
  goa_nmi       gradient-orientation preselection -> NMI rerank -> RANSAC
  sift_tps      SIFT rescue + smoothing TPS (lmbda=0.05), >=13 inliers + Gate 3
  loftr         LoFTR dense arm (skipped honestly if torch/kornia missing)

Run: PYTHONPATH=. python scripts/run_phase9_iirs_retry.py
Writes: results/phase9_iirs_retry.json, results/table_phase09_iirs_retry.csv
"""

import csv
import json
import numpy as np

IIRS_PATH = "/home/hatch/workspace/phase9_iirs_destriped.npy"
TMC_PATH = "/home/hatch/workspace/phase9_tmc_common_gsd.npy"


def u8(a):
    a = np.asarray(a, dtype=np.float64)
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) * (255.0 / (hi - lo + 1e-9)), 0, 255).astype(np.uint8)


def sift_pairs(A8, B8, lowe=0.75):
    import cv2
    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.005, edgeThreshold=15)
    ka, da = sift.detectAndCompute(A8, None)
    kb, db = sift.detectAndCompute(B8, None)
    info = {"n_det_a": len(ka), "n_det_b": len(kb), "n_desc_a": 0 if da is None else len(da),
            "n_desc_b": 0 if db is None else len(db), "n_raw": 0, "n_lowe": 0}
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2)), info
    matches = cv2.BFMatcher().knnMatch(da, db, k=2)
    info["n_raw"] = len(matches)
    good = [m for m, n in matches if m.distance < lowe * n.distance]
    info["n_lowe"] = len(good)
    pa = np.float64([[ka[m.queryIdx].pt[0], ka[m.queryIdx].pt[1]] for m in good])
    pb = np.float64([[kb[m.trainIdx].pt[0], kb[m.trainIdx].pt[1]] for m in good])
    return pa, pb, info


def gate_verdict(pa, pb, image_shape):
    from chandra_align.refine import verify_guarded
    from chandra_align.metrics import compute_quadrant_metrics, validate_registration_gate
    cfg = {"ransac_reproj_threshold": 3.0, "max_iters": 2000, "confidence": 0.99}
    g = verify_guarded(pa, pb, cfg, image_shape=image_shape)
    out = {"abstain": g["abstain_code"], "n_raw": g["n_raw"], "n_unique": g["n_unique"],
           "verdict": "ABSTAIN", "rmse_px": None, "inliers": 0}
    if not g["ok"]:
        return out
    M = g["model"]
    ia, ib = g["inliers_a"], g["inliers_b"]
    proj = ia @ M[:, :2].T + M[:, 2]
    resid = np.hypot(proj[:, 0] - ib[:, 0], proj[:, 1] - ib[:, 1])
    rmse = float(resid.mean())
    qm_counts, qm_entropy = compute_quadrant_metrics(ia, resid, image_shape)
    msg, code = validate_registration_gate(rmse, len(ia), 8, qm_entropy,
                                           qm_counts, model=M)
    out.update({"verdict": code, "rmse_px": rmse, "inliers": int(len(ia)),
                "entropy": float(qm_entropy), "quadrants": str(qm_counts),
                "message": msg})
    return out


def main():
    tmc = np.load(TMC_PATH)
    H, W = tmc.shape
    from chandra_align.xmodal.wallis import wallis_normalize

    arms = {}
    # --- multi-band SIFT funnel (712nm destriped / 2852nm / 5009nm) ---
    bands = {
        "sift_712nm": np.load(IIRS_PATH),
        "sift_2852nm": None,  # loaded below
        "sift_5009nm": None,
    }
    for key, bpath in [("sift_2852nm",
                        "/home/hatch/workspace/qa_lunar/pradan_iirs2/bands/iirs2_b128_2852.6nm.npy"),
                       ("sift_5009nm",
                        "/home/hatch/workspace/qa_lunar/pradan_iirs2/bands/iirs2_b256_5009.7nm.npy")]:
        raw = np.asarray(np.load(bpath, mmap_mode="r")[0:4634, 0:218]).astype(np.float64)
        col_med = np.median(raw, axis=0)
        bands[key] = (raw - (col_med - np.median(col_med))[None, :]).astype(np.float32)
    for arm_name, iirs in bands.items():
        pa, pb, info = sift_pairs(u8(iirs), u8(tmc))
        arms[arm_name] = {"funnel": info}
        arms[arm_name].update(gate_verdict(pa, pb, (H, W)))
    # --- sift_wallis (on 712nm) ---
    iirs = bands["sift_712nm"]
    pa, pb, info = sift_pairs(u8(wallis_normalize(iirs)), u8(wallis_normalize(tmc)))
    arms["sift_wallis"] = {"funnel": info}
    arms["sift_wallis"].update(gate_verdict(pa, pb, (H, W)))
    # --- goa_nmi ---
    # Cross-modal correspondences via NMI template search: for each
    # polarity-invariant GOA point in IIRS, find the max-NMI location in
    # TMC-2 over a ±40 px window (the crops are at common GSD over the same
    # geographic area, so the true transform is near-identity).
    try:
        from chandra_align.xmodal.goa_nmi import (
            gradient_magnitude_points, nmi_score)
        A8, B8 = u8(iirs), u8(tmc)
        ca = gradient_magnitude_points(A8, n=400)
        W_win, S = 21, 40
        r = W_win // 2
        cands_a, cands_b, cands_nmi = [], [], []
        for (x, y) in ca[:150]:
            xi, yi = int(round(x)), int(round(y))
            if not (r <= xi < W - r and r <= yi < H - r):
                continue
            patch_a = A8[yi - r:yi + r + 1, xi - r:xi + r + 1]
            x0, x1 = max(r, xi - S), min(W - r, xi + S + 1)
            y0, y1 = max(r, yi - S), min(H - r, yi + S + 1)
            best, best_xy = -1.0, None
            for yy in range(y0, y1, 2):
                for xx in range(x0, x1, 2):
                    patch_b = B8[yy - r:yy + r + 1, xx - r:xx + r + 1]
                    s = nmi_score(patch_a, patch_b)
                    if s > best:
                        best, best_xy = s, (xx, yy)
            if best_xy is not None and best > 0.35:
                cands_a.append((x, y))
                cands_b.append(best_xy)
                cands_nmi.append(best)
        cands_a = np.float64(cands_a).reshape(-1, 2)
        cands_b = np.float64(cands_b).reshape(-1, 2)
        arms["goa_nmi"] = {"n_searched": 150, "n_candidates": int(len(cands_a)),
                           "nmi_mean": float(np.mean(cands_nmi)) if cands_nmi else 0.0}
        arms["goa_nmi"].update(gate_verdict(cands_a, cands_b, (H, W)))
    except Exception as e:
        arms["goa_nmi"] = {"error": f"{type(e).__name__}: {e}"}
    # --- sift_tps ---
    try:
        from chandra_align.xmodal.sift_rescue import sift_rescue_tps
        pa0, pb0, _ = sift_pairs(u8(iirs), u8(tmc))
        r = sift_rescue_tps(u8(iirs), u8(tmc), pa0, pb0)
        arms["sift_tps"] = {k: v for k, v in r.items() if k != "warp"}
        arms["sift_tps"]["has_warp"] = r.get("warp") is not None
    except Exception as e:
        arms["sift_tps"] = {"error": f"{type(e).__name__}: {e}"}
    # --- loftr ---
    try:
        from chandra_align.xmodal.loftr_arm import LoFTRMatcher, loftr_available
        if not loftr_available():
            arms["loftr"] = {"skipped": "torch/kornia unavailable"}
        else:
            la, lb = LoFTRMatcher().match(u8(iirs), u8(tmc))
            arms["loftr"] = {"n_dense": int(len(la))}
            arms["loftr"].update(gate_verdict(la, lb, (H, W)))
    except Exception as e:
        arms["loftr"] = {"error": f"{type(e).__name__}: {e}"}

    with open("results/phase9_iirs_retry.json", "w") as fh:
        json.dump(arms, fh, indent=2, default=str)
    rows = []
    for arm, r in arms.items():
        rows.append({"arm": arm, "verdict": r.get("verdict", r.get("skipped", r.get("error", "?"))),
                     "inliers": r.get("inliers", ""), "rmse_px": r.get("rmse_px", ""),
                     "n_lowe": r.get("funnel", {}).get("n_lowe", "")})
    with open("results/table_phase09_iirs_retry.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "verdict", "inliers", "rmse_px", "n_lowe"])
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({a: {k: v for k, v in r.items() if k != "funnel"}
                      for a, r in arms.items()}, indent=2, default=str)[:2000])
    print("wrote results/phase9_iirs_retry.json")


if __name__ == "__main__":
    main()
