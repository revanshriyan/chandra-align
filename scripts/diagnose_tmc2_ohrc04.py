"""A22 follow-up: tmc2 hook-inlier characterization + ohrc_04 inlier trace.

Replicates the pipeline's hook input exactly (match_pair_hf -> hook RANSAC)
via the same calls _align_core makes, then characterizes:
  - tmc2: n raw matches, n hook-RANSAC inliers, inlier residual stats,
    spatial spread (quadrant counts, bounding-box fraction) -> degenerate
    clique vs real consensus; stage sweep detail already in table_blockers.
  - ohrc_04: full inlier-count trace hook -> balance -> bucket -> RANSAC ->
    NCC, with per-quadrant counts at each step, written to
    results/trace_ohrc04.csv.
Deterministic (seed 7), diagnosis only.
"""
import os
import sys
import csv

os.environ["CHANDRA_DEFORM_FIELD"] = "1"
sys.path.insert(0, "/tmp/stubs")
sys.path.insert(0, os.path.expanduser("~/workspace/chandra-align"))

import cv2
import numpy as np

import app
import chandra_align.deform_field as dfmod

CROP = os.path.expanduser("~/workspace/chandra-align/data/benchmark_crops")

# Capture stage hook inputs.
captured = {}
_real_stage = dfmod.apply_deform_field_stage


def stage_spy(p_src, p_ref, M, lambda_grid=None, seed=7):
    kw = {} if lambda_grid is None else {"lambda_grid": lambda_grid}
    info = _real_stage(np.asarray(p_src), np.asarray(p_ref), np.asarray(M),
                       seed=seed, **kw)
    captured["last"] = {
        "p_src": np.asarray(p_src), "p_ref": np.asarray(p_ref),
        "M": np.asarray(M), "n": int(np.asarray(p_src).shape[0]),
        "info": {k: v for k, v in info.items()
                 if k not in ("residuals_vec", "residuals_mag", "field")},
    }
    return info


dfmod.apply_deform_field_stage = stage_spy

# Trace wrappers.
trace_log = []
_ransac_calls = [0]
_real_ransac = app._estimate_partial_affine_with_threshold
_real_balance = app.select_quadrant_balanced_matches
_real_bucket = app.select_distributed_matches
_real_ncc = app.refine_subpixel_ncc


def qcounts(pts, shape):
    h, w = shape[:2]
    q = np.zeros(4, dtype=int)
    for x, y in np.asarray(pts).reshape(-1, 2):
        qi = (0 if y < h / 2 else 2) + (0 if x < w / 2 else 1)
        q[qi] += 1
    return q.tolist()


def ransac_spy(pts_sec, pts_ref, thr):
    _ransac_calls[0] += 1
    M, inl = _real_ransac(pts_sec, pts_ref, thr)
    n_in = int(np.asarray(inl).sum()) if inl is not None else 0
    trace_log.append(("ransac#%d" % _ransac_calls[0], len(pts_sec), n_in,
                      qcounts(np.asarray(pts_ref)[np.asarray(inl).astype(bool)]
                              if inl is not None and n_in else [], (2048, 2048))))
    return M, inl


def balance_spy(pts_ref, pts_sec, shape, quota_per_quadrant=50):
    out = _real_balance(pts_ref, pts_sec, shape,
                        quota_per_quadrant=quota_per_quadrant)
    trace_log.append(("balance", len(pts_ref), len(out[0]),
                      qcounts(out[0], shape)))
    return out


def bucket_spy(pts_ref, pts_sec, shape, **kw):
    out = _real_bucket(pts_ref, pts_sec, shape, **kw)
    trace_log.append(("bucket", len(pts_ref), len(out[0]),
                      qcounts(out[0], shape)))
    return out


def ncc_spy(*a, **k):
    r = _real_ncc(*a, **k)
    trace_log.append(("ncc_refine", len(a[2]), len(r[0]), r[2].get("status")))
    return r


def load(pair):
    ref = cv2.imread(os.path.join(CROP, "%s_reference.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(os.path.join(CROP, "%s_source.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    return ref, sec


def main():
    # ---- tmc2 hook characterization ----
    print("=== tmc2 hook-inlier characterization ===")
    tmc2_rows = []
    for pair in ["tmc2_01", "tmc2_02", "tmc2_03", "tmc2_04", "tmc2_05", "tmc2_06"]:
        ref, sec = load(pair)
        captured.pop("last", None)
        try:
            out = app._align_core(ref, sec, reference_sensor_name="TMC-2")
            status = out.get("status_code")
        except Exception as exc:
            status = "exception:%s" % type(exc).__name__
        cap = captured.get("last")
        if cap is None:
            print("%s: stage never attempted (%s)" % (pair, status))
            tmc2_rows.append([pair, status, "", "", "", "", "", ""])
            continue
        ps, pr, M = cap["p_src"], cap["p_ref"], cap["M"]
        n = cap["n"]
        # Residual stats of the hook set under M_hook.
        ones = np.ones((n, 1))
        pred = (M[:, :2] @ ps.T).T + M[:, 2]
        res = np.linalg.norm(pr - pred, axis=1)
        # Spatial spread: fraction of image diagonal covered by bbox.
        bb = pr.max(axis=0) - pr.min(axis=0)
        spread = float(np.linalg.norm(bb) / np.linalg.norm(ref.shape[:2][::-1]))
        info = cap["info"]
        sweep = info.get("lambda_sweep") or []
        best = min(sweep, key=lambda s: s[1]) if sweep else (None, None)
        print("%s: %s | hook_n=%d res_med=%.2e res_max=%.3f spread=%.3f "
              "q=%s | reason=%s best_l=%.3s best_ho=%.2e aff_ho=%.2e" % (
                  pair, status, n, float(np.median(res)), float(res.max()),
                  spread, qcounts(pr, ref.shape), info.get("reason"),
                  best[0], best[1], info.get("heldout_rmse_affine_px")))
        tmc2_rows.append([pair, status, n, "%.2e" % float(np.median(res)),
                          "%.3f" % float(res.max()), "%.3f" % spread,
                          qcounts(pr, ref.shape), info.get("reason")])
    with open(os.path.expanduser("~/workspace/chandra-align/results/table_tmc2_hook.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "status", "hook_n_inliers", "hook_resid_median_px",
                    "hook_resid_max_px", "hook_spread_frac",
                    "hook_quadrant_counts", "stage_reason"])
        w.writerows(tmc2_rows)
    print("wrote results/table_tmc2_hook.csv")

    # ---- ohrc_04 trace ----
    print("\n=== ohrc_04 inlier trace ===")
    app._estimate_partial_affine_with_threshold = ransac_spy
    app.select_quadrant_balanced_matches = balance_spy
    app.select_distributed_matches = bucket_spy
    app.refine_subpixel_ncc = ncc_spy
    trace_log.clear()
    _ransac_calls[0] = 0
    ref, sec = load("ohrc_04")
    out = app._align_core(ref, sec, reference_sensor_name="OHRC")
    app._estimate_partial_affine_with_threshold = _real_ransac
    app.select_quadrant_balanced_matches = _real_balance
    app.select_distributed_matches = _real_bucket
    app.refine_subpixel_ncc = _real_ncc
    jm = out.get("judge_metrics", {})
    gm = jm.get("global_metrics", {})
    print("final: %s gate=%.4f n=%s ent=%.3f quads=%s qcounts=%s" % (
        out.get("status_code"), jm.get("rmse_gate_px"), gm.get("inlier_count"),
        gm.get("spatial_entropy_score"), jm.get("active_quadrants_count"),
        jm.get("quadrant_counts")))
    with open(os.path.expanduser("~/workspace/chandra-align/results/trace_ohrc04.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "n_in", "n_out", "quadrant_counts_out"])
        for name, n_in, n_out, qc in trace_log:
            w.writerow([name, n_in, n_out, qc])
            print("  %-10s in=%6d out=%6s quads=%s" % (name, n_in, n_out, qc))
    print("wrote results/trace_ohrc04.csv")


if __name__ == "__main__":
    main()
