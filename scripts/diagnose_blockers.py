"""A22: diagnose exactly what blocks the remaining pairs. Diagnosis only.

Runs ohrc_04/05/06 + tmc2_01-06 through the true app.py::_align_core with
CHANDRA_DEFORM_FIELD=1 (throwaway /tmp stubs, seed 7). For each pair:
  (1) full gate breakdown + the single component blocking the next tier up;
  (2) stage decline reason + hook diagnostics (tmc2);
  (3) ohrc_04 inlier trace hook -> gate via logging wrappers on the
      intermediate selectors (no pipeline logic touched).
Writes results/table_blockers.csv. Deterministic, fail closed.
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
DESC = {}
PROD = {}
with open(os.path.expanduser("~/workspace/chandra-align/data/benchmark_pairs.csv")) as f:
    for r in csv.DictReader(f):
        DESC[r["pair_id"]] = r["terrain_note"].replace(" (visual only; no DEM label)", "") \
            .replace(" (visual only)", "").replace("south-polar highlands; ", "")
        PROD[r["pair_id"]] = (r["reference_product"], r["source_product"])
PAIRS = [
    ("ohrc_04", "OHRC"), ("ohrc_05", "OHRC"), ("ohrc_06", "OHRC"),
    ("tmc2_01", "TMC-2"), ("tmc2_02", "TMC-2"), ("tmc2_03", "TMC-2"),
    ("tmc2_04", "TMC-2"), ("tmc2_05", "TMC-2"), ("tmc2_06", "TMC-2"),
]

# ---------------------------------------------------------------------------
# Capture the stage's exact hook inputs + full stage_info (incl. sweep detail
# that the pipeline telemetry drops on decline).
# ---------------------------------------------------------------------------
captured = {}
_real_stage = dfmod.apply_deform_field_stage


def stage_spy(p_src, p_ref, M, lambda_grid=None, seed=7):
    kw = {} if lambda_grid is None else {"lambda_grid": lambda_grid}
    info = _real_stage(np.asarray(p_src), np.asarray(p_ref), np.asarray(M),
                       seed=seed, **kw)
    captured["last"] = {
        "n": int(np.asarray(p_src).shape[0]),
        "info": {k: v for k, v in info.items()
                 if k not in ("residuals_vec", "residuals_mag", "field")},
        "M": np.asarray(M).tolist(),
    }
    return info


dfmod.apply_deform_field_stage = stage_spy

# ---------------------------------------------------------------------------
# Trace wrappers for the ohrc_04 inlier pipeline (log only).
# ---------------------------------------------------------------------------
trace_log = []
_ransac_calls = [0]
_real_ransac = app._estimate_partial_affine_with_threshold
_real_balance = app.select_quadrant_balanced_matches
_real_bucket = app.select_distributed_matches
_real_ncc = app.refine_subpixel_ncc


def ransac_spy(pts_sec, pts_ref, thr):
    _ransac_calls[0] += 1
    M, inl = _real_ransac(pts_sec, pts_ref, thr)
    n_in = int(np.asarray(inl).sum()) if inl is not None else 0
    trace_log.append(("ransac#%d" % _ransac_calls[0], len(pts_sec), n_in,
                      None if M is None else "ok"))
    return M, inl


def balance_spy(pts_ref, pts_sec, shape, quota_per_quadrant=50):
    out = _real_balance(pts_ref, pts_sec, shape,
                        quota_per_quadrant=quota_per_quadrant)
    trace_log.append(("balance", len(pts_ref), len(out[0]), None))
    return out


def bucket_spy(pts_ref, pts_sec, shape, **kw):
    out = _real_bucket(pts_ref, pts_sec, shape, **kw)
    trace_log.append(("bucket", len(pts_ref), len(out[0]), None))
    return out


def ncc_spy(*a, **k):
    r = _real_ncc(*a, **k)
    trace_log.append(("ncc_refine", len(a[2]), len(r[0]), r[2].get("status")))
    return r


def enable_trace():
    app._estimate_partial_affine_with_threshold = ransac_spy
    app.select_quadrant_balanced_matches = balance_spy
    app.select_distributed_matches = bucket_spy
    app.refine_subpixel_ncc = ncc_spy
    trace_log.clear()
    _ransac_calls[0] = 0


def disable_trace():
    app._estimate_partial_affine_with_threshold = _real_ransac
    app.select_quadrant_balanced_matches = _real_balance
    app.select_distributed_matches = _real_bucket
    app.refine_subpixel_ncc = _real_ncc


# ---------------------------------------------------------------------------
def run_pair(pair, sensor, trace=False):
    ref = cv2.imread(os.path.join(CROP, "%s_reference.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(os.path.join(CROP, "%s_source.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    if ref is None or sec is None:
        return None, "missing_image"
    captured.pop("last", None)
    if trace:
        enable_trace()
    try:
        out = app._align_core(ref, sec, reference_sensor_name=sensor)
    except Exception as exc:  # fail closed in the harness too
        return None, "exception:%s:%s" % (type(exc).__name__, str(exc)[:100])
    finally:
        if trace:
            disable_trace()
    jm = out.get("judge_metrics", {}) or {}
    gm = jm.get("global_metrics", {}) or {}
    df = jm.get("deform_field_stage", {}) or {}
    row = {
        "pair": pair,
        "pair_description": DESC.get(pair, ""),
        "reference_product": PROD.get(pair, ("", ""))[0],
        "source_product": PROD.get(pair, ("", ""))[1],
        "status": out.get("status_code"),
        "gate_rmse": _f(jm.get("rmse_gate_px")),
        "gate_basis": jm.get("rmse_gate_basis"),
        "n_inliers": gm.get("inlier_count"),
        "entropy": gm.get("spatial_entropy_score"),
        "quads": jm.get("active_quadrants_count"),
        "qcounts": jm.get("quadrant_counts"),
        "stage_applied": df.get("applied"),
        "stage_reason": df.get("reason"),
        "stage_lambda": df.get("lambda_chosen"),
        "hook_n": df.get("n_hook_ransac_inliers"),
        "warp_kind": jm.get("warp_export_kind"),
        "hook": captured.get("last"),
    }
    return row, None


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def blockers(row):
    """Components blocking ACCEPT (frozen gates)."""
    fails = []
    if row["gate_rmse"] is None or row["gate_rmse"] > 0.50:
        fails.append("RMSE %.3f > 0.50" % (row["gate_rmse"] or float("nan")))
    if (row["n_inliers"] or 0) < 8:
        fails.append("inliers %s < 8" % row["n_inliers"])
    if row["entropy"] is None or row["entropy"] < 0.75:
        fails.append("entropy %.3f < 0.75" % (row["entropy"] or float("nan")))
    if (row["quads"] or 0) < 3:
        fails.append("quadrants %s < 3" % row["quads"])
    return fails


def main():
    rows = []
    for pair, sensor in PAIRS:
        row, err = run_pair(pair, sensor, trace=(pair == "ohrc_04"))
        if err:
            print("%s: %s" % (pair, err))
            rows.append({"pair": pair, "error": err})
            continue
        row["blockers"] = blockers(row)
        rows.append(row)
        hk = row["hook"] or {}
        hi = hk.get("info", {}) if isinstance(hk, dict) else {}
        print("%s [%s]: %s gate=%.4s n=%s ent=%.3s q=%s | stage applied=%s reason=%s "
              "hook_n=%s | sweep_best=%s aff_ho=%s improv=%s req=%s | BLOCKERS=%s"
              % (pair, row["pair_description"], row["status"], row["gate_rmse"], row["n_inliers"],
                 row["entropy"], row["quads"], row["stage_applied"],
                 row["stage_reason"], row["hook_n"],
                 _r(hi.get("lambda_sweep")), _f(hi.get("heldout_rmse_affine_px")),
                 _f(hi.get("improvement_px")), _f(hi.get("improvement_required_px")),
                 row["blockers"]))

    print("\n=== ohrc_04 trace (stage: hook -> balance -> bucket -> ransac -> ncc) ===")
    for name, n_in, n_out, extra in trace_log:
        print("  %-10s in=%6d out=%6s %s" % (name, n_in, n_out, extra or ""))

    # CSV: hook detail flattened
    outp = os.path.expanduser("~/workspace/chandra-align/results/table_blockers.csv")
    with open(outp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "pair_description", "reference_product", "source_product",
                    "status", "gate_rmse", "gate_basis", "n_inliers",
                    "entropy", "quadrants", "qcounts", "stage_applied",
                    "stage_reason", "stage_lambda", "hook_n_inliers",
                    "hook_sweep_best_lambda", "hook_sweep_best_check_px",
                    "hook_affine_check_px", "hook_improvement_px",
                    "hook_improvement_required_px", "hook_max_disp_px",
                    "hook_min_jac_det", "blockers"])
        for r in rows:
            if "error" in r:
                w.writerow([r["pair"], r["error"]] + [""] * 21)
                continue
            hi = (r["hook"] or {}).get("info", {}) or {}
            sweep = hi.get("lambda_sweep") or []
            best = min(sweep, key=lambda s: s[1]) if sweep else (None, None)
            w.writerow([r["pair"], r["pair_description"], r["reference_product"],
                        r["source_product"], r["status"], r["gate_rmse"], r["gate_basis"],
                        r["n_inliers"], r["entropy"], r["quads"], r["qcounts"],
                        r["stage_applied"], r["stage_reason"], r["stage_lambda"],
                        r["hook_n"], best[0], best[1],
                        hi.get("heldout_rmse_affine_px"),
                        hi.get("improvement_px"),
                        hi.get("improvement_required_px"),
                        hi.get("max_displacement_px"), hi.get("min_jacobian_det"),
                        "; ".join(r["blockers"])])
    print("\nwrote", outp)


def _r(sweep):
    if not sweep:
        return None
    b = min(sweep, key=lambda s: s[1])
    return (b[0], round(b[1], 4))


if __name__ == "__main__":
    main()
