"""A28: L1 field-guided match rescue (ohrc_04) + L2 lower-lambda grid (ohrc_05/06).

Experiment only. No app.py / chandra_align/ changes. Uses the true stage
machinery (chandra_align.deform_field.apply_deform_field_stage) and the true
frozen gate (chandra_align.metrics.quadrant.validate_registration_gate).

L1: fit the field on ohrc_04's Q3/Q4 hook inliers (flag-on path), re-score ALL
raw matches under the field, admit field-consistent matches, re-run the FROZEN
gates on the rescued set. Win = SUCCESS_SUBPIXEL under existing thresholds.

L2: extend the stage's lambda grid below 0.1 (0.03, 0.01) via the existing
stage machinery. Win = ohrc_05 crosses 0.50px held-out with zero downgrades.

Deterministic (seed 7). Fail closed. Memory-light (sequential).
"""
import csv
import os
import sys

REPO = os.path.expanduser("~/workspace/chandra-align")
sys.path.insert(0, "/tmp/stubs")
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ["CHANDRA_DEFORM_FIELD"] = "1"

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import app  # noqa: E402
import chandra_align.deform_field as dfmod  # noqa: E402
from chandra_align.metrics.quadrant import (  # noqa: E402
    compute_quadrant_metrics,
    validate_registration_gate,
)

CROP = os.path.join(REPO, "data", "benchmark_crops")
SEED = 7
MIN_INLIERS = 8  # MIN_REGISTRATION_INLIERS

PRODUCTS = {
    "ohrc_01": ("ch2_ohr_nrp_20240425T1209509264_d_img_d18",
                "ch2_ohr_nrp_20240425T1406019344_d_img_d18",
                "south-polar highlands; cratered surface with prominent shadowed crater rims"),
    "ohrc_02": ("ch2_ohr_nrp_20240425T1209509264_d_img_d18",
                "ch2_ohr_nrp_20240425T1406019344_d_img_d18",
                "south-polar highlands; rough cratered relief and shadowed massif/ridge-like slopes"),
    "ohrc_03": ("ch2_ohr_nrp_20240425T1209509264_d_img_d18",
                "ch2_ohr_nrp_20240425T1406019344_d_img_d18",
                "south-polar highlands; densely cratered, hummocky relief"),
    "ohrc_04": ("ch2_ohr_nrp_20240425T1209509264_d_img_d18",
                "ch2_ohr_nrp_20240425T1406019344_d_img_d18",
                "south-polar highlands; inter-crater terrain with clustered small craters"),
    "ohrc_05": ("ch2_ohr_nrp_20240425T1209509264_d_img_d18",
                "ch2_ohr_nrp_20240425T1406019344_d_img_d18",
                "south-polar highlands; mixed crater sizes and low-sun shadows"),
    "ohrc_06": ("ch2_ohr_nrp_20240425T1209509264_d_img_d18",
                "ch2_ohr_nrp_20240425T1406019344_d_img_d18",
                "south-polar highlands; prominent shadowed crater and rugged rim relief"),
}

EXTENDED_GRID = (0.01, 0.03, 0.1, 1.0, 10.0, 100.0)


def load_pair(pair):
    ref = cv2.imread(os.path.join(CROP, "%s_reference.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(os.path.join(CROP, "%s_source.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    if ref is None or sec is None:
        raise RuntimeError("missing crop for %s" % pair)
    return ref, sec


def hook_and_stage(pair):
    """Replicate the pipeline's hook + stage on the raw matches.

    Returns (pr, ps, M_hook, inl_hook, stage_info, ref_shape).
    """
    ref, sec = load_pair(pair)
    pts_ref, pts_sec, _, _ = app.match_pair_hf(ref, sec, 3.0)
    pr = np.asarray(pts_ref, dtype=np.float64)
    ps = np.asarray(pts_sec, dtype=np.float64)
    cv2.setRNGSeed(SEED)
    M_hook, inl = app._estimate_partial_affine_with_threshold(ps, pr, 3.0)
    inl_hook = np.asarray(inl).astype(bool)
    if M_hook is None or int(inl_hook.sum()) < MIN_INLIERS:
        return pr, ps, M_hook, inl_hook, {"applied": False, "reason": "hook_failed"}, ref.shape
    info = dfmod.apply_deform_field_stage(
        ps[inl_hook], pr[inl_hook], M_hook, seed=SEED)
    return pr, ps, M_hook, inl_hook, info, ref.shape


def field_residuals(pr, ps, M_hook, field):
    """|p_ref - (M_hook @ p_src + d(p_src))| for every raw match."""
    d_all = dfmod._eval_residual_field(field, ps)
    Ma = np.asarray(M_hook, dtype=np.float64)
    pred = (Ma[:, :2] @ ps.T).T + Ma[:, 2] + d_all
    return np.linalg.norm(pr - pred, axis=1)


def gate_on_points(pts_ref, residuals_vec, img_shape, stage_heldout_rmse):
    """Run the FROZEN gate on a point set.

    rmse = the stage's internal held-out RMSE (the gate basis when the stage
    applies); entropy/quadrants from compute_quadrant_metrics; the exact
    validate_registration_gate thresholds.
    """
    qm, entropy = compute_quadrant_metrics(pts_ref, residuals_vec, img_shape)
    quad_counts = {"Q1": qm["Q1"]["inlier_count"], "Q2": qm["Q2"]["inlier_count"],
                   "Q3": qm["Q3"]["inlier_count"], "Q4": qm["Q4"]["inlier_count"]}
    msg, code = validate_registration_gate(
        stage_heldout_rmse, len(pts_ref), MIN_INLIERS, entropy, quad_counts)
    return {
        "n": int(len(pts_ref)),
        "rmse_gate_px": float(stage_heldout_rmse),
        "entropy": float(entropy),
        "quadrants": int(sum(1 for c in quad_counts.values() if c > 0)),
        "quad_counts": [quad_counts["Q1"], quad_counts["Q2"],
                        quad_counts["Q3"], quad_counts["Q4"]],
        "status_code": code,
    }


def run_l1(pair="ohrc_04"):
    """Field-guided rescue: admit field-consistent raw matches, re-gate."""
    pr, ps, M_hook, inl_hook, info, shape = hook_and_stage(pair)
    out = {"lever": "L1", "pair": pair,
           "stage_applied": bool(info.get("applied")),
           "stage_reason": info.get("reason"),
           "stage_lambda": info.get("lambda_chosen")}
    if not info.get("applied"):
        out["outcome"] = "stage_declined_no_rescue_possible"
        return out
    field = info["field"]
    fres = field_residuals(pr, ps, M_hook, field)

    results = {}
    for thr, tag in ((3.0, "3px"), (1.0, "1px")):
        adm = fres < thr
        aps, apr = ps[adm], pr[adm]
        # Refit the stage on the admitted set (same machinery, same grid);
        # the gate scores the stage's internal held-out, never in-sample.
        st = dfmod.apply_deform_field_stage(aps, apr, M_hook, seed=SEED)
        if not st.get("applied") or st.get("heldout_rmse_px") is None:
            results[tag] = {"admitted": int(adm.sum()),
                            "outcome": "stage_declined_on_admitted_set",
                            "reason": st.get("reason")}
            continue
        # Field-corrected residuals for the admitted set (for the record;
        # the gate RMSE is the stage held-out).
        d_adm = dfmod._eval_residual_field(st["field"], aps)
        Ma = np.asarray(M_hook, dtype=np.float64)
        pred = (Ma[:, :2] @ aps.T).T + Ma[:, 2] + d_adm
        rvec = apr - pred
        g = gate_on_points(apr, rvec, shape, st["heldout_rmse_px"])
        g["admitted"] = int(adm.sum())
        g["admitted_quad_counts"] = [
            int(((apr[:, 1] < shape[0] / 2) & (apr[:, 0] < shape[1] / 2)).sum()),
            int(((apr[:, 1] < shape[0] / 2) & (apr[:, 0] >= shape[1] / 2)).sum()),
            int(((apr[:, 1] >= shape[0] / 2) & (apr[:, 0] < shape[1] / 2)).sum()),
            int(((apr[:, 1] >= shape[0] / 2) & (apr[:, 0] >= shape[1] / 2)).sum()),
        ]
        g["refit_lambda"] = st.get("lambda_chosen")
        g["refit_min_jac"] = st.get("min_jacobian_det")
        results[tag] = g
    out["rescue"] = results
    # Win condition: the conservative (<1px) set reaches SUCCESS_SUBPIXEL.
    win = results.get("1px", {}).get("status_code") == "SUCCESS_SUBPIXEL"
    out["outcome"] = "WIN" if win else "no_upgrade"
    return out


def run_l2(pair):
    """Lower-lambda grid THROUGH THE TRUE PIPELINE.

    Monkeypatches the stage's lambda grid in the harness only (no file
    changes); app.py imports apply_deform_field_stage at call time, so the
    true _align_core runs end-to-end with the extended grid. The verdict is
    the pipeline's own, under the frozen gates.
    """
    _orig = dfmod.apply_deform_field_stage

    def patched(p_src, p_ref, M, lambda_grid=EXTENDED_GRID, seed=SEED):
        return _orig(p_src, p_ref, M, lambda_grid=lambda_grid, seed=seed)

    dfmod.apply_deform_field_stage = patched
    try:
        ref, sec = load_pair(pair)
        out_core = app._align_core(ref, sec, reference_sensor_name="OHRC")
    finally:
        dfmod.apply_deform_field_stage = _orig
    jm = out_core.get("judge_metrics", {}) or {}
    df = jm.get("deform_field_stage", {}) or {}
    out = {"lever": "L2", "pair": pair,
           "status_code": out_core.get("status_code"),
           "n_inliers": out_core.get("inlier_cnt"),
           "rmse_gate_px": jm.get("rmse_gate_px"),
           "rmse_gate_basis": jm.get("rmse_gate_basis"),
           "df_applied": bool(df.get("applied")),
           "df_reason": df.get("reason"),
           "df_lambda": df.get("lambda_chosen"),
           "df_heldout_px": df.get("heldout_rmse_px"),
           "df_heldout_affine_px": df.get("heldout_rmse_affine_px"),
           "df_min_jac_det": df.get("min_jacobian_det"),
           "df_sweep": df.get("lambda_sweep")}
    # Win = verdict upgrade vs the flag-on baseline in table_quota_12pair.csv.
    out["outcome"] = ("WIN" if out["status_code"] == "SUCCESS_SUBPIXEL"
                      else "no_upgrade")
    return out


def run_control(pair):
    """True flag-on _align_core; must match table_quota_12pair.csv (no downgrades)."""
    ref, sec = load_pair(pair)
    out = app._align_core(ref, sec, reference_sensor_name="OHRC")
    jm = out.get("judge_metrics", {}) or {}
    return {"lever": "control", "pair": pair,
            "status_code": out.get("status_code"),
            "n_inliers": out.get("inlier_cnt"),
            "rmse_gate_px": jm.get("rmse_gate_px")}


def main():
    rows = []

    # L1 on ohrc_04.
    l1 = run_l1("ohrc_04")
    print("L1 ohrc_04:", {k: v for k, v in l1.items() if k != "rescue"})
    for tag, g in l1.get("rescue", {}).items():
        print("  <%s:" % tag, g)
        rows.append({
            "lever": "L1_rescue_%s" % tag, "pair": "ohrc_04",
            "n": g.get("admitted", g.get("n")),
            "rmse_gate_px": g.get("rmse_gate_px"),
            "entropy": g.get("entropy"), "quadrants": g.get("quadrants"),
            "quad_counts": g.get("quad_counts"),
            "lambda": g.get("refit_lambda"),
            "min_jac_det": g.get("refit_min_jac"),
            "status_code": g.get("status_code", g.get("outcome")),
            "outcome": l1["outcome"],
            "reference_product": PRODUCTS["ohrc_04"][0],
            "source_product": PRODUCTS["ohrc_04"][1],
            "terrain": PRODUCTS["ohrc_04"][2],
        })

    # L2 on ohrc_05, ohrc_06 (and ohrc_04 for the sweep record).
    for pair in ("ohrc_05", "ohrc_06", "ohrc_04"):
        l2 = run_l2(pair)
        print("L2 %s:" % pair,
              {k: l2.get(k) for k in ("status_code", "n_inliers",
                                      "rmse_gate_px", "df_lambda",
                                      "df_sweep", "outcome")})
        rows.append({
            "lever": "L2_lambda_grid", "pair": pair,
            "n": l2.get("n_inliers"),
            "rmse_gate_px": l2.get("rmse_gate_px"),
            "entropy": None, "quadrants": None, "quad_counts": None,
            "lambda": l2.get("df_lambda"),
            "min_jac_det": l2.get("df_min_jac_det"),
            "status_code": l2.get("status_code"),
            "outcome": l2["outcome"],
            "lambda_sweep": l2.get("df_sweep"),
            "reference_product": PRODUCTS[pair][0],
            "source_product": PRODUCTS[pair][1],
            "terrain": PRODUCTS[pair][2],
        })

    # Controls: ohrc_01/02/03 flag-on must not downgrade.
    for pair in ("ohrc_01", "ohrc_02", "ohrc_03"):
        c = run_control(pair)
        print("control %s:" % pair, c)
        rows.append({
            "lever": "control", "pair": pair,
            "n": c["n_inliers"], "rmse_gate_px": c["rmse_gate_px"],
            "entropy": None, "quadrants": None, "quad_counts": None,
            "lambda": None, "min_jac_det": None,
            "status_code": c["status_code"], "outcome": "control",
            "reference_product": PRODUCTS[pair][0],
            "source_product": PRODUCTS[pair][1],
            "terrain": PRODUCTS[pair][2],
        })

    # Determinism spot-check: re-run L1.
    l1b = run_l1("ohrc_04")
    det = (l1b.get("rescue", {}).get("1px", {}).get("status_code")
           == l1.get("rescue", {}).get("1px", {}).get("status_code")
           and l1b.get("rescue", {}).get("1px", {}).get("admitted")
           == l1.get("rescue", {}).get("1px", {}).get("admitted"))
    print("determinism L1 re-run identical:", det)

    cols = ["lever", "pair", "n", "rmse_gate_px", "entropy", "quadrants",
            "quad_counts", "lambda", "min_jac_det", "status_code", "outcome",
            "lambda_sweep", "reference_product", "source_product", "terrain"]
    with open(os.path.join(REPO, "results", "table_ohrc_levers.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote results/table_ohrc_levers.csv (%d rows)" % len(rows))
    print("determinism:", det)


if __name__ == "__main__":
    main()
