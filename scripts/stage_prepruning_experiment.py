#!/usr/bin/env python3
"""Worker A16: can the deform-field stage deliver in-pipeline if it runs
BEFORE pruning? (experiment only -- no app.py / chandra_align/ changes)

PREMISE CORRECTION (measured, not assumed): the task brief supposed the true
pipeline's pre-pruning RANSAC set was ~842 inliers (A14's standalone number).
Capturing the ACTUAL set flowing into refine_subpixel_ncc inside _align_core
shows it is 19/34/24/10 points on ohrc_01/02/03/05 -- the quadrant-balance
(50/quadrant) + 8x8 grid bucketing (8/bucket) + RANSAC already starved it,
because RANSAC on a spatially spread set under spatially-varying distortion
keeps only a locally-consistent cluster. So this experiment has two parts:

  Part A (as specified): stage on the pre-NCC RANSAC set (19-34 pts), then
    grid-bucket + NCC-prune the corrected points, then the frozen gate.
  Part B (the reformulation the brief actually meant): stage on the
    UNBUCKETED RANSAC set (~842 pts, A14's replication), i.e. before the
    quadrant-balance + bucketing + NCC pruning chain; then prune the
    corrected points through that same chain; then the frozen gate.

True _align_core order (verified by reading app.py):
  SIFT/Lowe -> quadrant balance (50/q) -> 8x8 bucketing (8/bucket) -> RANSAC
  -> [fallback] -> RICH set -> refine_subpixel_ncc + RANSAC refit (pruning)
  -> gate -> stage hook (opt-in, currently declines on ohrc_01: 19 pts).

Method notes:
- app imported with A15's stubs (/tmp/stubs) for gradio/spaces/matplotlib.
- Part A captures the rich set via wrappers (asserted == last RANSAC inliers).
- Part B reuses A14's proven sift_match/ransac_fit (ported below, same
  assertions on ohrc_01: 5061 matches / 844 inliers / 1.6087 px), then fits
  sec->ref via chandra_align.registration_transform.estimate_partial_affine
  (RANSAC + LSTSQ refit, cv2.setRNGSeed(7) before the call; direction checked
  by RMSE ~1.61 px).
- Corrected points: cref = M @ sec + F(sec); F refit at the stage-chosen
  lambda with the stage module's own _fit_residual_field (cross-checked
  against the stage's returned residuals to 1e-6).
- Pruning mirrors _align_core exactly: select_distributed_matches (8x8, 8),
  refine_subpixel_ncc (window 11, search 1) on the pipeline's captured
  CLAHE-processed images, RANSAC refit at 3 px (cv2.setRNGSeed(7),
  documented), keep iff >= 8 inliers. The refit affine is a PRUNING VEHICLE
  only; the gate scores field-corrected residuals under (M_rich, F).
- Gate basis mirrors the current hook: gate RMSE = the stage's internal
  stride-80/20 field held-out (generalization); inlier count / quadrants /
  entropy from the final pruned set. Frozen thresholds: ACCEPT <=0.50 px /
  >=8 inl / entropy>=0.75 / >=3 quads; COARSE <=2.50 px / >=8 inl /
  entropy>=0.50 / >=2 quads.
"""

import csv
import os
import sys

import numpy as np
import cv2

SEED = 7
REPO = os.path.expanduser("~/workspace/chandra-align")
CROP = os.path.join(REPO, "data/benchmark_crops")
PAIRS = ["ohrc_01", "ohrc_02", "ohrc_03", "ohrc_05"]

sys.path.insert(0, "/tmp/stubs")
sys.path.insert(0, REPO)
os.environ.pop("CHANDRA_DEFORM_FIELD", None)

import app as appmod  # noqa: E402
from chandra_align.deform_field import (  # noqa: E402
    apply_deform_field_stage,
    _fit_residual_field,
    _eval_residual_field,
    _apply_affine,
)
from chandra_align.features.distribution import (  # noqa: E402
    select_distributed_matches,
    select_quadrant_balanced_matches,
)
from chandra_align.metrics.quadrant import (  # noqa: E402
    compute_quadrant_metrics,
    validate_registration_gate,
)
from chandra_align.registration_transform import estimate_partial_affine  # noqa: E402

MIN_INL = int(appmod.MIN_REGISTRATION_INLIERS)
assert MIN_INL == 8, "frozen gate assumes 8"

# --------------------------------------------------------------------------
# Capture rig for Part A: the true pipeline's pre-NCC (post-RANSAC) set.
# --------------------------------------------------------------------------
_ransac_calls = []
_captured = {}
_orig_estimate = appmod._estimate_partial_affine_with_threshold
_orig_ncc = appmod.refine_subpixel_ncc


def _est_wrapper(src_pts, dst_pts, thr):
    M, inl = _orig_estimate(src_pts, dst_pts, thr)
    _ransac_calls.append({
        "M": None if M is None else np.array(M, dtype=np.float64),
        "inliers": np.array(inl, dtype=bool),
    })
    return M, inl


def _ncc_wrapper(img_a, img_b, pts_a, pts_b, ncc_window=11, search_range_px=1):
    _captured["ref_processed"] = np.array(img_a)
    _captured["sec_processed"] = np.array(img_b)
    _captured["rich_ref"] = np.asarray(pts_a, dtype=np.float64)
    _captured["rich_sec"] = np.asarray(pts_b, dtype=np.float64)
    assert _ransac_calls, "NCC called with no preceding RANSAC call"
    last = _ransac_calls[-1]
    assert last["M"] is not None, "last RANSAC produced no matrix"
    assert len(_captured["rich_ref"]) == int(last["inliers"].sum()), (
        "rich set size %d != last RANSAC inliers %d"
        % (len(_captured["rich_ref"]), int(last["inliers"].sum()))
    )
    _captured["M_rich"] = last["M"]
    return _orig_ncc(img_a, img_b, pts_a, pts_b,
                     ncc_window=ncc_window, search_range_px=search_range_px)


appmod._estimate_partial_affine_with_threshold = _est_wrapper
appmod.refine_subpixel_ncc = _ncc_wrapper

_VERDICT_RANK = {"SUCCESS_SUBPIXEL": 2, "COARSE_ADVISORY": 1, "DEGENERATE_FAILURE": 0}


# --------------------------------------------------------------------------
# A14's proven unbucketed matcher (ported verbatim from
# scripts/deformation_field_experiment.py) for Part B.
# --------------------------------------------------------------------------
def sift_match(a, b, nfeatures=8000, contrastThreshold=0.04, edgeThreshold=10):
    sift = cv2.SIFT_create(nfeatures=nfeatures,
                           contrastThreshold=contrastThreshold,
                           edgeThreshold=edgeThreshold)
    k1, d1 = sift.detectAndCompute(a, None)
    k2, d2 = sift.detectAndCompute(b, None)
    raw = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, nn in raw if m.distance < 0.75 * nn.distance]
    p1 = np.float32([k1[m.queryIdx].pt for m in good])  # ref-frame points
    p2 = np.float32([k2[m.trainIdx].pt for m in good])  # sec-frame points
    return p1, p2


def _quad_counts(qm):
    if isinstance(qm, dict):
        return {k: int(v["inlier_count"]) if isinstance(v, dict) else int(v)
                for k, v in qm.items()}
    return str(qm)


def _prune_corrected(ref_proc, sec_proc, cref, csec, shape):
    """Mirror _align_core's pruning chain on corrected points.

    Returns (final_ref, final_sec, pruned_by_refit). The RANSAC refit affine
    is a pruning vehicle only (mirrors the pipeline keeping the set on
    refit failure).
    """
    bref, bsec, _ = select_distributed_matches(
        cref, csec, shape, grid_shape=(8, 8), max_per_bucket=8)
    nref, nsec, _ = _orig_ncc(ref_proc, sec_proc, bref, bsec,
                              ncc_window=11, search_range_px=1)
    final_ref, final_sec = np.asarray(bref), np.asarray(bsec)
    pruned = False
    if len(nref) >= 3:
        cv2.setRNGSeed(SEED)  # documented: standalone refit seeded
        M2, inl2 = _orig_estimate(np.asarray(nsec), np.asarray(nref), 3.0)
        inl2 = np.asarray(inl2, dtype=bool)
        if M2 is not None and int(inl2.sum()) >= MIN_INL:
            final_ref, final_sec = np.asarray(nref)[inl2], np.asarray(nsec)[inl2]
            pruned = True
    return final_ref, final_sec, pruned


def _gate_on_field(info, M, field, final_sec, final_ref, shape, lam):
    """Frozen gate on field-corrected residuals of the final pruned set."""
    d_final = _eval_residual_field(field, final_sec)
    r_final = final_ref - (_apply_affine(M, final_sec) + d_final)
    rmse_inlier = (float(np.sqrt(np.mean(np.sum(r_final ** 2, axis=1))))
                   if len(r_final) else float("inf"))
    qm, ent = compute_quadrant_metrics(final_ref, r_final, shape)
    gate_rmse = float(info["heldout_rmse_px"])  # honest basis: field generalization
    msg, code = validate_registration_gate(gate_rmse, len(final_ref), MIN_INL, ent, qm)
    return {
        "verdict": code, "n_inliers": int(len(final_ref)),
        "inlier_rmse": rmse_inlier, "gate_rmse": gate_rmse,
        "gate_basis": "field held-out (stage set, stride 80/20)",
        "entropy": float(ent), "quadrants": _quad_counts(qm),
    }


def _run_stage_on_set(p_sec, p_ref, M):
    """Run the real stage module; reconstruct the identical field if applied."""
    info = apply_deform_field_stage(p_sec, p_ref, M)  # default grid, seed 7
    if not info.get("applied"):
        return info, None, None
    lam = float(info["lambda_chosen"])
    r_aff = np.asarray(p_ref) - _apply_affine(np.asarray(M), np.asarray(p_sec))
    field = _fit_residual_field(np.asarray(p_sec), r_aff, lam)
    d_all = _eval_residual_field(field, np.asarray(p_sec))
    r_corr_stage = np.asarray(info["residuals_vec"], dtype=np.float64)
    assert np.allclose(r_aff - d_all, r_corr_stage, atol=1e-6), \
        "reconstructed field disagrees with stage residuals"
    return info, field, lam


def _old_order(name):
    ref = cv2.imread(os.path.join(CROP, f"{name}_reference.png"), cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(os.path.join(CROP, f"{name}_source.png"), cv2.IMREAD_GRAYSCALE)
    assert ref is not None and sec is not None, f"missing crops for {name}"
    out = {}
    for flag in ("off", "on"):
        _ransac_calls.clear()
        _captured.clear()
        if flag == "on":
            os.environ["CHANDRA_DEFORM_FIELD"] = "1"
        else:
            os.environ.pop("CHANDRA_DEFORM_FIELD", None)
        res = appmod._align_core(ref, sec)
        jm = res["judge_metrics"]
        df = jm.get("deform_field_stage", {})
        out[flag] = {
            "verdict": res.get("status_code"),
            "n_inliers": int(res.get("inlier_cnt", 0)),
            "inlier_rmse": float(jm.get("rmse_in_sample_px", float("nan"))),
            "gate_rmse": (None if jm.get("rmse_gate_px") in (None, "UNMEASURED")
                          else float(jm.get("rmse_gate_px"))),
            "gate_basis": str(jm.get("rmse_gate_basis", "")),
            "stage_applied": bool(df.get("applied", False)),
            "stage_reason": str(df.get("reason")),
        }
    os.environ.pop("CHANDRA_DEFORM_FIELD", None)
    rich = {
        "n_rich": len(_captured.get("rich_ref", [])),
        "M_rich": _captured.get("M_rich"),
        "rich_ref": _captured.get("rich_ref"),
        "rich_sec": _captured.get("rich_sec"),
        "ref_processed": _captured.get("ref_processed"),
        "sec_processed": _captured.get("sec_processed"),
        "ref_raw": ref,
        "sec_raw": sec,
    }
    return out["off"], out["on"], rich


def _new_order_A(name, rich):
    """Part A: stage on the pre-NCC (post-RANSAC, post-bucketing) set."""
    row = {"stage_applied": False, "stage_reason": None, "n_rich": int(rich["n_rich"])}
    if rich["n_rich"] == 0 or rich["M_rich"] is None:
        row["stage_reason"] = "no_rich_set_captured"
        return row
    rsec = np.asarray(rich["rich_sec"], dtype=np.float64)
    rref = np.asarray(rich["rich_ref"], dtype=np.float64)
    M = np.asarray(rich["M_rich"], dtype=np.float64)
    shape = tuple(int(v) for v in rich["ref_processed"].shape[:2])
    info, field, lam = _run_stage_on_set(rsec, rref, M)
    row["stage_applied"] = bool(info.get("applied"))
    row["stage_reason"] = str(info.get("reason"))
    if not info.get("applied"):
        return row
    row["lambda"] = lam
    row["min_jac_det"] = float(info["min_jacobian_det"])
    row["max_disp"] = float(info["max_displacement_px"])
    r_corr = np.asarray(info["residuals_vec"], dtype=np.float64)
    cref = rref - r_corr
    final_ref, final_sec, pruned = _prune_corrected(
        rich["ref_processed"], rich["sec_processed"], cref, rsec, shape)
    row["pruned_by_refit"] = pruned
    row.update(_gate_on_field(info, M, field, final_sec, final_ref, shape, lam))
    return row


def _new_order_B(name, rich):
    """Part B: stage on the UNBUCKETED RANSAC set (~842), then prune corrected
    points through quadrant-balance + bucketing + NCC-refit, then gate."""
    row = {"stage_applied": False, "stage_reason": None}
    ref, sec = rich["ref_raw"], rich["sec_raw"]
    shape = tuple(int(v) for v in ref.shape[:2])
    p1b, p2b = sift_match(ref, sec)  # p1b: ref pts, p2b: sec pts (A14 convention)
    if name == "ohrc_01":
        assert len(p1b) == 5061, f"match count changed: {len(p1b)}"
    cv2.setRNGSeed(SEED)
    # Module convention: p_src=sec -> p_ref=ref.
    M84, m84 = estimate_partial_affine(p2b, p1b, 3.0)
    m84 = np.asarray(m84, dtype=bool)
    row["n_unbucketed"] = int(m84.sum())
    if M84 is None or int(m84.sum()) < 100:
        row["stage_reason"] = "unbucketed_ransac_failed"
        return row
    rmse_dir = float(np.sqrt(np.mean(np.sum(
        (p1b[m84] - _apply_affine(np.asarray(M84), p2b[m84])) ** 2, axis=1))))
    row["unbucketed_rmse"] = rmse_dir
    if name == "ohrc_01":
        assert 800 <= int(m84.sum()) <= 900, f"inlier count changed: {m84.sum()}"
        assert abs(rmse_dir - 1.6087) < 0.05, f"direction/RMSE wrong: {rmse_dir}"
    s84, r84 = p2b[m84].astype(np.float64), p1b[m84].astype(np.float64)
    info, field, lam = _run_stage_on_set(s84, r84, np.asarray(M84, dtype=np.float64))
    row["stage_applied"] = bool(info.get("applied"))
    row["stage_reason"] = str(info.get("reason"))
    if not info.get("applied"):
        return row
    row["lambda"] = lam
    row["min_jac_det"] = float(info["min_jacobian_det"])
    row["max_disp"] = float(info["max_displacement_px"])
    r_corr = np.asarray(info["residuals_vec"], dtype=np.float64)
    cref = r84 - r_corr
    # Quadrant balance + bucketing on corrected points (mirror pipeline caps).
    qref, qsec, _ = select_quadrant_balanced_matches(
        cref, s84, shape, quota_per_quadrant=50)
    final_ref, final_sec, pruned = _prune_corrected(
        rich["ref_processed"], rich["sec_processed"], qref, qsec, shape)
    row["pruned_by_refit"] = pruned
    row["n_quadrant_balanced"] = int(len(qref))
    row.update(_gate_on_field(info, np.asarray(M84, dtype=np.float64),
                              field, final_sec, final_ref, shape, lam))
    return row


def main():
    rows = []
    print("== Step 1: reproducing A15 true-path numbers ==", flush=True)
    checks = {
        "ohrc_01": {"n": 19, "inlier": 1.3920, "gate": 1.6445},
        "ohrc_02": {"n": 16},
        "ohrc_03": {"n": 14},
        "ohrc_05": {"n": 10},
    }
    old_orders = {}
    for name in PAIRS:
        off, on, rich = _old_order(name)
        old_orders[name] = (off, on, rich)
        exp = checks[name]
        ok = (off["n_inliers"] == exp["n"]
              and abs(off["inlier_rmse"] - exp.get("inlier", off["inlier_rmse"])) < 0.01
              and abs((off["gate_rmse"] or 0) - exp.get("gate", off["gate_rmse"] or 0)) < 0.01)
        print(f"  {name}: off {off['verdict']} n={off['n_inliers']} "
              f"inlier={off['inlier_rmse']:.4f} gate={off['gate_rmse']:.4f} | "
              f"on: {on['verdict']} stage={on['stage_reason']} | "
              f"PRE-NCC rich n={rich['n_rich']}  reproduce={'OK' if ok else 'FAIL'}",
              flush=True)
        for order, o in (("old_off", off), ("old_on", on)):
            rows.append({"pair": name, "order": order, "verdict": o["verdict"],
                         "n_inliers": o["n_inliers"], "inlier_rmse": o["inlier_rmse"],
                         "gate_rmse": o["gate_rmse"], "gate_basis": o["gate_basis"],
                         "stage_applied": o["stage_applied"],
                         "stage_reason": o["stage_reason"], "n_rich": rich["n_rich"]})
        assert ok, f"REPRODUCTION FAILED on {name}; STOPPING"
    assert old_orders["ohrc_01"][1]["stage_reason"] == "no_improvement"
    assert old_orders["ohrc_01"][1]["verdict"] == old_orders["ohrc_01"][0]["verdict"]
    print("  reproduction OK", flush=True)

    print("== Step 2: Part A -- stage on pre-NCC set, then prune corrected ==", flush=True)
    for name in PAIRS:
        off, on, rich = old_orders[name]
        new = _new_order_A(name, rich)
        _emit(name, "new_A", new, off, rows)

    print("== Step 3: Part B -- stage on UNBUCKETED set, then full prune chain ==", flush=True)
    for name in PAIRS:
        off, on, rich = old_orders[name]
        new = _new_order_B(name, rich)
        _emit(name, "new_B", new, off, rows)

    print("== Step 4: determinism re-check (ohrc_01, both new orders) ==", flush=True)
    off, on, rich = old_orders["ohrc_01"]
    for fn, tag in ((_new_order_A, "A"), (_new_order_B, "B")):
        a, b = fn("ohrc_01", rich), fn("ohrc_01", rich)
        assert a.get("verdict") == b.get("verdict"), f"Part {tag} verdict not deterministic"
        assert abs(a.get("inlier_rmse", 0) - b.get("inlier_rmse", 0)) < 1e-9, \
            f"Part {tag} RMSE not deterministic"
        print(f"  Part {tag}: identical", flush=True)

    print("== Step 5: pre-registered clauses ==", flush=True)
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair"], {})[r["order"]] = r
    for tag in ("new_A", "new_B"):
        dg = [n for n in PAIRS
              if _VERDICT_RANK[by_pair[n][tag]["verdict"]]
              < _VERDICT_RANK[by_pair[n]["old_off"]["verdict"]]]
        sp = [n for n in PAIRS if by_pair[n][tag]["verdict"] == "SUCCESS_SUBPIXEL"]
        jf = all((by_pair[n][tag].get("min_jac_det") or 1.0) >= 0.5 for n in PAIRS)
        print(f"  Part {tag}: downgrades={dg or 'none'} | SUBPIXEL={sp or 'none'} | "
              f"no-folding={'HOLD' if jf else 'FAIL'}", flush=True)

    cols = ["pair", "order", "verdict", "n_inliers", "inlier_rmse", "gate_rmse",
            "gate_basis", "stage_applied", "stage_reason", "lambda",
            "min_jac_det", "max_disp", "n_rich"]
    with open(os.path.join(REPO, "results/table_stage_prepruning.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote results/table_stage_prepruning.csv", flush=True)


def _emit(name, order, new, off, rows):
    if new.get("stage_applied"):
        print(f"  {name} [{order}]: APPLIED λ={new['lambda']} jac={new['min_jac_det']:.3f} "
              f"-> {new['verdict']} n={new['n_inliers']} inlier={new['inlier_rmse']:.4f} "
              f"gate={new['gate_rmse']:.4f} quads={new['quadrants']}", flush=True)
    else:
        print(f"  {name} [{order}]: declined ({new['stage_reason']})", flush=True)
    rows.append({"pair": name, "order": order, "n_rich": new.get("n_rich"),
                 "verdict": new.get("verdict", off["verdict"]),
                 "n_inliers": new.get("n_inliers", off["n_inliers"]),
                 "inlier_rmse": new.get("inlier_rmse", off["inlier_rmse"]),
                 "gate_rmse": new.get("gate_rmse", off["gate_rmse"]),
                 "gate_basis": new.get("gate_basis", off["gate_basis"]),
                 "stage_applied": new["stage_applied"],
                 "stage_reason": new["stage_reason"],
                 "lambda": new.get("lambda"),
                 "min_jac_det": new.get("min_jac_det"),
                 "max_disp": new.get("max_disp")})


if __name__ == "__main__":
    main()
