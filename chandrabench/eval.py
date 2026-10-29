#!/usr/bin/env python3
"""ChandraBench v0.1 — one-command evaluator.

Scores a participant's 2x3 affine transform (source -> reference pixel
coordinates, the cv2.warpAffine convention) against the frozen human-verified
landmark sets.

Usage:
    python eval.py --pair ohrc_pair --matrix "1,0,0,0,1,0"
    python eval.py --pair tmc2_fore_nadir --matrix-file my_transform.csv
    python eval.py --self-test

The matrix is 6 numbers in row-major order: [a b c d e f] meaning
    x_ref = a*x_src + b*y_src + c
    y_ref = d*x_src + e*y_src + f

A --matrix-file may hold the same 6 numbers as CSV, or a 2x3 / 3x3 matrix.

Metric (fixed for v0.1):
    forward RMSE  = sqrt(mean(|| M @ p_src - p_ref ||^2))   over all GT points
    inverse RMSE  = sqrt(mean(|| M^-1 @ p_ref - p_src ||^2))
Both are reported; neither is hidden. The headline score is forward RMSE.

Phase 2 extension (additive, opt-in via --extended or evaluate(extended=True)):
    pairs that meet the Phase 2 per-pair landmark bar additionally report a
    bootstrap 95% CI on the forward RMSE, the per-landmark error distribution
    (median / p90 / max), and a repeat-noise estimate from repeated clicks of
    the same feature. Pairs that fail the bar report ABSTAIN with the failing
    reasons and no metrics. The v0.1 metric and default report are unchanged.

Circularity guard (fail-closed):
    * The landmark CSVs are frozen at v0.1. Their sha256 is pinned below; a
      tampered or substituted GT file aborts the run.
    * GT points must never be used to fit the submitted transform. This
      evaluator cannot see your fitting process, so compliance is by
      declaration: --i-declare-gt-not-used-for-fitting is required. The flag
      is recorded in the output report.
    * GT quadrant coverage is checked and reported (protocol floor: >=3 per
      quadrant); the run is marked accordingly.

Only dependency: numpy.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

PAIRS = {
    "ohrc_pair": {
        "csv": "landmarks_ohrc_pair_v0.1.csv",
        "sha256": "8de1613a70181dcfc286bea9d0168dc718de8579323483ef5c30f615cb8f2fa5",
        "n_points": 20,
    },
    "tmc2_fore_nadir": {
        "csv": "landmarks_tmc2_fore_nadir_v0.1.csv",
        "sha256": "254d9d2c24b43d947dc15f3e9c1745ea238ac8b80f9d4774d8b0abf70a9a79c0",
        "n_points": 20,
    },
}

VERSION = "0.1"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_landmarks(pair):
    spec = PAIRS[pair]
    path = HERE / spec["csv"]
    if not path.exists():
        sys.exit(f"error: landmark file missing: {path}")
    digest = sha256_of(path)
    if digest != spec["sha256"]:
        sys.exit(
            "CIRCULARITY_GUARD_FAIL: landmark file sha256 mismatch.\n"
            f"  expected: {spec['sha256']}\n"
            f"  got:      {digest}\n"
            "The v0.1 landmark set is frozen; do not modify it."
        )
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != spec["n_points"]:
        sys.exit(f"CIRCULARITY_GUARD_FAIL: expected {spec['n_points']} rows, got {len(rows)}")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        sys.exit("CIRCULARITY_GUARD_FAIL: duplicate landmark ids")
    ref = np.array([[float(r["x_ref"]), float(r["y_ref"])] for r in rows])
    src = np.array([[float(r["x_src"]), float(r["y_src"])] for r in rows])
    if not (np.isfinite(ref).all() and np.isfinite(src).all()):
        sys.exit("CIRCULARITY_GUARD_FAIL: non-finite landmark coordinates")
    return ids, ref, src, digest


def quadrant_counts(ref):
    """Quadrant split on the reference-point bounding box. Returns dict + pass flag."""
    x0, x1 = ref[:, 0].min(), ref[:, 0].max()
    y0, y1 = ref[:, 1].min(), ref[:, 1].max()
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    lt = int(((ref[:, 0] < mx) & (ref[:, 1] < my)).sum())
    rt = int(((ref[:, 0] >= mx) & (ref[:, 1] < my)).sum())
    lb = int(((ref[:, 0] < mx) & (ref[:, 1] >= my)).sum())
    rb = int(((ref[:, 0] >= mx) & (ref[:, 1] >= my)).sum())
    counts = {"LT": lt, "RT": rt, "LB": lb, "RB": rb}
    return counts, all(v >= 3 for v in counts.values())


def parse_matrix(matrix_str=None, matrix_file=None):
    if matrix_file:
        vals = []
        with open(matrix_file, "r", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                for cell in row:
                    cell = cell.strip()
                    if cell:
                        vals.append(float(cell))
    else:
        vals = [float(v.strip()) for v in matrix_str.split(",") if v.strip()]
    if len(vals) == 6:
        return np.array(vals, dtype=float).reshape(2, 3)
    if len(vals) == 9:
        M33 = np.array(vals, dtype=float).reshape(3, 3)
        return M33[:2, :]
    sys.exit(f"error: expected 6 (2x3) or 9 (3x3) numbers, got {len(vals)}")


def score(M, ids, ref, src):
    M = np.asarray(M, dtype=float).reshape(2, 3)
    fwd_pred = src @ M[:, :2].T + M[:, 2]
    fwd_res = np.hypot(fwd_pred[:, 0] - ref[:, 0], fwd_pred[:, 1] - ref[:, 1])
    M33 = np.vstack([M, [0.0, 0.0, 1.0]])
    try:
        Minv = np.linalg.inv(M33)[:2, :]
    except np.linalg.LinAlgError:
        sys.exit("error: submitted transform is singular; cannot compute inverse RMSE")
    inv_pred = ref @ Minv[:, :2].T + Minv[:, 2]
    inv_res = np.hypot(inv_pred[:, 0] - src[:, 0], inv_pred[:, 1] - src[:, 1])
    return {
        "rmse_forward_px": float(np.sqrt((fwd_res ** 2).mean())),
        "rmse_inverse_px": float(np.sqrt((inv_res ** 2).mean())),
        "max_fwd_res_px": float(fwd_res.max()),
        "per_point": [
            {"id": i, "fwd_res_px": float(fr), "inv_res_px": float(ir)}
            for i, fr, ir in zip(ids, fwd_res, inv_res)
        ],
    }


# ---------------------------------------------------------------------------
# Phase 2 extended evaluation (additive, opt-in).
#
# The v0.1 metric above is frozen; nothing below changes it or the default
# report. The Phase 2 per-pair landmark bar is: >= 25 landmarks, >= 4 per
# quadrant (reference-frame bounding-box split, same convention as
# quadrant_counts), >= 50 px minimum spacing between landmarks (reference
# frame), and non-collinear landmarks. A pair that meets the bar additionally
# reports a bootstrap confidence interval on the forward RMSE, the
# per-landmark error distribution, and a repeat-noise estimate. A pair that
# fails the bar reports status ABSTAIN with the failing reasons and no
# metrics at all: an abstention never carries numbers borrowed from another
# pair or another run.
# ---------------------------------------------------------------------------

PHASE2_MIN_LANDMARKS = 25
PHASE2_MIN_PER_QUADRANT = 4
PHASE2_MIN_SPACING_PX = 50.0
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261029


def bootstrap_rmse_ci(residuals, seed=BOOTSTRAP_SEED, n_resamples=BOOTSTRAP_RESAMPLES):
    """Percentile bootstrap 95% CI for the RMSE of per-landmark residuals.

    Resamples the residuals with replacement (>= 1000 resamples), recomputes
    the RMSE each time, and returns the 2.5 / 97.5 percentiles. The generator
    is seeded, so the same residuals always yield the same interval.
    """
    res = np.asarray(residuals, dtype=float)
    if res.ndim != 1 or len(res) == 0:
        raise ValueError("bootstrap_rmse_ci needs a non-empty 1-D residual array")
    if not np.isfinite(res).all():
        raise ValueError("bootstrap_rmse_ci: non-finite residual")
    if n_resamples < 1000:
        raise ValueError("bootstrap_rmse_ci: n_resamples must be >= 1000")
    rng = np.random.default_rng(seed)
    sq = res ** 2
    idx = rng.integers(0, len(res), size=(n_resamples, len(res)))
    rmses = np.sqrt(sq[idx].mean(axis=1))
    lo, hi = np.percentile(rmses, [2.5, 97.5])
    return float(lo), float(hi)


def error_distribution(residuals):
    """Median / p90 / max of per-landmark radial errors (px)."""
    res = np.asarray(residuals, dtype=float)
    if res.ndim != 1 or len(res) == 0:
        raise ValueError("error_distribution needs a non-empty 1-D residual array")
    if not np.isfinite(res).all():
        raise ValueError("error_distribution: non-finite residual")
    return {
        "error_median_px": float(np.median(res)),
        "error_p90_px": float(np.percentile(res, 90)),
        "error_max_px": float(res.max()),
    }


def repeat_groups(ids, repeat_of):
    """Group row indices by feature from a parallel repeat_of column.

    A row's group key is its repeat_of value when non-empty, else its own
    id — so a repeat click whose repeat_of names the original landmark's id
    lands in the original's group. Only groups with >= 2 rows are returned.
    """
    groups = {}
    for i, lid in enumerate(ids):
        key = None
        if repeat_of is not None:
            val = repeat_of[i]
            if val is not None and str(val).strip():
                key = str(val).strip()
        groups.setdefault(key if key is not None else str(lid), []).append(i)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def repeat_noise(ids, ref, src, repeat_of):
    """Click scatter (px) across repeated clicks of the same feature.

    For every repeat group, each click contributes its distance to the group
    centroid, in the reference frame and in the source frame; the reported
    figure is the RMS of all those distances. When the pair has no repeat
    group, the value is None with the reason — never a borrowed number.
    """
    empty = {"repeat_noise_px": None, "n_repeat_groups": 0, "n_repeat_clicks": 0}
    if repeat_of is None:
        return {**empty, "reason": "no repeat_of data for this pair"}
    groups = repeat_groups(ids, repeat_of)
    if not groups:
        return {**empty,
                "reason": "no repeated landmarks (no repeat_of group with >= 2 clicks)"}
    dists = []
    for idxs in groups.values():
        for pts in (ref[idxs], src[idxs]):
            centroid = pts.mean(axis=0)
            dists.extend(np.hypot(pts[:, 0] - centroid[0],
                                  pts[:, 1] - centroid[1]).tolist())
    dists = np.asarray(dists, dtype=float)
    return {
        "repeat_noise_px": float(np.sqrt((dists ** 2).mean())),
        "n_repeat_groups": len(groups),
        "n_repeat_clicks": int(sum(len(v) for v in groups.values())),
        "reason": None,
    }


def phase2_bar(ids, ref, src):
    """Check the Phase 2 per-pair landmark bar.

    Returns (passed, reasons, detail). Spacing and collinearity are judged in
    the reference frame, the frame the benchmark scores in.
    """
    reasons = []
    n = len(ids)
    if n < PHASE2_MIN_LANDMARKS:
        reasons.append(f"landmark count {n} < {PHASE2_MIN_LANDMARKS}")
    if n:
        qcounts, _ = quadrant_counts(ref)
    else:
        qcounts = {"LT": 0, "RT": 0, "LB": 0, "RB": 0}
    low = [k for k, v in qcounts.items() if v < PHASE2_MIN_PER_QUADRANT]
    if low:
        reasons.append("quadrant counts below "
                       f"{PHASE2_MIN_PER_QUADRANT}: "
                       + ", ".join(f"{k}={qcounts[k]}" for k in low))
    min_spacing = None
    if n >= 2:
        diff = ref[:, None, :] - ref[None, :, :]
        dist = np.hypot(diff[..., 0], diff[..., 1])
        min_spacing = float(dist[np.triu_indices(n, k=1)].min())
        if min_spacing < PHASE2_MIN_SPACING_PX:
            reasons.append(f"minimum landmark spacing {min_spacing:.1f} px "
                           f"< {PHASE2_MIN_SPACING_PX:.1f} px (reference frame)")
    collinear = None
    if n >= 3:
        sv = np.linalg.svd(ref - ref.mean(axis=0), compute_uv=False)
        collinear = bool(sv[1] <= 1e-6 * sv[0]) if sv[0] > 0 else True
        if collinear:
            reasons.append("landmarks are collinear (reference frame)")
    detail = {
        "n_landmarks": n,
        "quadrant_counts_ref": qcounts,
        "min_spacing_ref_px": min_spacing,
        "collinear_ref": collinear,
    }
    return not reasons, reasons, detail


def extended_block(ids, ref, src, fwd_residuals, repeat_of=None,
                   seed=BOOTSTRAP_SEED, n_resamples=BOOTSTRAP_RESAMPLES):
    """Phase 2 extended report block for one pair.

    Status OK pairs carry metrics; ABSTAIN pairs carry only the bar detail
    and the failing reasons, with metrics set to None.
    """
    res = np.asarray(fwd_residuals, dtype=float)
    if len(res) != len(ids):
        raise ValueError("extended_block: residuals and landmarks length mismatch")
    if not np.isfinite(res).all():
        raise ValueError("extended_block: non-finite residual")
    passed, reasons, detail = phase2_bar(ids, ref, src)
    block = {
        "bar": {
            "min_landmarks": PHASE2_MIN_LANDMARKS,
            "min_per_quadrant": PHASE2_MIN_PER_QUADRANT,
            "min_spacing_px": PHASE2_MIN_SPACING_PX,
            "non_collinear_required": True,
        },
        **detail,
        "status": "OK" if passed else "ABSTAIN",
        "abstain_reasons": reasons,
        "metrics": None,
    }
    if passed:
        lo, hi = bootstrap_rmse_ci(res, seed=seed, n_resamples=n_resamples)
        block["metrics"] = {
            "rmse_forward_px": float(np.sqrt((res ** 2).mean())),
            "rmse_ci95_px": [lo, hi],
            "bootstrap": {"method": "percentile",
                          "n_resamples": n_resamples, "seed": seed},
            **error_distribution(res),
            "repeat_noise": repeat_noise(ids, ref, src, repeat_of),
        }
    return block


def load_landmark_csv(path):
    """Load a landmark CSV for a pair that is not in the frozen v0.1 registry.

    Expects the clicker / results-table columns id,x_ref,y_ref,x_src,y_src
    and accepts an optional repeat_of column. Same fail-closed validation as
    load_landmarks, minus the sha pin: the file is not frozen yet.
    Returns (ids, ref, src, repeat_of) with repeat_of None when absent.
    """
    path = Path(path)
    if not path.exists():
        sys.exit(f"error: landmark file missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"error: landmark file has no rows: {path}")
    required = {"id", "x_ref", "y_ref", "x_src", "y_src"}
    missing = required - set(rows[0].keys())
    if missing:
        sys.exit(f"error: landmark file missing columns {sorted(missing)}: {path}")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        sys.exit("error: duplicate landmark ids")
    ref = np.array([[float(r["x_ref"]), float(r["y_ref"])] for r in rows])
    src = np.array([[float(r["x_src"]), float(r["y_src"])] for r in rows])
    if not (np.isfinite(ref).all() and np.isfinite(src).all()):
        sys.exit("error: non-finite landmark coordinates")
    repeat_of = [r["repeat_of"] for r in rows] if "repeat_of" in rows[0] else None
    return ids, ref, src, repeat_of


def evaluate_table(ids, ref, src, M, repeat_of=None,
                   seed=BOOTSTRAP_SEED, n_resamples=BOOTSTRAP_RESAMPLES):
    """Score an arbitrary landmark table and attach the Phase 2 block.

    This is the entry point for Phase 2 pairs that are not (yet) in the
    frozen v0.1 registry: same scoring as score(), plus phase2_extended.
    """
    s = score(M, ids, ref, src)
    fwd = np.array([p["fwd_res_px"] for p in s["per_point"]])
    return {**s, "phase2_extended": extended_block(
        ids, ref, src, fwd, repeat_of=repeat_of, seed=seed,
        n_resamples=n_resamples)}


def _repeat_of_for_pair(pair):
    """repeat_of column of a registered pair's CSV, or None when absent."""
    path = HERE / PAIRS[pair]["csv"]
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if rows and "repeat_of" in rows[0]:
        return [r["repeat_of"] for r in rows]
    return None


def evaluate(pair, M, declare_flag, extended=False):
    ids, ref, src, digest = load_landmarks(pair)
    qcounts, qpass = quadrant_counts(ref)
    s = score(M, ids, ref, src)
    report = {
        "chandrabench_version": VERSION,
        "pair": pair,
        "n_gt_points": len(ids),
        "gt_sha256": digest,
        "gt_quadrant_counts_ref": qcounts,
        "gt_quadrant_floor_pass": qpass,
        "gt_not_used_for_fitting_declared": bool(declare_flag),
        "submitted_matrix_2x3": np.asarray(M, dtype=float).reshape(2, 3).tolist(),
        **s,
    }
    if extended:
        fwd = np.array([p["fwd_res_px"] for p in s["per_point"]])
        report["phase2_extended"] = extended_block(
            ids, ref, src, fwd, repeat_of=_repeat_of_for_pair(pair))
    return report


def self_test():
    """Self-consistency: identity submission and a GT self-fit submission."""
    print("ChandraBench v0.1 self-test")
    ok = True
    for pair in PAIRS:
        ids, ref, src, digest = load_landmarks(pair)
        # 1. Identity: RMSE must equal the raw src-ref disparity, computed two ways.
        r_id = evaluate(pair, np.array([1, 0, 0, 0, 1, 0], float), True)
        direct = float(np.sqrt((np.hypot(src[:, 0] - ref[:, 0],
                                         src[:, 1] - ref[:, 1]) ** 2).mean()))
        c1 = abs(r_id["rmse_forward_px"] - direct) < 1e-9
        # 2. Least-squares self-fit of the GT points must substantially beat
        #    identity. (A perfect fit is impossible: the landmarks carry real
        #    non-affine signal — parallax between the two views — plus ~1px
        #    NCC noise. This is exactly why the benchmark is comparative.)
        A = np.hstack([src, np.ones((len(src), 1))])
        Mx, *_ = np.linalg.lstsq(A, ref[:, 0], rcond=None)
        My, *_ = np.linalg.lstsq(A, ref[:, 1], rcond=None)
        Mfit = np.array([[Mx[0], Mx[1], Mx[2]], [My[0], My[1], My[2]]])
        r_fit = evaluate(pair, Mfit, True)
        # 2b. Independent brute-force recomputation of the same RMSE.
        px = Mfit[0, 0] * src[:, 0] + Mfit[0, 1] * src[:, 1] + Mfit[0, 2]
        py = Mfit[1, 0] * src[:, 0] + Mfit[1, 1] * src[:, 1] + Mfit[1, 2]
        brute = float(np.sqrt(np.mean((px - ref[:, 0]) ** 2 + (py - ref[:, 1]) ** 2)))
        c2 = (r_fit["rmse_forward_px"] < 0.5 * r_id["rmse_forward_px"]
              and abs(r_fit["rmse_forward_px"] - brute) < 1e-9)
        c3 = r_id["gt_quadrant_floor_pass"]
        print(f"  {pair}: identity_rmse={r_id['rmse_forward_px']:.4f}px "
              f"(independent check {direct:.4f}px) "
              f"self_fit_rmse={r_fit['rmse_forward_px']:.2e}px "
              f"quadrants={r_id['gt_quadrant_counts_ref']}")
        if not (c1 and c2 and c3):
            ok = False
            print(f"    FAIL: c1={c1} c2={c2} c3={c3}")
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="ChandraBench v0.1 evaluator")
    ap.add_argument("--pair", choices=list(PAIRS), help="benchmark pair")
    ap.add_argument("--matrix", help='6 comma-separated numbers "a,b,c,d,e,f"')
    ap.add_argument("--matrix-file", help="CSV file with 6 (2x3) or 9 (3x3) numbers")
    ap.add_argument("--i-declare-gt-not-used-for-fitting", action="store_true",
                    help="required: declare the GT landmarks were not used to fit the transform")
    ap.add_argument("--json", help="write the full score report to this JSON path")
    ap.add_argument("--extended", action="store_true",
                    help="add the Phase 2 extended block (bootstrap CI, error "
                         "distribution, repeat noise, abstain status)")
    ap.add_argument("--self-test", action="store_true", help="run the self-consistency checks")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.pair or not (args.matrix or args.matrix_file):
        ap.error("--pair and one of --matrix / --matrix-file are required")
    if not args.i_declare_gt_not_used_for_fitting:
        sys.exit("CIRCULARITY_GUARD_FAIL: re-run with "
                 "--i-declare-gt-not-used-for-fitting to declare the GT set "
                 "was not used to fit the submitted transform.")

    M = parse_matrix(args.matrix, args.matrix_file)
    report = evaluate(args.pair, M, True, extended=args.extended)

    print(f"ChandraBench v{VERSION} — {args.pair}")
    print(f"  GT points: {report['n_gt_points']}  sha256: {report['gt_sha256'][:16]}...")
    print(f"  GT quadrants (ref): {report['gt_quadrant_counts_ref']} "
          f"floor_pass={report['gt_quadrant_floor_pass']}")
    print(f"  RMSE forward : {report['rmse_forward_px']:.4f} px   <-- headline score")
    print(f"  RMSE inverse : {report['rmse_inverse_px']:.4f} px")
    print(f"  Max fwd residual: {report['max_fwd_res_px']:.4f} px")
    if "phase2_extended" in report:
        ext = report["phase2_extended"]
        print(f"  Phase 2 extended: status={ext['status']} "
              f"(n={ext['n_landmarks']}, quadrants={ext['quadrant_counts_ref']}, "
              f"min_spacing_ref={ext['min_spacing_ref_px']}, "
              f"collinear_ref={ext['collinear_ref']})")
        if ext["status"] == "ABSTAIN":
            for reason in ext["abstain_reasons"]:
                print(f"    abstain: {reason}")
        else:
            m = ext["metrics"]
            print(f"    RMSE 95% CI (bootstrap, n={m['bootstrap']['n_resamples']}, "
                  f"seed={m['bootstrap']['seed']}): "
                  f"[{m['rmse_ci95_px'][0]:.4f}, {m['rmse_ci95_px'][1]:.4f}] px")
            print(f"    Error median / p90 / max: {m['error_median_px']:.4f} / "
                  f"{m['error_p90_px']:.4f} / {m['error_max_px']:.4f} px")
            rn = m["repeat_noise"]
            if rn["repeat_noise_px"] is None:
                print(f"    Repeat noise: null ({rn['reason']})")
            else:
                print(f"    Repeat noise: {rn['repeat_noise_px']:.4f} px "
                      f"({rn['n_repeat_groups']} groups, {rn['n_repeat_clicks']} clicks)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"  full report -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
