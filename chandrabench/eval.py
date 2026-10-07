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


def evaluate(pair, M, declare_flag):
    ids, ref, src, digest = load_landmarks(pair)
    qcounts, qpass = quadrant_counts(ref)
    s = score(M, ids, ref, src)
    return {
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
    report = evaluate(args.pair, M, True)

    print(f"ChandraBench v{VERSION} — {args.pair}")
    print(f"  GT points: {report['n_gt_points']}  sha256: {report['gt_sha256'][:16]}...")
    print(f"  GT quadrants (ref): {report['gt_quadrant_counts_ref']} "
          f"floor_pass={report['gt_quadrant_floor_pass']}")
    print(f"  RMSE forward : {report['rmse_forward_px']:.4f} px   <-- headline score")
    print(f"  RMSE inverse : {report['rmse_inverse_px']:.4f} px")
    print(f"  Max fwd residual: {report['max_fwd_res_px']:.4f} px")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"  full report -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
