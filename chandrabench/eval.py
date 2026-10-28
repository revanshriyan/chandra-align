#!/usr/bin/env python3
"""ChandraBench v0.2 — one-command evaluator.

Scores a participant's 2x3 affine transform (source -> reference pixel
coordinates, the cv2.warpAffine convention) against the frozen human-verified
landmark sets.

Usage:
    python eval.py --pair ohrc_pair --matrix "1,0,0,0,1,0"
    python eval.py --pair tmc2_fore_nadir --matrix-file my_transform.csv
    python eval.py --pair ohrc_01 --matrix "1,0,0,0,1,0"
    python eval.py --self-test

The matrix is 6 numbers in row-major order: [a b c d e f] meaning
    x_ref = a*x_src + b*y_src + c
    y_ref = d*x_src + e*y_src + f

A --matrix-file may hold the same 6 numbers as CSV, or a 2x3 / 3x3 matrix.

Metric (fixed for v0.2):
    forward RMSE  = sqrt(mean(|| M @ p_src - p_ref ||^2))   over all GT points
    inverse RMSE  = sqrt(mean(|| M^-1 @ p_ref - p_src ||^2))
Both are reported; neither is hidden. The headline score is forward RMSE.

COORDINATE SPACES — UNIFIED FOR THE OHRC PAIRS (2026-10-28 amendment):
    * The three OHRC pairs (ohrc_pair, ohrc_01, ohrc_04) now score in
      one unified v0.1 space (loaded-4096): the OHRC strips
      (12000x93693 px) loaded with max_dimension=4096 (loaded space
      525x4096 px). The package's historical wording "canonical
      full-resolution pixel space" means this loaded space, not native
      strip pixels.
    * v0.2 pairs (ohrc_01, ohrc_04) were re-derived into v0.1 space on
      2026-10-28 from their 2048x2048 benchmark-crop coordinates
      (X_v01 = (origin_x + x_crop) x 4096/93693; origins in each CSV's
      caveat and in README/CHANGELOG). Historical note: the v0.2 draft
      as first prepared (2026-10-26) used crop coordinates; that split
      is now resolved for the OHRC pairs.
    * tmc2_fore_nadir keeps its audited 2048x2048 windows, as stated
      in its CSV; it was never in OHRC space.
    * Precision caveat: the native-to-loaded point map is a linear
      convention good to ~half a loaded pixel (~11 native px) in the
      worst case; cross-space landmark comparisons inherit that
      quantization.

Circularity guard (fail-closed):
    * The landmark CSVs are frozen at their version. Their sha256 is pinned
      below; a tampered or substituted GT file aborts the run.
    * GT points must never be used to fit the submitted transform. This
      evaluator cannot see your fitting process, so compliance is by
      declaration: --i-declare-gt-not-used-for-fitting is required. The flag
      is recorded in the output report.
    * GT quadrant coverage is checked and reported (protocol floor: >=3 per
      quadrant). The v0.2 pairs fall short of the floor (ohrc_01 LT=2,
      ohrc_04 LB=2); every report flags floor_pass=false for them, and the
      shortfall is stated in their CSV caveats and the README. The floor
      is a coverage report, not a hidden score adjustment.

Self-test note: the self-test asserts evaluator arithmetic (identity RMSE
matches a direct computation; a GT self-fit beats identity and matches a
brute-force recomputation). Quadrant coverage is a property of the GT set,
not of the evaluator, so the self-test REPORTS floor status per pair
instead of gating on it; a below-floor pair is printed as a coverage
caveat, not a self-test failure.

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
        "sha256": "23f1aaf67ca8ed544fd3c21adf54355b31db3ac627d7dbee36cbb53c1a6b3752",
        "n_points": 20,
    },
    "tmc2_fore_nadir": {
        "csv": "landmarks_tmc2_fore_nadir_v0.1.csv",
        "sha256": "84500aaa42d2ee529f302b5320eb7b2123c5a1d19602b14385fb1c3002def1cc",
        "n_points": 20,
    },
    "ohrc_01": {
        "csv": "landmarks_ohrc_01_v0.2.csv",
        "sha256": "c51326bbffccbf7e2ad3dabaf134ebee340f94820e42f518fd6398f77a9a59be",
        "n_points": 21,
    },
    "ohrc_04": {
        "csv": "landmarks_ohrc_04_v0.2.csv",
        "sha256": "b98ad22bc3b4506b56cd53e72ce1389f591df72e68b1d4560207d79931c1165e",
        "n_points": 18,
    },
}

VERSION = "0.2"


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
            "The landmark set is frozen; do not modify it."
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
    """Self-consistency: identity submission and a GT self-fit submission.

    Gates on evaluator arithmetic only (c1, c2). Quadrant coverage is
    reported per pair, not gated: a below-floor pair is a GT-coverage
    caveat printed explicitly, not an evaluator failure.
    """
    print(f"ChandraBench v{VERSION} self-test")
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
        floor_note = "" if r_id["gt_quadrant_floor_pass"] else \
            "  COVERAGE CAVEAT: below the >=3/quadrant protocol floor"
        print(f"  {pair}: identity_rmse={r_id['rmse_forward_px']:.4f}px "
              f"(independent check {direct:.4f}px) "
              f"self_fit_rmse={r_fit['rmse_forward_px']:.2e}px "
              f"quadrants={r_id['gt_quadrant_counts_ref']} "
              f"floor_pass={r_id['gt_quadrant_floor_pass']}")
        if floor_note:
            print(f"    {floor_note.strip()}")
        if not (c1 and c2):
            ok = False
            print(f"    FAIL: c1={c1} c2={c2}")
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="ChandraBench evaluator")
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
