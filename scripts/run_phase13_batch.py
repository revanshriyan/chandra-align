"""Phase 13 — window-tiling batch harness (anti-cherry-picking).

Systematic per-window registration over real products. Every eligible
window is evaluated and reported, including failures: failures are ROWS,
not exceptions. This table is kept strictly separate from hand-picked
pairs (results/table_canonical_v1.csv).

Pairs: window origins are generated in product A's frame; product B's
window origin comes from the MEASURED inter-product affine (from
scripts/make_benchmark_crops.py, mapped on full-strip browse PNGs —
not fitted here). This yields true shared-ground windows:
  tmc2_fore_nadir: TMC-2 fore vs nadir, 189998x4000 uint16 LE.
                   Fore->nadir map: ~1.004 scale, (-315, +7780) px.
                   Fore footprint: lines 120000-135000, cols 512-3584.
  ohrc_pair:       OHRC 12:09 vs 14:06, 93693x12000 uint8.
                   12:09->14:06 map: ~1.006 scale, (+1060, +1302) px.
                   12:09 footprint: lines 40000-55000, cols 4096-7168.

The harness measures what the default pipeline does per window — no
coarse pre-alignment beyond the measured product-level map. Residual
cross-view geometry appears as honest DEGENERATE_FAILURE rows.

Pipeline per window (frozen; measured, never modified):
  SIFT detect+compute -> BF kNN + Lowe 0.75 -> verify_guarded ->
  quadrant metrics -> validate_registration_gate (min 8 inliers) ->
  trust_evaluate held-out split on the guarded inliers ->
  build_confidence_assessment (0-100 heuristic, report-only).

Writes results/table_phase13_windows.csv. One script, one table.

Run: PYTHONPATH=. python3 scripts/run_phase13_batch.py
"""

import csv
import time

import numpy as np

from chandra_align.eval.robustness import measure_pair
from chandra_align.eval.tiling import generate_windows
from chandra_align.metrics.reporting import build_confidence_assessment
from chandra_align.trust import evaluate as trust_evaluate

WINDOW = 1024
SEED = 13

QA = "/home/hatch/workspace/qa_lunar/pradan"

PAIRS = [
    {
        "name": "tmc2_fore_nadir",
        "path_a": f"{QA}/ch2_tmc_ncf_20231101T0125121344_d_img_d18/data/calibrated/20231101/ch2_tmc_ncf_20231101T0125121344_d_img_d18.img",
        "shape_a": (190000, 4000), "dtype_a": "<u2",
        "path_b": f"{QA}/ch2_tmc_ncn_20231101T0125121377_d_img_d18/data/calibrated/20231101/ch2_tmc_ncn_20231101T0125121377_d_img_d18.img",
        "shape_b": (189998, 4000), "dtype_b": "<u2",
        "region": (120000, 135000, 512, 3584),
        # Measured fore->nadir (scripts/make_benchmark_crops.py).
        "map_a_to_b": [[1.00421806, 0.00238950, -315.0303],
                       [-0.00238950, 1.00421806, 7780.0726]],
    },
    {
        "name": "ohrc_pair",
        "path_a": f"{QA}/ch2_ohr_nrp_20240425T1209509264_d_img_d18/data/raw/20240425/ch2_ohr_nrp_20240425T1209509264_d_img_d18.img",
        "shape_a": (93693, 12000), "dtype_a": "u1",
        "path_b": f"{QA}/ch2_ohr_nrp_20240425T1406019344_d_img_d18/data/raw/20240425/ch2_ohr_nrp_20240425T1406019344_d_img_d18.img",
        "shape_b": (93693, 12000), "dtype_b": "u1",
        "region": (40000, 55000, 4096, 7168),
        # Measured 12:09->14:06 (scripts/make_benchmark_crops.py).
        "map_a_to_b": [[1.00553448, -0.01578604, 1060.4775],
                       [0.01578604, 1.00553448, 1302.2119]],
    },
]


def map_origin(M, x0, y0):
    """Map a window origin from product A to product B via the 2x3 map."""
    M = np.asarray(M, dtype=np.float64).reshape(2, 3)
    xb = M[0, 0] * x0 + M[0, 1] * y0 + M[0, 2]
    yb = M[1, 0] * x0 + M[1, 1] * y0 + M[1, 2]
    return int(round(xb)), int(round(yb))

COLUMNS = ["window_id", "pair", "x0", "y0", "size", "n_matches",
           "n_inliers_raw", "n_inliers_unique", "rmse_px", "rmse_heldout_px",
           "entropy", "quadrants", "verdict", "confidence", "runtime_s",
           "notes"]


def _fmt_quadrants(qd):
    if not qd:
        return ""
    counts = [qd.get(f"Q{i}", 0) for i in range(1, 5)]
    active = sum(1 for c in counts if c > 0)
    return f"{active}/4 ({','.join(str(c) for c in counts)})"


def run_window(pair_name, mm_a, mm_b, win, b_origin):
    """Run the frozen pipeline on one window; never raises.

    win: A-frame window dict (y0, x0, size). b_origin: (xb0, yb0) mapped
    origin in B's frame. Returns a dict with exactly the COLUMNS keys.
    All failures become rows.
    """
    t0 = time.time()
    row = {c: "" for c in COLUMNS}
    row.update({"window_id": f"{pair_name}_{win['window_id']}",
                "pair": pair_name, "x0": win["x0"], "y0": win["y0"],
                "size": win["size"],
                "notes": f"b_origin=({b_origin[0]},{b_origin[1]})"})
    try:
        y0, x0, s = win["y0"], win["x0"], win["size"]
        xb0, yb0 = b_origin
        a = np.asarray(mm_a[y0:y0 + s, x0:x0 + s])
        b = np.asarray(mm_b[yb0:yb0 + s, xb0:xb0 + s])
        if a.shape != (s, s) or b.shape != (s, s):
            row.update(verdict="ERROR", confidence=0.0,
                       notes="window read returned wrong shape")
            return row
        rec = measure_pair(a, b, None, image_shape=(s, s),
                           return_inliers=True)
        heldout, heldout_note = "", ""
        ia = rec.get("_inliers_a")
        if (rec["_model"] is not None and ia is not None and len(ia) >= 5):
            try:
                _Mr, rmse_dict = trust_evaluate(
                    rec["_model"], np.asarray(ia),
                    np.asarray(rec["_inliers_b"]))
                v = rmse_dict.get("rmse_px")
                if v == "UNMEASURED":
                    heldout_note = "held-out unmeasurable (<5 check points)"
                else:
                    heldout = round(float(v), 4)
            except Exception as exc:
                heldout_note = f"held-out error: {type(exc).__name__}"
        conf = build_confidence_assessment(
            status_code=rec["verdict"],
            status_message=rec.get("accuracy_note", ""),
            rmse_px=rec["rmse_px"], inlier_count=rec["inliers"],
            correspondence_count=rec["n_lowe"], entropy=rec["entropy"] or 0.0,
            quadrant_counts=rec.get("_quadrants", {}), min_inliers=8,
            execution_diagnostics={})
        notes = "; ".join(n for n in
                          [f"abstain={rec['abstain']}" if rec["abstain"] else "",
                           rec.get("accuracy_note", ""), heldout_note] if n)
        row.update({
            "n_matches": rec["n_lowe"],
            "n_inliers_raw": rec.get("_n_raw", ""),
            "n_inliers_unique": rec["inliers"],
            "rmse_px": ("" if rec["rmse_px"] is None
                        else round(float(rec["rmse_px"]), 4)),
            "rmse_heldout_px": heldout,
            "entropy": ("" if rec["entropy"] is None
                       else round(float(rec["entropy"]), 4)),
            "quadrants": _fmt_quadrants(rec.get("_quadrants")),
            "verdict": rec["verdict"],
            "confidence": conf["confidence_score"],
            "notes": notes,
        })
    except Exception as exc:  # failures are rows, not exceptions
        row.update(verdict="ERROR", confidence=0.0,
                   notes=f"{type(exc).__name__}: {str(exc)[:160]}")
    finally:
        row["runtime_s"] = round(time.time() - t0, 2)
    return row


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="skip window_ids already present in the CSV")
    args = parser.parse_args()

    done = set()
    if args.resume:
        try:
            with open("results/table_phase13_windows.csv") as fh:
                done = {r["window_id"] for r in csv.DictReader(fh)}
        except FileNotFoundError:
            pass
        if done:
            print(f"resume: skipping {len(done)} completed windows", flush=True)

    # Incremental writes: each row is flushed to the CSV as it completes,
    # so a restart loses at most the in-flight window.
    write_header = not (args.resume and done)
    fh = open("results/table_phase13_windows.csv",
              "a" if args.resume and done else "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=COLUMNS)
    if write_header:
        writer.writeheader()
        fh.flush()
    n_new = 0
    try:
        for pair in PAIRS:
            mm_a = np.memmap(pair["path_a"], dtype=pair["dtype_a"], mode="r",
                             shape=pair["shape_a"])
            mm_b = np.memmap(pair["path_b"], dtype=pair["dtype_b"], mode="r",
                             shape=pair["shape_b"])

            def read_a(y0, x0, size, _mm=mm_a):
                return np.asarray(_mm[y0:y0 + size, x0:x0 + size])

            wins = generate_windows(pair["shape_a"][:2], WINDOW, pair["region"],
                                    seed=SEED, read_window=read_a)
            eligible = [w for w in wins if w["eligible"]]
            skipped = len(wins) - len(eligible)
            print(f"{pair['name']}: {len(wins)} windows, {len(eligible)} eligible, "
                  f"{skipped} skipped (dark/flat/unreadable)", flush=True)
            hb, wb = pair["shape_b"]
            b_skipped = 0
            for w in eligible:
                wid = f"{pair['name']}_{w['window_id']}"
                if wid in done:
                    continue
                xb0, yb0 = map_origin(pair["map_a_to_b"], w["x0"], w["y0"])
                # B window must be fully inside B's raster and lit/textured.
                if not (0 <= xb0 and 0 <= yb0
                        and xb0 + WINDOW <= wb and yb0 + WINDOW <= hb):
                    b_skipped += 1
                    continue
                from chandra_align.eval.tiling import window_stats
                b_win = np.asarray(mm_b[yb0:yb0 + WINDOW, xb0:xb0 + WINDOW])
                _m, _s, _d = window_stats(b_win)
                if _d > 0.5 or _s < 8.0:
                    b_skipped += 1
                    continue
                r = run_window(pair["name"], mm_a, mm_b, w, (xb0, yb0))
                writer.writerow(r)
                fh.flush()
                n_new += 1
                print(f"  {r['window_id']}: {r['verdict']} "
                      f"inliers={r['n_inliers_unique']} rmse={r['rmse_px']} "
                      f"heldout={r['rmse_heldout_px']} ({r['runtime_s']}s)",
                      flush=True)
            if b_skipped:
                print(f"  ({b_skipped} A-eligible windows skipped: "
                      f"B-side out of bounds or dark/flat)", flush=True)
    finally:
        fh.close()
    print(f"done: {n_new} new rows "
          f"(results/table_phase13_windows.csv)")


if __name__ == "__main__":
    main()
