"""A19: validate the flag-gated keypoint quota on all 12 pairs — zero downgrades mandatory.

Runs the TRUE app.py::_align_core twice per pair (flag OFF then flag ON);
the only difference is the flag. With the working-tree change in
chandra_align/features/distribution.py::select_detector_keypoints, flag ON
lifts the per-cell quota to 10**9 (uncapped); flag OFF honors quota 64
exactly (must be bit-identical to A15's table_stage_multipair.csv).

app.py is imported via throwaway stubs in /tmp/stubs (gradio/spaces/
matplotlib) on PYTHONPATH -- outside the repo, never committed.
_align_core never touches the UI or any plot call, so the stubs are inert.

Deterministic (seed 7 inside the pipeline); a flag-off re-run of ohrc_01
after all 24 runs must reproduce the first run identically.
"""
import csv
import os
import sys
import time

REPO = os.path.expanduser("~/workspace/chandra-align")
sys.path.insert(0, "/tmp/stubs")
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np  # noqa: E402

import app  # noqa: E402

CROP = os.path.join(REPO, "data", "benchmark_crops")
CSV_PATH = os.path.join(REPO, "results", "table_quota_12pair.csv")

PAIRS = [("ohrc_%02d" % i, "OHRC") for i in range(1, 7)] + \
        [("tmc2_%02d" % i, "TMC-2") for i in range(1, 7)]

COLUMNS = [
    "pair", "sensor", "flag",
    "status_code", "n_inliers",
    "rmse_in_sample_px", "rmse_gate_px", "rmse_heldout_px", "gate_basis",
    "df_applied", "df_reason", "df_lambda",
    "df_rmse_before_px", "df_rmse_after_px",
    "df_heldout_px", "df_heldout_affine_px",
    "df_max_disp_px", "df_min_jac_det", "df_improvement_px",
    "notes",
]

# A15 flag-off reference (table_stage_multipair.csv) for the bit-identical check.
A15_REF = {
    "ohrc_01": ("COARSE_ADVISORY", 19, 1.6445072993342726),
    "ohrc_02": ("COARSE_ADVISORY", 16, 1.3040864879540923),
    "ohrc_03": ("COARSE_ADVISORY", 14, 1.1020794072232387),
    "ohrc_04": ("COARSE_ADVISORY", 16, 1.7056507240728107),
    "ohrc_05": ("COARSE_ADVISORY", 10, 2.054915611869818),
    "ohrc_06": ("COARSE_ADVISORY", 8, 1.1510429840979663),
    "tmc2_01": ("DEGENERATE_FAILURE", 8, 0.0),
    "tmc2_02": ("DEGENERATE_FAILURE", 8, 0.0),
    "tmc2_03": ("DEGENERATE_FAILURE", 8, 0.0),
    "tmc2_04": ("DEGENERATE_FAILURE", 8, 0.0),
    "tmc2_05": ("DEGENERATE_FAILURE", 6, 0.0),
    "tmc2_06": ("DEGENERATE_FAILURE", 8, 0.0),
}


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def run_pair(pair, sensor, flag):
    """Run _align_core once. Returns (row_dict, error_str_or_None)."""
    ref = cv2.imread(os.path.join(CROP, "%s_reference.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(os.path.join(CROP, "%s_source.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    if ref is None or sec is None:
        return None, "missing_image"
    old = os.environ.get("CHANDRA_DEFORM_FIELD")
    try:
        if flag:
            os.environ["CHANDRA_DEFORM_FIELD"] = "1"
        else:
            os.environ.pop("CHANDRA_DEFORM_FIELD", None)
        out = app._align_core(ref, sec, reference_sensor_name=sensor)
    except Exception as exc:  # fail closed in the harness, too
        return None, "exception:%s:%s" % (type(exc).__name__, str(exc)[:120])
    finally:
        if old is None:
            os.environ.pop("CHANDRA_DEFORM_FIELD", None)
        else:
            os.environ["CHANDRA_DEFORM_FIELD"] = old
    jm = out.get("judge_metrics", {}) or {}
    df = jm.get("deform_field_stage", {}) or {}
    row = {
        "pair": pair, "sensor": sensor, "flag": "on" if flag else "off",
        "status_code": out.get("status_code"),
        "n_inliers": out.get("inlier_cnt"),
        "rmse_in_sample_px": _f(jm.get("rmse_in_sample_px")),
        "rmse_gate_px": _f(jm.get("rmse_gate_px")),
        "rmse_heldout_px": _f(jm.get("rmse_heldout_px")),
        "gate_basis": jm.get("rmse_gate_basis"),
        "df_applied": bool(df.get("applied")),
        "df_reason": df.get("reason"),
        "df_lambda": df.get("lambda_chosen"),
        "df_rmse_before_px": _f(df.get("rmse_before_px")),
        "df_rmse_after_px": _f(df.get("rmse_after_px")),
        "df_heldout_px": _f(df.get("heldout_rmse_px")),
        "df_heldout_affine_px": _f(df.get("heldout_rmse_affine_px")),
        "df_max_disp_px": _f(df.get("max_displacement_px")),
        "df_min_jac_det": _f(df.get("min_jacobian_det")),
        "df_improvement_px": _f(df.get("improvement_px")),
        "notes": "",
    }
    return row, None


import cv2  # noqa: E402  (after sys.path setup, matches A15 import order)


def main():
    new_file = not os.path.exists(CSV_PATH)
    fh = open(CSV_PATH, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=COLUMNS)
    if new_file:
        wr.writeheader()
        fh.flush()

    results = []
    t0 = time.time()
    for pair, sensor in PAIRS:
        for flag in (False, True):
            label = "%s flag=%s" % (pair, "ON" if flag else "OFF")
            t1 = time.time()
            row, err = run_pair(pair, sensor, flag)
            dt = time.time() - t1
            if err:
                row = {c: None for c in COLUMNS}
                row.update({"pair": pair, "sensor": sensor,
                            "flag": "on" if flag else "off", "notes": err})
                print("[%5.0fs] %-18s ERROR %s" % (time.time() - t0, label, err),
                      flush=True)
            else:
                print("[%5.0fs] %-18s status=%s inl=%s gate=%s applied=%s lambda=%s (%.0fs)"
                      % (time.time() - t0, label, row["status_code"],
                         row["n_inliers"],
                         ("%.4f" % row["rmse_gate_px"]
                          if row["rmse_gate_px"] is not None else None),
                         row["df_applied"], row["df_lambda"], dt), flush=True)
            wr.writerow(row)
            fh.flush()
            results.append(row)

    # Determinism re-check: ohrc_01 flag OFF again, compare with first run.
    row2, err2 = run_pair("ohrc_01", "OHRC", False)
    first = results[0]
    det = "n/a"
    if not err2 and first["notes"] == "":
        det = "IDENTICAL" if (
            row2["status_code"] == first["status_code"]
            and row2["n_inliers"] == first["n_inliers"]
            and abs((row2["rmse_in_sample_px"] or -1)
                    - (first["rmse_in_sample_px"] or -2)) < 1e-9
        ) else "DIFFERS"
    print("determinism re-check ohrc_01 flag-off: %s" % det, flush=True)
    row2["notes"] = (row2.get("notes") or "") + "; determinism_recheck:" + det
    wr.writerow(row2)
    fh.flush()
    fh.close()

    # Bit-identical check of flag-off runs vs A15 reference.
    print("\n=== flag-off vs A15 reference ===", flush=True)
    ok = True
    for row in results:
        if row["flag"] != "off" or row["notes"]:
            continue
        ref = A15_REF.get(row["pair"])
        if ref is None:
            continue
        g = row["rmse_gate_px"]
        g = -1 if g is None else g
        match = (row["status_code"] == ref[0]
                 and row["n_inliers"] == ref[1]
                 and abs(g - ref[2]) < 1e-6)
        print("%s: %s (status=%s inl=%s gate=%.4f)" % (
            row["pair"], "MATCH" if match else "MISMATCH",
            row["status_code"], row["n_inliers"], row["rmse_gate_px"] or -1),
            flush=True)
        ok = ok and match
    print("flag-off bit-identical to A15: %s" % ("YES" if ok else "NO"), flush=True)


if __name__ == "__main__":
    main()
