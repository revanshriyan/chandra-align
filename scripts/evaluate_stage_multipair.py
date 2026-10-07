"""A15: evaluate the opt-in deformation-field stage across all benchmark pairs.

Runs the TRUE app.py::_align_core twice per pair (flag OFF then flag ON);
the ONLY difference between arms is the CHANDRA_DEFORM_FIELD env flag.
Import path: app.py imports gradio/spaces/matplotlib at module level;
this script expects stub modules for those on PYTHONPATH (see
/tmp/stubs/*) because the system matplotlib is ABI-broken vs numpy 2.5.3
and _align_core never renders. Documented, no repo files touched.

Deterministic: the pipeline seeds OpenCV RNG per fit (cv2.setRNGSeed(0));
SIFT detection is deterministic. A flag-off re-run of ohrc_01 at the end
checks determinism empirically.
"""
import csv
import json
import os
import sys
import time
import traceback

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import app  # noqa: E402  (needs /tmp/stubs on PYTHONPATH for gradio/spaces/mpl)

CROP = os.path.join(REPO, "data", "benchmark_crops")
CSV_PATH = os.path.join(REPO, "results", "table_stage_multipair.csv")

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
                print("[%5.0fs] %-18s status=%s inl=%s rmse_in=%s applied=%s reason=%s (%.0fs)"
                      % (time.time() - t0, label, row["status_code"],
                         row["n_inliers"],
                         ("%.4f" % row["rmse_in_sample_px"]
                          if row["rmse_in_sample_px"] is not None else None),
                         row["df_applied"], row["df_reason"], dt), flush=True)
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
    print("wrote %s (%.0fs total)" % (CSV_PATH, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
