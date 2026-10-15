"""Regression: strip-collapse guard on the 12-pair benchmark crops.

Runs app._align_core on data/benchmark_crops (ohrc_01..06, tmc2_01..06)
with CHANDRA_DEFORM_FIELD=1 (production config behind the 9 sub-pixel
verdicts), once with the strip guard OFF (CHANDRA_STRIP_GUARD=0) and once
with it ON (default). The ONLY acceptable verdict change is a
downgrade on a positively-collapsed inlier set; any other flip is
reported, not hidden.

app.py is imported via MagicMock stubs for gradio/spaces/matplotlib
(same method as scripts/lroc_pipeline_adjudication.py) -- _align_core
never touches the UI.

Output: results/table_strip_guard_regression_2026-10-15.csv
"""

import csv
import os
import sys
import time
from unittest.mock import MagicMock

for _mod in ('gradio', 'spaces', 'matplotlib', 'matplotlib.pyplot',
             'matplotlib.gridspec', 'matplotlib.colors', 'matplotlib.ticker',
             'matplotlib.transforms', 'matplotlib._path', 'matplotlib.cbook'):
    sys.modules[_mod] = MagicMock()

REPO = os.path.expanduser("~/workspace/chandra-align")
sys.path.insert(0, REPO)
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import app  # noqa: E402

CROP = os.path.join(REPO, "data", "benchmark_crops")
OUT = os.path.join(REPO, "results",
                   "table_strip_guard_regression_2026-10-15.csv")

PAIRS = [("ohrc_%02d" % i, "OHRC") for i in range(1, 7)] + \
        [("tmc2_%02d" % i, "TMC-2") for i in range(1, 7)]

COLUMNS = ["pair", "sensor", "guard", "status_code", "n_inliers",
           "rmse_gate_px", "strip_valid", "strip_collapsed",
           "strip_minor_ratio", "strip_major_ratio", "elapsed_s"]


def run_pair(pair, sensor, guard_on):
    ref = cv2.imread(os.path.join(CROP, "%s_reference.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(os.path.join(CROP, "%s_source.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    if ref is None or sec is None:
        return None, "missing_image"
    old_guard = os.environ.get("CHANDRA_STRIP_GUARD")
    old_df = os.environ.get("CHANDRA_DEFORM_FIELD")
    try:
        os.environ["CHANDRA_DEFORM_FIELD"] = "1"
        if guard_on:
            os.environ.pop("CHANDRA_STRIP_GUARD", None)
        else:
            os.environ["CHANDRA_STRIP_GUARD"] = "0"
        t = time.time()
        out = app._align_core(ref, sec, reference_sensor_name=sensor)
        el = time.time() - t
    except Exception as exc:  # fail closed in the harness, too
        return None, "exception:%s:%s" % (type(exc).__name__, str(exc)[:120])
    finally:
        if old_guard is None:
            os.environ.pop("CHANDRA_STRIP_GUARD", None)
        else:
            os.environ["CHANDRA_STRIP_GUARD"] = old_guard
        if old_df is None:
            os.environ.pop("CHANDRA_DEFORM_FIELD", None)
        else:
            os.environ["CHANDRA_DEFORM_FIELD"] = old_df
    jm = out.get("judge_metrics", {}) or {}
    ss = out.get("strip_support_metrics", {}) or jm.get("strip_support_metrics", {}) or {}

    def _f(x):
        try:
            v = float(x)
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    return {
        "pair": pair, "sensor": sensor,
        "guard": "on" if guard_on else "off",
        "status_code": out.get("status_code"),
        "n_inliers": out.get("inlier_cnt"),
        "rmse_gate_px": _f(jm.get("rmse_gate_px")),
        "strip_valid": bool(ss.get("valid", False)),
        "strip_collapsed": bool(ss.get("collapsed", False)),
        "strip_minor_ratio": _f(ss.get("minor_ratio")),
        "strip_major_ratio": _f(ss.get("major_ratio")),
        "elapsed_s": round(el, 1),
    }, None


def main():
    rows = []
    for pair, sensor in PAIRS:
        for guard_on in (False, True):
            row, err = run_pair(pair, sensor, guard_on)
            if err:
                row = {"pair": pair, "sensor": sensor,
                       "guard": "on" if guard_on else "off",
                       "status_code": "ERROR:" + err, "n_inliers": None,
                       "rmse_gate_px": None, "strip_valid": None,
                       "strip_collapsed": None, "strip_minor_ratio": None,
                       "strip_major_ratio": None, "elapsed_s": None}
            rows.append(row)
            print("%s guard=%s -> %s (n=%s, minor=%s)" % (
                pair, row["guard"], row["status_code"], row["n_inliers"],
                row["strip_minor_ratio"]), flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT)
    # Verdict-change summary.
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair"], {})[r["guard"]] = r["status_code"]
    print("\nverdict changes (guard off -> on):")
    changed = 0
    for pair in sorted(by_pair):
        off, on = by_pair[pair].get("off"), by_pair[pair].get("on")
        if off != on:
            changed += 1
            print("  %s: %s -> %s" % (pair, off, on))
    print("changed:", changed, "of", len(by_pair))


if __name__ == "__main__":
    main()
