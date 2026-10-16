"""Strip-guard calibration from human-clicked landmarks (ohrc_01, ohrc_04).

Background
----------
The strip-collapse guard (chandra_align/metrics/spatial_support.py) measures
frame-normalized PCA support of the inlier cloud and can downgrade the verdict
to STRIP_COLLAPSE when minor_ratio < 0.25. That downgrade is OPT-IN
(CHANDRA_STRIP_GUARD=1) because the 12-pair regression showed 0.25 trips
verified-good fits: ohrc_01 (minor_ratio 0.243, SUCCESS_SUBPIXEL 0.328 px) and
ohrc_04's main fit (0.19, verified 0.29 px via L1 rescue). See
docs/strip-collapse-guard-2026-10-15.md.

Revan clicks ~20 landmarks per pair in
~/workspace/your_files/landmark_clicker_ohrc01_ohrc04.html ("algorithm proposes,
human verifies"). This script consumes the two exported CSVs and produces the
calibration evidence for the default-ON decision.

Method per pair
--------------
 1. Run app._align_core on data/benchmark_crops (production config behind the
    nine sub-pixel verdicts: CHANDRA_DEFORM_FIELD=1, CHANDRA_STRIP_GUARD=0,
    cv2.setRNGSeed(0)).
 2. Take affine_matrix M (warp-style: maps source coords -> reference coords),
    the inlier cloud, and strip_support_metrics.
 3. Load the landmark CSV
    (id,x_ref,y_ref,x_src,y_src,ncc,model_error_small_px,review_status);
    keep rows with review_status == VERIFIED (>= MIN_VERIFIED required).
 4. Landmark RMSE = rms( M @ p_src - p_ref ) over verified landmarks, in
    benchmark-crop pixels.
 5. Threshold sweep over candidate T: count verified-good pairs each T would
    flip, and whether the known-collapsed anchor (TMC-2 fore/nadir, 0.075)
    is caught.

This script changes NOTHING: no gates, thresholds, or defaults are touched.
The default-ON / threshold-value decision stays with Revan; the script only
produces the evidence table and a recommendation.

Usage
-----
    python scripts/calibrate_strip_guard.py \
        --landmarks ohrc_01=/path/to/gt_ohrc_01_human_clicked.csv \
                    ohrc_04=/path/to/gt_ohrc_04_human_clicked.csv

If a CSV is missing, the pipeline half still runs and the landmark columns
are left empty with a "landmarks pending" note.

Output: results/table_strip_guard_calibration_<YYYY-MM-DD>.csv
"""

import csv
import os
import sys
import time
from datetime import date
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
REGRESSION_CSV = os.path.join(
    REPO, "results", "table_strip_guard_regression_2026-10-15.csv")
OUT = os.path.join(REPO, "results",
                   "table_strip_guard_calibration_%s.csv" % date.today().isoformat())

PAIRS = [("ohrc_01", "OHRC"), ("ohrc_04", "OHRC")]

# Anchors from the 2026-10-15 human-landmark rerun (different, full-PDS4
# windows -- not benchmark crops, so they are cited, not re-run).
# Sources: docs/strip-collapse-guard-2026-10-15.md (strip metrics),
# results/table_issue01_landmark_rerun_2026-10-15.csv (landmark RMSE).
ANCHORS = [
    {"pair": "TMC2_fore_nadir_anchor", "collapsed": True,
     "minor_ratio": 0.075, "landmark_rmse_px": 28.031705, "n_verified": 20,
     "notes": "known-collapsed: 13 SIFT inliers in thin strip; 28.03px landmark RMSE"},
    {"pair": "OHRC_pair_anchor", "collapsed": False,
     "minor_ratio": 0.717, "landmark_rmse_px": 5.471325, "n_verified": 20,
     "notes": "known-healthy: no collapse; 5.47px is ~60% non-projective parallax"},
]

MIN_VERIFIED = 10          # hard floor; protocol asks ~20
WARN_VERIFIED = 20         # below this, flag thin coverage
SELFCHECK_TOL = 0.005      # vs the 12-pair regression strip ratios

COLUMNS = ["pair", "source", "n_verified_landmarks", "landmark_rmse_px",
           "n_inliers", "status_code", "strip_minor_ratio", "strip_major_ratio",
           "collapsed_at_0.25", "notes"]


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_landmarks(path):
    """Read a clicker-exported CSV; return verified (x_ref,y_ref,x_src,y_src)."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("review_status") or "").strip().upper() != "VERIFIED":
                continue
            try:
                rows.append((float(r["x_ref"]), float(r["y_ref"]),
                             float(r["x_src"]), float(r["y_src"])))
            except (KeyError, ValueError, TypeError):
                continue
    return rows


def landmark_rmse(M, landmarks):
    """rms( M @ p_src - p_ref ); M is warp-style 2x3 (source -> reference)."""
    M = np.asarray(M, dtype=np.float64).reshape(2, 3)
    errs = []
    for xr, yr, xs, ys in landmarks:
        px = M[0, 0] * xs + M[0, 1] * ys + M[0, 2]
        py = M[1, 0] * xs + M[1, 1] * ys + M[1, 2]
        errs.append((px - xr) ** 2 + (py - yr) ** 2)
    return float(np.sqrt(np.mean(errs))) if errs else None


def run_pair(pair, sensor):
    ref = cv2.imread(os.path.join(CROP, "%s_reference.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(os.path.join(CROP, "%s_source.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    if ref is None or sec is None:
        return None, "missing_image"
    old = {k: os.environ.get(k) for k in
           ("CHANDRA_STRIP_GUARD", "CHANDRA_DEFORM_FIELD")}
    try:
        os.environ["CHANDRA_STRIP_GUARD"] = "0"
        os.environ["CHANDRA_DEFORM_FIELD"] = "1"
        cv2.setRNGSeed(0)
        t = time.time()
        out = app._align_core(ref, sec, reference_sensor_name=sensor)
        el = time.time() - t
    except Exception as exc:  # fail closed in the harness, too
        return None, "exception:%s:%s" % (type(exc).__name__, str(exc)[:120])
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    ss = out.get("strip_support_metrics", {}) or {}
    return {
        "pair": pair,
        "M": out.get("affine_matrix"),
        "n_inliers": out.get("inlier_cnt"),
        "status_code": out.get("status_code"),
        "minor_ratio": _f(ss.get("minor_ratio")),
        "major_ratio": _f(ss.get("major_ratio")),
        "collapsed_025": bool(ss.get("collapsed", False)),
        "elapsed_s": round(el, 1),
    }, None


def selfcheck(rows):
    """Compare strip ratios against the 12-pair regression (config-drift guard)."""
    if not os.path.exists(REGRESSION_CSV):
        print("self-check: regression CSV not found, skipping")
        return
    reg = {}
    with open(REGRESSION_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("guard") == "off":
                reg[r["pair"]] = _f(r.get("strip_minor_ratio"))
    for row in rows:
        exp = reg.get(row["pair"])
        got = row.get("minor_ratio")
        if exp is None or got is None:
            print("self-check %s: missing value (exp=%s got=%s)" %
                  (row["pair"], exp, got))
        elif abs(exp - got) > SELFCHECK_TOL:
            print("self-check %s: DRIFT exp=%.4f got=%.4f -- config changed "
                  "since the 12-pair regression; investigate before trusting "
                  "calibration" % (row["pair"], exp, got))
        else:
            print("self-check %s: strip ratio reproduces (%.4f)" %
                  (row["pair"], got))


def recommend(labeled):
    """Threshold recommendation from labeled (collapsed?, minor_ratio) points."""
    bad = [r["minor_ratio"] for r in labeled
           if r["collapsed_label"] and r["minor_ratio"] is not None]
    good = [r["minor_ratio"] for r in labeled
            if not r["collapsed_label"] and r["minor_ratio"] is not None]
    print("\n--- threshold analysis ---")
    if not bad or not good:
        print("insufficient labeled data; keep guard opt-in")
        return
    bad_max, good_min = max(bad), min(good)
    prov = {}
    for r in labeled:
        if not r["collapsed_label"]:
            prov[r.get("provenance", "?")] = \
                prov.get(r.get("provenance", "?"), 0) + 1
    print("verified-good labels by provenance: %s" % prov)
    print("known-collapsed max minor_ratio: %.3f" % bad_max)
    print("verified-good   min minor_ratio: %.3f" % good_min)
    if bad_max < good_min:
        t_star = (bad_max + good_min) / 2.0
        print("clean separation exists.")
        print("recommended threshold T* = %.3f "
              "(midpoint; margin %.3f each side)" %
              (t_star, (good_min - bad_max) / 2.0))
        print("0.25 would flip %d verified-good pair(s) -> too aggressive "
              "as default-ON" %
              sum(1 for g in good if g < 0.25))
        print("NOTE: adopting any new threshold / default-ON is Revan's call; "
              "this script changes nothing.")
    else:
        print("NO clean separation: a collapsed and a verified-good pair "
              "overlap. Keep the guard opt-in; wider landmark GT needed.")


def main():
    lm_paths = {}
    for arg in sys.argv[1:]:
        if arg.startswith("--landmarks"):
            continue
        if "=" in arg:
            k, v = arg.split("=", 1)
            lm_paths[k.strip()] = v.strip()
    # default search locations
    for pair, _ in PAIRS:
        if pair not in lm_paths:
            for cand in (os.path.join(REPO, "gt_%s_human_clicked.csv" % pair),
                         os.path.expanduser(
                             "~/workspace/your_files/gt_%s_human_clicked.csv" % pair),
                         os.path.join(os.getcwd(),
                                      "gt_%s_human_clicked.csv" % pair)):
                if os.path.exists(cand):
                    lm_paths[pair] = cand
                    break

    pipe_rows = []
    for pair, sensor in PAIRS:
        print("running %s ..." % pair, flush=True)
        res, err = run_pair(pair, sensor)
        if err:
            pipe_rows.append({"pair": pair, "error": err})
            print("  %s -> %s" % (pair, err))
            continue
        print("  %s -> %s (n=%s, minor=%.4f, %.1fs)" % (
            pair, res["status_code"], res["n_inliers"],
            res["minor_ratio"], res["elapsed_s"]))
        pipe_rows.append(res)

    selfcheck([r for r in pipe_rows if "error" not in r])

    table, labeled = [], []
    for res in pipe_rows:
        pair = res["pair"]
        row = {"pair": pair, "source": "benchmark_crops+_align_core",
               "n_verified_landmarks": None, "landmark_rmse_px": None,
               "n_inliers": res.get("n_inliers"),
               "status_code": res.get("status_code"),
               "strip_minor_ratio": res.get("minor_ratio"),
               "strip_major_ratio": res.get("major_ratio"),
               "collapsed_at_0.25": res.get("collapsed_025"),
               "notes": ""}
        if "error" in res:
            row["notes"] = "pipeline " + res["error"]
            table.append(row)
            continue
        path = lm_paths.get(pair)
        if not path:
            row["notes"] = ("landmarks pending: export gt_%s_human_clicked.csv "
                            "from the clicker" % pair)
            table.append(row)
            # Labeled good on PRIOR verification only (benchmark sub-pixel /
            # L1-rescue verdicts); the landmark check for this pair is pending.
            labeled.append({"collapsed_label": False,
                            "minor_ratio": res.get("minor_ratio"),
                            "provenance": "prior-verified"})
            continue
        lms = load_landmarks(path)
        row["n_verified_landmarks"] = len(lms)
        if len(lms) < MIN_VERIFIED:
            row["notes"] = ("only %d VERIFIED landmarks (< %d); insufficient "
                            "for calibration" % (len(lms), MIN_VERIFIED))
            table.append(row)
            continue
        if len(lms) < WARN_VERIFIED:
            row["notes"] = "thin coverage: %d verified (< %d)" % (len(lms),
                                                                   WARN_VERIFIED)
        rmse = landmark_rmse(res["M"], lms)
        row["landmark_rmse_px"] = None if rmse is None else round(rmse, 4)
        if row["notes"] == "":
            row["notes"] = "landmarks from %s" % os.path.basename(path)
        table.append(row)
        # A pair counts as verified-good only if human landmarks agree with
        # the fit; otherwise its label is unknown, not good.
        if rmse is not None and rmse < 2.0:
            labeled.append({"collapsed_label": False,
                            "minor_ratio": res.get("minor_ratio"),
                            "provenance": "landmark-verified"})

    for a in ANCHORS:
        table.append({"pair": a["pair"], "source": "2026-10-15 landmark rerun",
                      "n_verified_landmarks": a["n_verified"],
                      "landmark_rmse_px": a["landmark_rmse_px"],
                      "n_inliers": None, "status_code": None,
                      "strip_minor_ratio": a["minor_ratio"],
                      "strip_major_ratio": None,
                      "collapsed_at_0.25": a["minor_ratio"] < 0.25,
                      "notes": a["notes"]})
        labeled.append({"collapsed_label": a["collapsed"],
                        "minor_ratio": a["minor_ratio"],
                        "provenance": "landmark-verified (2026-10-15 rerun)"})

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(table)
    print("\nwrote", OUT)
    for r in table:
        print("  %(pair)s: n_ver=%(n_verified_landmarks)s rmse=%(landmark_rmse_px)s "
              "minor=%(strip_minor_ratio)s collapsed@0.25=%(collapsed_at_0.25)s "
              "[%(notes)s]" % r)
    recommend(labeled)


if __name__ == "__main__":
    main()
