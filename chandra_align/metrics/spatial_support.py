"""Post-RANSAC spatial-support (strip-collapse) guard.

Lesson from the 2026-10-15 TMC-2 fore/nadir landmark rerun (audit item #1):
13 SIFT inliers collapsed to a thin horizontal strip (ref y in [944,1096]
of a 2048 frame) while the true correspondence field spanned the frame.
The global partial-affine was locally valid inside the strip but
extrapolated catastrophically outside it: 1.77 px inlier RMSE vs 28.03 px
frame-wide human-landmark RMSE (~16x understatement). Quadrant entropy was
1.95 with 4/4 quadrants -- the gate did not catch the collapse because the
strip straddled the y=1024 quadrant boundary.

This module measures whether the inlier support actually spans the frame
in every principal direction. It is a post-fit, fail-closed backstop:
when enabled it may only DOWNGRADE a verdict to DEGENERATE_FAILURE (code
STRIP_COLLAPSE), never upgrade. Frozen gate thresholds are untouched.

2026-10-15 calibration decision: the backstop ships DIAGNOSTIC-ONLY by
default (the measurement is always computed and reported in telemetry).
The 0.25 threshold cleanly separates the TMC-2 pathology (minor_ratio
0.075) from healthy frame-filling sets (0.72), BUT it also trips on two
verified-good fits -- ohrc_01 (0.243, verified SUCCESS_SUBPIXEL 0.328px)
and ohrc_04's main fit (0.19, verified 0.29px via L1 rescue) -- and it
splits a tight cluster of healthy pairs (0.2428 vs 0.260/0.262). Without
independent ground truth for those pairs, a verdict-changing default
cannot be justified. The downgrade is therefore opt-in for experimentation
via CHANDRA_STRIP_GUARD=1, clearly marked uncalibrated. Telemetry always
carries the numbers so analysts see the red flag.

Threshold rationale (principled, not tuned): an inlier cloud whose extent
along any principal direction covers less than one quarter of the frame
in that direction leaves three quarters of the frame as pure
extrapolation -- geometrically it cannot constrain a global 2D transform.
STRIP_SUPPORT_RATIO_MIN = 0.25 is that constant.

Frame-relative normalization is required: raw PCA axis ratios are
confounded by naturally non-square frames (e.g. the OHRC 525x4096
operating frame has raw axis ratio 0.10 even for a healthy, frame-filling
inlier set). All ratios below are normalized by the frame dimensions.

Opt-in: CHANDRA_STRIP_GUARD=1 enables the fail-closed verdict downgrade
(default: diagnostic telemetry only).
"""

import os

import numpy as np

# Principled geometric constant: inliers must span >= 1/4 of the frame
# along both principal directions. Frozen -- do not tune for optics.
STRIP_SUPPORT_RATIO_MIN = 0.25

# Minimum inliers before the support measurement is meaningful.
STRIP_SUPPORT_MIN_POINTS = 6


def strip_guard_enabled():
    """Opt-in: CHANDRA_STRIP_GUARD=1 enables the fail-closed verdict
    downgrade. Default is diagnostic-only telemetry (the measurement is
    always computed; the verdict is never changed unless opted in).

    Rationale (2026-10-15): the 0.25 threshold trips on verified-good
    fits (ohrc_01 at 0.243, ohrc_04 main fit at 0.19), so a default-ON
    downgrade is not justified without wider ground-truth calibration.
    """
    return os.environ.get("CHANDRA_STRIP_GUARD", "0") == "1"


def _convex_hull_area(pts):
    """Convex-hull area via monotone chain (pure numpy; no scipy)."""
    p = np.unique(np.asarray(pts, dtype=np.float64).reshape(-1, 2), axis=0)
    if len(p) < 3:
        return 0.0
    order = np.lexsort((p[:, 1], p[:, 0]))
    p = p[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper = []
    for q in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3:
        return 0.0
    return float(0.5 * abs(
        np.dot(hull[:, 0], np.roll(hull[:, 1], -1))
        - np.dot(hull[:, 1], np.roll(hull[:, 0], -1))
    ))


def spatial_support_report(pts, frame_shape):
    """Measure inlier spatial support relative to the frame.

    Returns a dict with frame-normalized ratios and a `collapsed` flag.
    Non-finite or degenerate inputs fail closed (collapsed=True) only when
    there are enough points to judge; with too few points the report is
    marked invalid and never collapses on its own.
    """
    report = {
        "valid": False,
        "collapsed": False,
        "n_points": 0,
        "minor_ratio": 0.0,
        "major_ratio": 0.0,
        "bbox_coverage_min": 0.0,
        "hull_area_ratio": 0.0,
        "threshold": STRIP_SUPPORT_RATIO_MIN,
    }
    try:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        h, w = float(frame_shape[0]), float(frame_shape[1])
    except (TypeError, ValueError, IndexError):
        report["error"] = "unparseable points or frame shape"
        return report
    n = len(p)
    report["n_points"] = int(n)
    if n < STRIP_SUPPORT_MIN_POINTS or h <= 0 or w <= 0:
        report["error"] = "insufficient points or degenerate frame"
        return report
    if not np.all(np.isfinite(p)):
        report["error"] = "non-finite points"
        return report
    c = p - p.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(c, full_matrices=False)
    except np.linalg.LinAlgError:
        report["error"] = "svd failed"
        return report
    proj = c @ vt.T
    ext = proj.max(axis=0) - proj.min(axis=0)  # [major, minor] extents
    frame_min, frame_max = (h, w) if h <= w else (w, h)
    minor_ratio = float(ext[1] / frame_min) if frame_min > 0 else 0.0
    major_ratio = float(ext[0] / frame_max) if frame_max > 0 else 0.0
    bb = p.max(axis=0) - p.min(axis=0)
    bbox_cov = float(np.min(bb / np.array([w, h]))) \
        if w > 0 and h > 0 else 0.0
    hull_ratio = _convex_hull_area(p) / (h * w) if h * w > 0 else 0.0
    collapsed = bool(
        minor_ratio < STRIP_SUPPORT_RATIO_MIN
        or major_ratio < STRIP_SUPPORT_RATIO_MIN
    )
    report.update({
        "valid": True,
        "collapsed": collapsed,
        "minor_ratio": minor_ratio,
        "major_ratio": major_ratio,
        "bbox_coverage_min": bbox_cov,
        "hull_area_ratio": float(hull_ratio),
    })
    return report


def check_spatial_support(pts, frame_shape):
    """Post-fit strip-collapse check. Returns (ok, report).

    ok is True when support is healthy OR the measurement is invalid
    (invalid measurements never collapse on their own -- the pre-fit
    ABSTAIN codes and inlier-count gate own the too-few-points case).
    ok is False only on a valid, positively-collapsed measurement.
    """
    report = spatial_support_report(pts, frame_shape)
    if not report["valid"]:
        return True, report
    return (not report["collapsed"]), report
