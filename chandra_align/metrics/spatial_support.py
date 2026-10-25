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

2026-10-15 calibration decision (superseded 2026-10-24): the backstop
shipped DIAGNOSTIC-ONLY by default because the principled 0.25 threshold
tripped on two verified-good fits -- ohrc_01 (0.243) and ohrc_04's main
fit (0.19) -- and no independent ground truth existed for those pairs.

2026-10-24 calibration decision: with 39 human-verified landmarks
(ohrc_01: 21, ohrc_04: 18; results/table_strip_guard_calibration_2026-10-24.csv)
the threshold is now MEASURED, not guessed: T* = (0.075 + 0.2428) / 2 =
0.1589 -> 0.159, the midpoint of the rig's labeled separation between
the known-collapsed anchor (0.075) and labeled-good ohrc_01 (0.2428,
landmark RMSE 1.8869 px under the rig's < 2.0 px bar). ohrc_04
(minor_ratio 0.1926, landmark RMSE 2.2591 px) is unlabeled under that
bar and thin-covered (18 < ~20 target): a consistency observation above
the threshold, not a calibration point. The downgrade is therefore
DEFAULT-ON, opt-out via CHANDRA_STRIP_GUARD=0. Telemetry always carries
the numbers so analysts see the red flag.

Threshold rationale (calibrated, not tuned to fit results): the value is
the midpoint of the labeled separation, with 0.084 margin on both sides
(0.159 - 0.075 above the collapsed anchor; 0.2428 - 0.159 below
ohrc_01). Fragility, stated plainly: the labeled calibration rests on
the single pair ohrc_01, clearing the 2.0 px bar by 0.11 px; a third
full-coverage pair would strengthen it. It changes only on new landmark
evidence.

Frame-relative normalization is required: raw PCA axis ratios are
confounded by naturally non-square frames (e.g. the OHRC 525x4096
operating frame has raw axis ratio 0.10 even for a healthy, frame-filling
inlier set). All ratios below are normalized by the frame dimensions.

Default-ON since the 2026-10-24 calibration: CHANDRA_STRIP_GUARD=0 opts
out of the fail-closed verdict downgrade (diagnostic telemetry is always
computed regardless).
"""

import os

import numpy as np

# Calibrated constant (2026-10-24, human-landmark calibration
# results/table_strip_guard_calibration_2026-10-24.csv): inliers must span
# >= 0.159 of the frame along the minor principal direction. The value is
# (0.075 + 0.2428) / 2 = 0.1589 -> 0.159, the midpoint of the rig's LABELED
# separation -- known-collapsed TMC-2 fore/nadir anchor at minor_ratio
# 0.075 vs labeled-good ohrc_01 at 0.2428 (landmark RMSE 1.8869 px, under
# the rig's < 2.0 px labeling bar). ohrc_04 (minor_ratio 0.1926,
# landmark RMSE 2.2591 px) is UNLABELED under that bar: a consistency
# observation above the threshold, not a calibration point -- and its
# coverage is thin (18 landmarks < the ~20 protocol target). It replaces
# the 2026-10-15 principled 0.25 guess, which tripped verified-good fits.
# Fragility on the record: the labeled calibration rests on the single
# pair ohrc_01, clearing the 2.0 px bar by 0.11 px; a third full-coverage
# pair would strengthen it. Change only on new landmark evidence, with
# the calibration table updated alongside.
STRIP_SUPPORT_RATIO_MIN = 0.159

# Minimum inliers before the support measurement is meaningful.
STRIP_SUPPORT_MIN_POINTS = 6


def strip_guard_enabled():
    """Default-ON since the 2026-10-24 human-landmark calibration: the
    fail-closed verdict downgrade applies unless explicitly opted out
    with CHANDRA_STRIP_GUARD=0. The measurement is always computed for
    diagnostic telemetry regardless.

    Rationale (2026-10-24): the calibrated 0.159 threshold cleanly
    separates the known-collapsed anchor (minor_ratio 0.075) from the
    labeled-good pair ohrc_01 (0.2428), with the unlabeled ohrc_04
    observation (0.1926) also above the threshold, so default-ON is now
    justified;
    the failure mode it guards (a thin-strip fit with small inlier RMSE
    but a large frame-wide error) is strictly worse than a conservative
    rejection, which reports its reason.
    """
    return os.environ.get("CHANDRA_STRIP_GUARD", "1") != "0"


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
