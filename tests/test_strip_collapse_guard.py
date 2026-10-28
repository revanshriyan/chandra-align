"""Tests for the post-RANSAC strip-collapse guard (audit item #1 follow-up).

The guard measures whether the inlier set collapsed to a strip (e.g. the
TMC-2 fore/nadir case: 13 inliers in ref y in [944,1096] of a 2048 frame).
The measurement is always available for diagnostic telemetry; the
fail-closed verdict downgrade (code STRIP_COLLAPSE) is DEFAULT-ON since
the 2026-10-24 human-landmark calibration (opt-out via
CHANDRA_STRIP_GUARD=0). The calibrated 0.159 threshold separates the
known-collapsed anchor (minor_ratio 0.075) from every verified-good pair
(lowest 0.1926). The downgrade may only ever demote to
DEGENERATE_FAILURE, never upgrade.
"""

import os

import numpy as np
import pytest

from chandra_align.metrics import validate_registration_gate
from chandra_align.metrics.spatial_support import (
    STRIP_SUPPORT_RATIO_MIN,
    check_spatial_support,
    spatial_support_report,
    strip_guard_enabled,
)


@pytest.fixture()
def restore_env():
    old = os.environ.get("CHANDRA_STRIP_GUARD")
    yield
    if old is None:
        os.environ.pop("CHANDRA_STRIP_GUARD", None)
    else:
        os.environ["CHANDRA_STRIP_GUARD"] = old


def _strip_points(n=40, seed=0):
    """Thin horizontal strip in a 2048x2048 frame (TMC-2-like)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(300, 1900, n)
    y = rng.uniform(944, 1096, n)
    return np.column_stack([x, y])


def _spread_points(n=200, seed=1, frame=(2048, 2048)):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(0, frame[1], n), rng.uniform(0, frame[0], n)])


def test_metric_fires_on_strip():
    p = _strip_points()
    rep = spatial_support_report(p, (2048, 2048))
    assert rep["valid"]
    assert rep["collapsed"] is True
    assert rep["minor_ratio"] < STRIP_SUPPORT_RATIO_MIN
    ok, _ = check_spatial_support(p, (2048, 2048))
    assert ok is False


def test_metric_silent_on_well_spread_square_frame():
    p = _spread_points()
    rep = spatial_support_report(p, (2048, 2048))
    assert rep["valid"]
    assert rep["collapsed"] is False
    assert rep["minor_ratio"] >= STRIP_SUPPORT_RATIO_MIN
    ok, _ = check_spatial_support(p, (2048, 2048))
    assert ok is True


def test_metric_silent_on_tall_thin_filled_frame():
    # Frame-relative normalization: a healthy inlier set filling a
    # naturally tall-thin frame (OHRC 525x4096 operating frame) must NOT
    # collapse even though its raw PCA axis ratio is ~0.10.
    p = _spread_points(frame=(4096, 525))
    rep = spatial_support_report(p, (4096, 525))
    assert rep["valid"]
    assert rep["collapsed"] is False


def test_metric_fires_on_diagonal_strip():
    # BBox covers the frame but the cloud is 1-D: PCA must catch it.
    rng = np.random.default_rng(3)
    t = rng.uniform(0, 2000, 60)
    p = np.column_stack([t, t + rng.normal(0, 8, 60)])
    rep = spatial_support_report(p, (2048, 2048))
    assert rep["valid"]
    assert rep["collapsed"] is True


def test_guard_downgrades_collapsed_fit(restore_env):
    os.environ["CHANDRA_STRIP_GUARD"] = "1"  # explicitly on
    assert strip_guard_enabled() is True
    p = _strip_points()
    # Would-be COARSE: 1.77px, 13 inliers, entropy 1.95, 4/4 quads.
    msg, code = validate_registration_gate(
        1.77, 13, 8, 1.95,
        {"Q1": 4, "Q2": 4, "Q3": 3, "Q4": 2},
        pts=p, frame_shape=(2048, 2048))
    assert code == "DEGENERATE_FAILURE"
    assert "STRIP_COLLAPSE" in msg


def test_guard_downgrades_would_be_subpixel_too(restore_env):
    # Fail-closed applies regardless of tier: a collapsed "sub-pixel" fit
    # is still rejected when the guard is opted in.
    os.environ["CHANDRA_STRIP_GUARD"] = "1"
    p = _strip_points()
    _, code = validate_registration_gate(
        0.40, 13, 8, 1.90,
        {"Q1": 4, "Q2": 4, "Q3": 3, "Q4": 2},
        pts=p, frame_shape=(2048, 2048))
    assert code == "DEGENERATE_FAILURE"


def test_guard_leaves_good_fit_untouched(restore_env):
    os.environ["CHANDRA_STRIP_GUARD"] = "1"  # guard on, healthy set
    p = _spread_points()
    msg, code = validate_registration_gate(
        0.40, 200, 8, 1.90,
        {"Q1": 50, "Q2": 50, "Q3": 50, "Q4": 50},
        pts=p, frame_shape=(2048, 2048))
    assert code == "SELF_CONSISTENT_SUBPIXEL"
    assert "STRIP_COLLAPSE" not in msg


def test_default_is_guard_on(restore_env):
    # Since the 2026-10-24 human-landmark calibration, the guard is
    # DEFAULT-ON: without any env setting, a collapsed set IS downgraded.
    os.environ.pop("CHANDRA_STRIP_GUARD", None)
    assert strip_guard_enabled() is True
    p = _strip_points()
    _, code = validate_registration_gate(
        1.77, 13, 8, 1.95,
        {"Q1": 4, "Q2": 4, "Q3": 3, "Q4": 2},
        pts=p, frame_shape=(2048, 2048))
    assert code == "DEGENERATE_FAILURE"


def test_explicit_zero_is_diagnostic_only(restore_env):
    os.environ["CHANDRA_STRIP_GUARD"] = "0"
    assert strip_guard_enabled() is False


def test_old_callers_without_pts_unaffected(restore_env):
    os.environ.pop("CHANDRA_STRIP_GUARD", None)
    _, code = validate_registration_gate(
        1.77, 13, 8, 1.95, {"Q1": 4, "Q2": 4, "Q3": 3, "Q4": 2})
    assert code == "COARSE_ADVISORY"


def test_too_few_points_never_collapse():
    p = np.array([[100.0, 100.0], [101.0, 100.5], [100.5, 101.0]])
    rep = spatial_support_report(p, (2048, 2048))
    assert rep["valid"] is False
    ok, _ = check_spatial_support(p, (2048, 2048))
    assert ok is True  # invalid measurement never collapses on its own


def test_threshold_is_calibrated_constant():
    # The threshold is the measured midpoint of the 2026-10-24
    # human-landmark calibration (results/table_strip_guard_calibration_2026-10-24.csv),
    # not a tuned value: (0.075 + 0.2428) / 2 = 0.1589 -> 0.159, the
    # midpoint of the rig's LABELED separation (collapsed anchor 0.075
    # vs labeled-good ohrc_01 0.2428). ohrc_04 (0.1926) is unlabeled
    # under the rig's < 2.0 px landmark-RMSE bar and sits above the
    # threshold as a consistency observation, not a calibration point.
    assert STRIP_SUPPORT_RATIO_MIN == 0.159
    assert 0.075 < STRIP_SUPPORT_RATIO_MIN < 0.2428
    assert STRIP_SUPPORT_RATIO_MIN < 0.1926  # unlabeled ohrc_04 also above
