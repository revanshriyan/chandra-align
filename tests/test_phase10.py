"""Phase 10 tests — independent verification layer (synthetic, no torch)."""

import numpy as np
import pytest

from chandra_align.trust.area_check import area_check, calibrate_thresholds
from chandra_align.trust.loop_closure import loop_closure, compose_affine
from chandra_align.trust.gt_hardened import HardenedGT
from chandra_align.testing import make_pair_shift


def test_area_check_verifies_true_transform():
    ref, mov, M_gt = make_pair_shift(shape=(256, 256), dx=5.0, dy=-2.0, seed=0)
    r = area_check(mov, ref, M_gt, grid=(4, 4))
    assert r["overall"] == "verified"
    assert r["verified_fraction"] >= 0.6


def test_area_check_rejects_wrong_transform():
    ref, mov, M_gt = make_pair_shift(shape=(256, 256), dx=5.0, dy=-2.0, seed=0)
    M_bad = M_gt.copy()
    M_bad[:, 2] += 25.0
    r = area_check(mov, ref, M_bad, grid=(4, 4))
    assert r["overall"] != "verified"
    assert r["n_verified"] == 0


def test_calibration_separates():
    cal = calibrate_thresholds()
    assert cal["separates"] is True


def test_loop_closure_perfect():
    M_ab = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 2.0]])
    M_bc = np.array([[1.0, 0.0, -3.0], [0.0, 1.0, 1.0]])
    M_ca = np.array([[1.0, 0.0, -2.0], [0.0, 1.0, -3.0]])
    pts = np.array([[10.0, 10.0], [50.0, 80.0], [100.0, 20.0]])
    r = loop_closure(M_ab, M_bc, M_ca, pts)
    assert r["closed"] is True
    assert r["cyclic_rmse_px"] < 1e-9


def test_loop_closure_detects_drift():
    M_ab = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 2.0]])
    M_bc = np.array([[1.0, 0.0, -3.0], [0.0, 1.0, 1.0]])
    M_ca = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, -3.0]])  # wrong: +12 x-drift
    pts = np.array([[10.0, 10.0], [50.0, 80.0]])
    r = loop_closure(M_ab, M_bc, M_ca, pts)
    assert r["closed"] is False
    assert r["cyclic_rmse_px"] > 5.0


def test_loop_closure_poison_on_failed_leg():
    M_ab = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 2.0]])
    r = loop_closure(M_ab, None, M_ab, np.array([[10.0, 10.0]]))
    assert r["cyclic_rmse_px"] == float("inf")
    assert r["closed"] is False
    assert r["failed_legs"] == ["B->C"]


def test_gt_hardening_bidirectional(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("id,x_ref,y_ref,x_src,y_src,description\n"
                 "p1,10,20,15,22,crater rim\n"
                 "p2,30,40,35,42,bright ejecta\n")
    gt = HardenedGT(p, pair_id="test_pair", direction="src_to_ref",
                    descriptions={"p1": "crater rim NW"})
    assert gt.used_for_fitting is False
    assert len(gt.sha256) == 64
    M = np.array([[1.0, 0.0, -5.0], [0.0, 1.0, -2.0]])  # src->ref
    rep = gt.bidirectional_residuals(M)
    assert rep["rmse_forward_px"] == pytest.approx(0.0, abs=1e-9)
    assert rep["rmse_inverse_px"] == pytest.approx(0.0, abs=1e-9)
    assert rep["per_point"][0]["description"] == "crater rim NW"
    # wrong transform: both directions suffer
    M_bad = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 0.0]])
    rep_bad = gt.bidirectional_residuals(M_bad)
    assert rep_bad["rmse_forward_px"] > 5.0
    assert rep_bad["rmse_inverse_px"] > 5.0


def test_gt_hardening_rejects_fitting_use(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("id,x_ref,y_ref,x_src,y_src\np1,10,20,15,22\n")
    with pytest.raises(AssertionError):
        HardenedGT(p, pair_id="x", used_for_fitting=True)


def test_gt_hardening_rejects_bad_direction(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("id,x_ref,y_ref,x_src,y_src\np1,10,20,15,22\n")
    with pytest.raises(AssertionError):
        HardenedGT(p, pair_id="x", direction="sideways")
