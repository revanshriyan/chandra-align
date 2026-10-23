"""Unit tests for the opt-in arm-vs-fallback point-set selector."""

import numpy as np

from chandra_align.matching.arbitration import arbitrate_point_sets


def _consistent_pair(n=60, seed=3):
    """Correspondences related by a known similarity transform + noise."""
    rng = np.random.default_rng(seed)
    sec = rng.uniform(100, 1900, size=(n, 2))
    angle = np.deg2rad(2.0)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    ref = sec @ rot.T + np.array([30.0, -45.0]) + rng.normal(0, 0.3, size=(n, 2))
    return ref, sec


def _rotated_wrong_pair(n=60, seed=3):
    """Same source points and count, but 60% of the targets are mismatched
    (permuted): the failure mode of a bad matcher — correspondences that do
    not follow any single transform. (A coherently transformed wrong set
    would still be one similarity, which a held-out affine score cannot
    and does not claim to detect.)"""
    ref, sec = _consistent_pair(n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    ref = ref.copy()
    idx = rng.choice(n, size=int(0.6 * n), replace=False)
    ref[idx] = ref[rng.permutation(idx)]
    return ref, sec


def test_selector_picks_consistent_set_over_confidently_wrong():
    good = _consistent_pair()
    bad = _rotated_wrong_pair()
    winner, info = arbitrate_point_sets(bad, good)  # arm=bad, fallback=good
    assert winner == "fallback"
    assert info["fallback_score_px"] < info["arm_score_px"]


def test_selector_keeps_good_arm():
    good = _consistent_pair()
    bad = _rotated_wrong_pair()
    winner, _ = arbitrate_point_sets(good, bad)  # arm=good, fallback=bad
    assert winner == "arm"


def test_selector_is_deterministic():
    good = _consistent_pair()
    bad = _rotated_wrong_pair()
    first = arbitrate_point_sets(bad, good)
    second = arbitrate_point_sets(bad, good)
    assert first == second


def test_small_sets_cannot_win():
    good = _consistent_pair()
    tiny_ref, tiny_sec = good[0][:5], good[1][:5]
    winner, info = arbitrate_point_sets((tiny_ref, tiny_sec), good)
    assert winner == "fallback"
    assert info["arm_score_px"] == float("inf")


def test_both_unusable_keeps_arm():
    ref, sec = _consistent_pair()
    tiny = (ref[:4], sec[:4])
    winner, _ = arbitrate_point_sets(tiny, tiny)
    assert winner == "arm"
