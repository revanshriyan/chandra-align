"""Phase 15 — joint 2D pose-graph optimization tests.

All synthetic (no real-data triplet exists in the repo; the driver ABSTAINS
on real data). Deterministic: fixed seeds, no randomness in the optimizer.
"""

import numpy as np
import pytest

from chandra_align.optimize.posegraph import (
    GATED_VERDICTS,
    affine_inverse,
    check_gated,
    optimize_posegraph,
)


def rt_affine(deg, tx, ty):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, tx], [s, c, ty]])


def true_M(T_true, i, j):
    Hi = np.eye(3); Hi[:2, :] = T_true[i]
    Hj = np.eye(3); Hj[:2, :] = T_true[j]
    return (Hj @ np.linalg.inv(Hi))[:2, :]


@pytest.fixture()
def triplet():
    rng = np.random.default_rng(77)
    T_true = [np.eye(2, 3), rt_affine(1.5, 40.0, -25.0),
              rt_affine(-2.0, -60.0, 80.0)]
    edges = []
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        M = true_M(T_true, i, j).copy()
        M[:, 2] += rng.normal(0, 2.0, 2)      # translation noise (px)
        M[:, :2] += rng.normal(0, 0.002, (2, 2))  # linear noise
        edges.append({"i": i, "j": j, "M": M, "weight": 1.0,
                      "verdict": "COARSE_ADVISORY"})
    return T_true, edges


def test_misclosure_collapses(triplet):
    _T_true, edges = triplet
    res = optimize_posegraph(edges, 3)
    before, after = res["misclosure_before_px"], res["misclosure_after_px"]
    assert before is not None and after is not None
    assert before > 0.5, "test needs a nontrivial initial misclosure"
    assert after < 0.5 * before
    assert after < 1.0


def test_image0_pinned(triplet):
    _T_true, edges = triplet
    res = optimize_posegraph(edges, 3)
    T0 = np.asarray(res["T"][0])
    np.testing.assert_allclose(T0, np.eye(2, 3), atol=0)


def test_huber_robust_to_one_bad_edge(triplet):
    T_true, edges = triplet
    # A single 3-cycle cannot isolate a fault (the 30 px error must live on
    # SOME edge; Huber only caps its influence). Give the graph redundancy:
    # two independent measurements of leg 1->2, one good and one
    # confident-but-wrong (+30 px, still gated). The bad edge must lose the
    # vote and be flagged by a large per-edge residual.
    bad = dict(edges[1])
    M_bad = np.asarray(bad["M"]).copy()
    M_bad[:, 2] += 30.0
    bad["M"] = M_bad
    four = [edges[0], edges[1], bad, edges[2]]
    # Huber scale above the inlier noise (translation sigma 2 px) and well
    # below the 30 px corruption.
    res = optimize_posegraph(four, 3, huber_scale=5.0)
    g = np.linspace(-400, 400, 6)
    yy, xx = np.meshgrid(g, g, indexing="ij")
    probes = np.stack([xx.ravel(), yy.ravel()], axis=1)
    errs = []
    for k in range(3):
        A = np.asarray(res["T"][k])
        pa = probes @ A[:, :2].T + A[:, 2]
        pb = probes @ T_true[k][:, :2].T + T_true[k][:, 2]
        errs.append(float(np.hypot(*(pa - pb).T).mean()))
    # Huber must keep the solution near truth despite the bad edge.
    assert max(errs) < 5.0, f"bad edge poisoned the solution: {errs}"
    # The bad edge (index 2) must show a large per-edge residual (it is
    # downweighted, not silently absorbed); the good duplicate stays small.
    assert res["per_edge_rmse_px"][2] > 10.0
    assert res["per_edge_rmse_px"][1] < 5.0


def test_ungated_input_refused(triplet):
    _T_true, edges = triplet
    for bad_verdict in ["DEGENERATE_FAILURE", "ABSTAIN", None, "SUCCESS"]:
        bad = dict(edges[0]); bad["verdict"] = bad_verdict
        with pytest.raises(ValueError):
            check_gated([bad, edges[1], edges[2]])
        with pytest.raises(ValueError):
            optimize_posegraph([bad, edges[1], edges[2]], 3)
    bad_m = dict(edges[0]); bad_m["M"] = None
    with pytest.raises(ValueError):
        optimize_posegraph([bad_m, edges[1], edges[2]], 3)


def test_deterministic(triplet):
    _T_true, edges = triplet
    r1 = optimize_posegraph(edges, 3)
    # shuffle input order: chaining init sorts, result must not change
    r2 = optimize_posegraph([edges[2], edges[0], edges[1]], 3)
    for a, b in zip(r1["T"], r2["T"]):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-9)


def test_affine_inverse_roundtrip():
    M = rt_affine(3.0, 12.5, -7.25)
    Mi = affine_inverse(M)
    H = np.eye(3); H[:2, :] = M
    Hi = np.eye(3); Hi[:2, :] = Mi
    np.testing.assert_allclose(H @ Hi, np.eye(3), atol=1e-12)


def test_consistent_triplet_barely_moves():
    # Perfectly consistent edges: optimizer should return ~chained init.
    T_true = [np.eye(2, 3), rt_affine(1.0, 10.0, 5.0), rt_affine(-1.0, -8.0, 3.0)]
    edges = [{"i": i, "j": j, "M": true_M(T_true, i, j), "weight": 1.0,
              "verdict": "SUCCESS_SUBPIXEL"} for i, j in [(0, 1), (1, 2), (2, 0)]]
    res = optimize_posegraph(edges, 3)
    assert res["misclosure_after_px"] < 1e-6
    for got, want in zip(res["T"], T_true):
        np.testing.assert_allclose(np.asarray(got), want, atol=1e-6)


def test_gated_verdicts_are_the_frozen_two():
    assert GATED_VERDICTS == {"SUCCESS_SUBPIXEL", "COARSE_ADVISORY"}
