"""Phase 18 — behavioral test matrix (CheckList-style).

Capability x test-type matrix over the FROZEN default pipeline. Each test
runs a real cell from chandra_align.eval.behavioral and asserts the
expectation held. No gate, threshold, or pipeline behavior is changed here.
"""

import pytest

from chandra_align.eval.behavioral import CELLS, run_matrix, summarize


def _cell_by_name(name):
    for cname, cap, typ, fn in CELLS:
        if cname == name:
            return fn
    raise KeyError(name)


def test_matrix_has_minimum_coverage():
    assert len(CELLS) >= 12
    caps = {c for _, c, _, _ in CELLS}
    typs = {t for _, _, t, _ in CELLS}
    assert {"rotation", "illumination", "cross-modal", "low-texture",
            "uniform-shift"} <= caps
    assert {"invariance", "directional", "minimum-functionality"} <= typs


def test_rotation_minimum_functionality():
    r = _cell_by_name("rotation / minimum-functionality")()
    assert r["expectation_met"], r["measured"]


def test_illumination_directional_sunflip():
    # The Phase 11 lesson as a live regression check: inlier RMSE must not
    # be mistaken for correctness under polarity reversal.
    r = _cell_by_name("illumination / directional")()
    assert r["expectation_met"], r["measured"]


def test_xmodal_directional_polarity():
    # Polarity inversion must never be confidently accepted.
    r = _cell_by_name("cross-modal / directional")()
    assert r["expectation_met"], r["measured"]


def test_lowtexture_directional_degenerate():
    # Featureless input must abstain with a named code, never a verdict.
    r = _cell_by_name("low-texture / directional")()
    assert r["expectation_met"], r["measured"]


def test_shift_directional_loop_closure():
    # Clean loop closes; a corrupted leg is caught.
    r = _cell_by_name("uniform-shift / directional")()
    assert r["expectation_met"], r["measured"]


def test_full_matrix_runs_clean():
    recs = run_matrix()
    s = summarize(recs)
    failed = [r["cell"] for r in recs if not r.get("expectation_met")]
    assert s["pass_rate"] >= 0.8, {"failed_cells": failed, "summary": s}
