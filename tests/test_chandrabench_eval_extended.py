"""Tests for the Phase 2 extended evaluation in chandrabench/eval.py.

Covers the bootstrap CI (determinism under a fixed seed, exact interval on
constant residuals), the error-distribution percentiles on a toy set, the
abstain path for below-bar pairs, and the repeat-noise null / hand-computed
cases. The v0.1 default report shape must be unchanged.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_EVAL_PATH = Path(__file__).resolve().parent.parent / "chandrabench" / "eval.py"
_spec = importlib.util.spec_from_file_location("chandrabench_eval", _EVAL_PATH)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def _grid(nx, ny, step=60.0):
    """Well-spread landmark grid: spacing = step, all quadrants populated."""
    pts = np.array([[x * step, y * step]
                    for y in range(ny) for x in range(nx)], dtype=float)
    ids = [f"lm-{i:02d}" for i in range(len(pts))]
    return ids, pts


# ---------------------------------------------------------------- bootstrap

def test_bootstrap_deterministic_under_fixed_seed():
    res = np.linspace(0.1, 2.0, 30)
    first = ev.bootstrap_rmse_ci(res, seed=123, n_resamples=1000)
    second = ev.bootstrap_rmse_ci(res, seed=123, n_resamples=1000)
    assert first == second
    lo, hi = first
    assert 0.0 <= lo <= hi


def test_bootstrap_constant_residuals_exact_interval():
    lo, hi = ev.bootstrap_rmse_ci(np.full(25, 2.0), seed=7, n_resamples=1000)
    assert lo == pytest.approx(2.0)
    assert hi == pytest.approx(2.0)


def test_bootstrap_rejects_too_few_resamples():
    with pytest.raises(ValueError, match=">= 1000"):
        ev.bootstrap_rmse_ci(np.ones(10), n_resamples=500)


# ------------------------------------------------------- error distribution

def test_error_distribution_toy_set():
    stats = ev.error_distribution(np.arange(10, dtype=float))
    assert stats["error_median_px"] == pytest.approx(4.5)
    assert stats["error_p90_px"] == pytest.approx(8.1)
    assert stats["error_max_px"] == pytest.approx(9.0)


# ------------------------------------------------------------- repeat noise

def test_repeat_noise_null_without_repeat_of():
    ids, ref = _grid(3, 3)
    out = ev.repeat_noise(ids, ref, ref.copy(), None)
    assert out["repeat_noise_px"] is None
    assert "repeat_of" in out["reason"]


def test_repeat_noise_null_when_no_group_repeats():
    ids, ref = _grid(3, 3)
    out = ev.repeat_noise(ids, ref, ref.copy(), [""] * len(ids))
    assert out["repeat_noise_px"] is None
    assert out["n_repeat_groups"] == 0


def test_repeat_noise_hand_computed():
    # One feature clicked twice. Ref clicks (0,0),(2,0): centroid (1,0),
    # distances 1,1. Src clicks (10,10),(10,14): centroid (10,12),
    # distances 2,2. Pooled RMS = sqrt((1+1+4+4)/4) = sqrt(2.5).
    ids = ["lm-01", "lm-01b"]
    ref = np.array([[0.0, 0.0], [2.0, 0.0]])
    src = np.array([[10.0, 10.0], [10.0, 14.0]])
    out = ev.repeat_noise(ids, ref, src, ["", "lm-01"])
    assert out["n_repeat_groups"] == 1
    assert out["n_repeat_clicks"] == 2
    assert out["repeat_noise_px"] == pytest.approx(np.sqrt(2.5))
    assert out["reason"] is None


# ------------------------------------------------------------------- abstain

def test_abstain_on_count_below_bar_only():
    # 20 landmarks on a clean 5x4 grid (spacing 60, quadrants 4/6/4/6,
    # non-collinear): the only failing bar item is the count.
    ids, ref = _grid(5, 4)
    block = ev.extended_block(ids, ref, ref.copy(), np.full(len(ids), 0.5))
    assert block["status"] == "ABSTAIN"
    assert block["metrics"] is None
    assert block["abstain_reasons"] == ["landmark count 20 < 25"]


def test_abstain_collinear_reports_reason():
    pts = np.array([[i * 60.0, i * 60.0] for i in range(30)])
    ids = [f"lm-{i:02d}" for i in range(30)]
    _, reasons, detail = ev.phase2_bar(ids, pts, pts.copy())
    assert detail["collinear_ref"] is True
    assert any("collinear" in r for r in reasons)


def test_abstain_spacing_below_bar():
    # 30 landmarks crammed into a 40x40 px patch: spacing fails.
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 40.0, size=(30, 2))
    ids = [f"lm-{i:02d}" for i in range(30)]
    _, reasons, detail = ev.phase2_bar(ids, pts, pts.copy())
    assert detail["min_spacing_ref_px"] < 50.0
    assert any("spacing" in r for r in reasons)


# ------------------------------------------------------------------ OK path

def test_ok_block_carries_full_metrics():
    ids, ref = _grid(6, 5)  # 30 landmarks, quadrants 6/6/9/9, spacing 60
    res = np.linspace(0.1, 1.0, len(ids))
    block = ev.extended_block(ids, ref, ref.copy(), res, repeat_of=None)
    assert block["status"] == "OK"
    assert block["abstain_reasons"] == []
    m = block["metrics"]
    assert m["rmse_forward_px"] == pytest.approx(float(np.sqrt((res ** 2).mean())))
    lo, hi = m["rmse_ci95_px"]
    assert 0.0 <= lo <= hi
    assert m["bootstrap"]["n_resamples"] >= 1000
    assert m["error_median_px"] == pytest.approx(float(np.median(res)))
    assert m["error_max_px"] == pytest.approx(1.0)
    # No repeat data on this pair: null + reason, never a borrowed number.
    assert m["repeat_noise"]["repeat_noise_px"] is None
    assert m["repeat_noise"]["reason"]
    # Determinism: a second call reproduces the interval exactly.
    again = ev.extended_block(ids, ref, ref.copy(), res, repeat_of=None)
    assert again["metrics"]["rmse_ci95_px"] == m["rmse_ci95_px"]


def test_evaluate_table_scores_and_extends():
    ids, ref = _grid(6, 5)
    src = ref + np.array([3.0, 4.0])  # constant offset -> every residual 5 px
    out = ev.evaluate_table(ids, ref, src, np.array([1, 0, 0, 0, 1, 0], float))
    assert out["rmse_forward_px"] == pytest.approx(5.0)
    assert out["phase2_extended"]["status"] == "OK"
    assert out["phase2_extended"]["metrics"]["error_median_px"] == pytest.approx(5.0)


# ------------------------------------------------- registry integration

def test_evaluate_default_shape_unchanged_and_extended_abstains(tmp_path, monkeypatch):
    # Register a synthetic 20-landmark pair with its true sha, so evaluate()
    # runs through the real registry path. (The v0.1 pins in PAIRS do not
    # match the on-disk v0.1 CSVs on this base — a pre-existing condition,
    # verified against HEAD — so the frozen pairs cannot be loaded here.)
    ids, ref = _grid(5, 4)
    src = ref + np.array([3.0, 4.0])
    lines = ["id,x_ref,y_ref,x_src,y_src"]
    lines += [f"{i},{r[0]},{r[1]},{s[0]},{s[1]}"
              for i, r, s in zip(ids, ref, src)]
    csv_path = tmp_path / "tmp_pair.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(ev, "HERE", tmp_path)
    monkeypatch.setitem(ev.PAIRS, "tmp_pair", {
        "csv": "tmp_pair.csv",
        "sha256": ev.sha256_of(csv_path),
        "n_points": len(ids),
    })
    identity = np.array([1, 0, 0, 0, 1, 0], float)
    base = ev.evaluate("tmp_pair", identity, True)
    explicit = ev.evaluate("tmp_pair", identity, True, extended=False)
    assert set(base.keys()) == set(explicit.keys())
    assert "phase2_extended" not in base
    ext = ev.evaluate("tmp_pair", identity, True, extended=True)
    assert set(ext.keys()) == set(base.keys()) | {"phase2_extended"}
    # 20 landmarks: below the Phase 2 bar, and no repeat_of column.
    block = ext["phase2_extended"]
    assert block["status"] == "ABSTAIN"
    assert block["metrics"] is None
    assert block["abstain_reasons"] == ["landmark count 20 < 25"]
    assert block["n_landmarks"] == 20


# ------------------------------------------------------------ CSV loading

def test_load_landmark_csv_with_and_without_repeat_of(tmp_path):
    with_rep = tmp_path / "with_rep.csv"
    with_rep.write_text(
        "id,x_ref,y_ref,x_src,y_src,repeat_of\n"
        "a,0,0,1,1,\n"
        "b,2,0,3,1,a\n",
        encoding="utf-8")
    ids, ref, src, repeat_of = ev.load_landmark_csv(with_rep)
    assert ids == ["a", "b"]
    assert repeat_of == ["", "a"]
    assert ref.shape == (2, 2)

    without_rep = tmp_path / "without_rep.csv"
    without_rep.write_text(
        "id,x_ref,y_ref,x_src,y_src\na,0,0,1,1\n", encoding="utf-8")
    _, _, _, repeat_of2 = ev.load_landmark_csv(without_rep)
    assert repeat_of2 is None
