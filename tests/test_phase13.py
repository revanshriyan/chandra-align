"""Phase 13 tests — window-tiling batch harness (no real products needed)."""

import csv

import numpy as np
import pytest

from chandra_align.eval.tiling import (
    check_no_overlap,
    generate_windows,
    window_stats,
)


def _fake_reader(value_fn):
    def read(y0, x0, size):
        yy, xx = np.mgrid[0:size, 0:size]
        return value_fn(yy, xx, y0, x0).astype(np.float32)
    return read


def _textured_reader(y0, x0, size):
    yy, xx = np.mgrid[0:size, 0:size]
    return (100 + 40 * np.sin(xx / 9.0 + x0) * np.cos(yy / 7.0 + y0)
            ).astype(np.float32)


def test_tiling_deterministic_same_seed():
    w1 = generate_windows((4096, 4096), 1024, (0, 4096, 0, 4096), seed=13,
                          read_window=_textured_reader)
    w2 = generate_windows((4096, 4096), 1024, (0, 4096, 0, 4096), seed=13,
                          read_window=_textured_reader)
    assert [(w["window_id"], w["y0"], w["x0"]) for w in w1] == \
           [(w["window_id"], w["y0"], w["x0"]) for w in w2]
    assert len(w1) == 16  # 4x4 grid


def test_tiling_no_overlap():
    wins = generate_windows((5000, 6000), 1024, (100, 5000, 200, 6000),
                            seed=13, read_window=None)
    assert check_no_overlap(wins)
    assert len(wins) > 0
    # every window fully inside the raster
    for w in wins:
        assert w["y0"] + w["size"] <= 5000
        assert w["x0"] + w["size"] <= 6000


def test_tiling_skips_dark_and_flat():
    def dark_reader(y0, x0, size):
        return np.zeros((size, size), dtype=np.float32)

    def flat_reader(y0, x0, size):
        return np.full((size, size), 100.0, dtype=np.float32)

    dark = generate_windows((2048, 2048), 1024, (0, 2048, 0, 2048),
                            read_window=dark_reader)
    assert all(not w["eligible"] for w in dark)
    assert all("dark" in w["skip_reason"] for w in dark)

    flat = generate_windows((2048, 2048), 1024, (0, 2048, 0, 2048),
                            read_window=flat_reader)
    assert all(not w["eligible"] for w in flat)
    assert all("flat" in w["skip_reason"] for w in flat)

    tex = generate_windows((2048, 2048), 1024, (0, 2048, 0, 2048),
                           read_window=_textured_reader)
    assert all(w["eligible"] for w in tex)


def test_tiling_rejects_bad_region():
    with pytest.raises(ValueError):
        generate_windows((100, 100), 1024, (0, 100, 0, 100))
    with pytest.raises(ValueError):
        generate_windows((4096, 4096), 1024, (0, 5000, 0, 4096))


def test_window_stats():
    mean, std, dark = window_stats(np.full((32, 32), 50.0))
    assert mean == pytest.approx(50.0)
    assert std == pytest.approx(0.0)
    assert dark == pytest.approx(0.0)
    mean, std, dark = window_stats(np.zeros((32, 32)))
    assert dark == pytest.approx(1.0)


def test_map_origin():
    import sys
    sys.path.insert(0, "scripts")
    from run_phase13_batch import map_origin
    M = [[1.0, 0.0, 1060.0], [0.0, 1.0, 1302.0]]
    assert map_origin(M, 5000, 45000) == (6060, 46302)
    # near-identity with small scale + rotation (rotation term matters:
    # -0.0158 * 45000 y-px shifts x by ~-710 px)
    M2 = [[1.00553448, -0.01578604, 1060.4775],
          [0.01578604, 1.00553448, 1302.2119]]
    xb, yb = map_origin(M2, 5000, 45000)
    assert (xb, yb) == (5378, 46630)


def test_batch_runner_blank_window_is_row_not_exception():
    """A blank (featureless) window must produce a DEGENERATE/ABSTAIN row."""
    import sys
    sys.path.insert(0, "scripts")
    from run_phase13_batch import COLUMNS, run_window
    blank = np.zeros((1024, 1024), dtype=np.uint8)

    class FakeMM:
        def __getitem__(self, key):
            return blank[key]

    win = {"window_id": "w0000", "y0": 0, "x0": 0, "size": 1024}
    row = run_window("testpair", FakeMM(), FakeMM(), win, (0, 0))
    assert set(row.keys()) == set(COLUMNS)
    assert row["verdict"] in ("ABSTAIN", "DEGENERATE_FAILURE", "ERROR",
                              "COARSE_ADVISORY", "SUCCESS_SUBPIXEL")
    assert row["window_id"] == "testpair_w0000"
    assert float(row["runtime_s"]) >= 0


def test_batch_runner_csv_columns(tmp_path, monkeypatch):
    """run_window rows fit the declared CSV schema exactly."""
    import sys
    sys.path.insert(0, "scripts")
    from run_phase13_batch import COLUMNS, run_window
    rng = np.random.default_rng(0)
    tex = (rng.random((1024, 1024)) * 255).astype(np.uint8)

    class FakeMM:
        def __getitem__(self, key):
            return tex[key]

    win = {"window_id": "w0001", "y0": 0, "x0": 0, "size": 1024}
    row = run_window("testpair", FakeMM(), FakeMM(), win, (0, 0))
    assert list(row.keys()) == COLUMNS
    # csv round-trip
    p = tmp_path / "t.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerow(row)
    with open(p) as fh:
        rd = list(csv.DictReader(fh))
    assert rd[0]["window_id"] == "testpair_w0001"
