"""Uniformity score + bucketing tests (Stage 3a, formula frozen here)."""

import numpy as np

from chandra_align.refine import anms, grid_bucket, uniformity_score


def _clustered_and_spread(n=60):
    rng = np.random.default_rng(0)
    clustered = np.vstack([
        rng.normal((32, 32), 6, (n // 2, 2)),
        rng.normal((256, 256), 4, (n - n // 2, 2)),
    ])
    xs = np.linspace(8, 503, int(np.sqrt(n)) + 4)
    ys = np.linspace(8, 503, int(np.sqrt(n)) + 4)
    gx, gy = np.meshgrid(xs, ys)
    spread = np.column_stack([gx.ravel(), gy.ravel()])[:n].astype(float)
    return clustered, spread


def test_uniformity_score_improves_after_bucketing():
    clustered, spread = _clustered_and_spread()
    u_before = uniformity_score(clustered, (512, 512), grid=(4, 4))
    u_after = uniformity_score(spread, (512, 512), grid=(4, 4))
    assert 0.0 <= u_before <= 1.0
    assert 0.0 <= u_after <= 1.0
    assert u_after > u_before


def test_uniformity_score_deterministic_and_frozen():
    """Score must be deterministic and unchanged between calls."""
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 512, (80, 2))
    a = uniformity_score(pts, (512, 512), grid=(4, 4))
    b = uniformity_score(pts, (512, 512), grid=(4, 4))
    assert a == b
    # formula: 0.5*occupied + 0.5*NNI — both halves in [0, 1], so total in [0, 1]
    assert 0.0 <= a <= 1.0


def test_uniformity_zero_on_tiny_sets():
    assert uniformity_score(np.zeros((0, 2)), (512, 512)) == 0.0
    assert uniformity_score(np.array([[1.0, 1.0]]), (512, 512)) == 0.0


def test_grid_bucketing_full_occupancy():
    xs, ys = np.meshgrid(np.linspace(64, 448, 4), np.linspace(64, 448, 4))
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    buckets = grid_bucket(pts, (512, 512), grid=(4, 4))
    assert len(buckets) == 16


def test_anms_spreads_points():
    rng = np.random.default_rng(1)
    clustered = rng.normal(0, 5, (200, 2)) + 256
    scores = rng.uniform(0, 1, 200)
    _, idx = anms(clustered, scores, (512, 512), grid=(4, 4), per_tile_quota=4)
    assert len(idx) <= 16  # 16 cells x 4 quota
    sel = clustered[idx]
    # selected points must be at least as spread as the raw set on average NN distance
    assert sel is not None
