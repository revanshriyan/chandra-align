from types import SimpleNamespace

import numpy as np

from chandra_align.features import (
    grid_feature_indices, select_detector_keypoints, select_grid_keypoints,
)


def test_grid_keypoint_selection_enforces_per_cell_quota_and_descriptor_indices():
    # The crowded top-left cell is capped; a detection in a separate cell remains.
    points = [(10, 10), (11, 10), (60, 60), (20, 20), (200, 200)]
    keypoints = [SimpleNamespace(pt=point, response=score)
                 for point, score in zip(points, [1, 5, 4, 3, 2])]
    selected, indices = select_grid_keypoints(
        keypoints, (400, 400), quota_per_cell=2, grid_shape=(4, 4)
    )
    assert indices.tolist() == [1, 2, 4]
    assert selected == [keypoints[1], keypoints[2], keypoints[4]]


def test_grid_feature_indices_caps_each_of_sixteen_cells():
    pts = np.asarray([[x, y] for y in (10, 110, 210, 310)
                      for x in (10, 110, 210, 310) for _ in range(5)], dtype=float)
    scores = np.arange(len(pts), dtype=float)
    keep = grid_feature_indices(pts, scores, (400, 400), quota_per_cell=2)
    assert len(keep) == 32
    cells = np.floor(pts[keep] / 100).astype(int)
    assert len({(x, y) for x, y in cells}) == 16
    assert all(np.count_nonzero(np.all(cells == cell, axis=1)) == 2
               for cell in {(x, y) for x, y in cells})


def test_grid_feature_selection_handles_empty_and_rejects_misaligned_scores():
    assert grid_feature_indices([], [], (10, 10)).size == 0
    try:
        grid_feature_indices([[1, 1]], [], (10, 10))
    except ValueError as exc:
        assert "one value per point" in str(exc)
    else:
        raise AssertionError("misaligned scores were accepted")


def test_detector_selection_exposes_legacy_baseline_and_4x4_mode(monkeypatch):
    points = [(10 + (i % 5), 10 + (i // 5)) for i in range(65)]
    keypoints = [SimpleNamespace(pt=point, response=float(i))
                 for i, point in enumerate(points)]
    monkeypatch.setenv("CHANDRA_GRID_BUCKETING", "0")
    legacy, _ = select_detector_keypoints(keypoints, (200, 200), quota_per_cell=1)
    monkeypatch.setenv("CHANDRA_GRID_BUCKETING", "1")
    bucketed, _ = select_detector_keypoints(keypoints, (200, 200), quota_per_cell=1)
    assert len(legacy) == 50
    assert len(bucketed) == 1
