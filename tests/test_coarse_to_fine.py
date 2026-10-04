import numpy as np

from chandra_align.matching.coarse_to_fine import coarse_to_fine_match


def test_coarse_to_fine_propagates_affine_and_returns_original_coordinates():
    image = np.zeros((64, 64), dtype=np.uint8)
    calls = []

    def fake_matcher(reference, source, threshold):
        calls.append(reference.shape)
        if reference.shape == (16, 16):
            pts_ref = np.array([[4, 4], [12, 4], [4, 12], [12, 12]], np.float32)
            # At 1/4 scale this is a (3, -2)-pixel residual, or (12, -8) full scale.
            pts_source_warped = pts_ref - np.array([3, -2], np.float32)
        elif reference.shape == (32, 32):
            pts_ref = np.array([[8, 8], [24, 8], [8, 24], [24, 24]], np.float32)
            pts_source_warped = pts_ref.copy()
        else:
            pts_ref = np.array([[20, 20], [40, 20], [20, 40], [40, 40]], np.float32)
            pts_source_warped = pts_ref.copy()
        return pts_ref, pts_source_warped, "fake", {"status": "ok"}

    pts_ref, pts_source, engine, diagnostics = coarse_to_fine_match(image, image, fake_matcher)

    assert calls == [(16, 16), (32, 32), (64, 64)]
    assert engine == "fake"
    assert diagnostics["coarse_to_fine"] is True
    assert all(stage["refined"] for stage in diagnostics["pyramid_levels"])
    matrix = np.asarray(diagnostics["coarse_to_fine_matrix"])
    assert np.allclose(matrix[:, 2], [12, -8], atol=0.1)
    projected = pts_source @ matrix[:, :2].T + matrix[:, 2]
    assert np.allclose(projected, pts_ref, atol=0.1)


def test_coarse_to_fine_requires_three_ascending_levels_ending_at_full_resolution():
    image = np.zeros((8, 8), dtype=np.uint8)
    def unused(*args):
        raise AssertionError("invalid levels must fail before matching")

    try:
        coarse_to_fine_match(image, image, unused, levels=(0.5, 1.0))
    except ValueError as exc:
        assert "exactly three" in str(exc)
    else:
        raise AssertionError("expected invalid pyramid levels to be rejected")
