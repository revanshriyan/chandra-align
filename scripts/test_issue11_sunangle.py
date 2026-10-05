import numpy as np

from scripts.run_issue11_sunangle import (
    UPSCALE,
    hapke_style_render,
    identity_rmse,
    to_common_uint8,
)


def test_photometric_renders_are_finite_and_change_with_sun_direction():
    yy, xx = np.mgrid[:32, :32]
    height = 12.0 * np.sin(xx / 3.0) + 8.0 * np.cos(yy / 4.0)
    low_sun = hapke_style_render(height, -45.0, 15.0)
    high_sun = hapke_style_render(height, 45.0, 60.0)
    assert np.isfinite(low_sun).all()
    assert np.isfinite(high_sun).all()
    assert not np.allclose(low_sun, high_sun)


def test_shared_grid_identity_is_exact_ground_truth():
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert identity_rmse(identity, 512) == 0.0


def test_common_stretch_preserves_equal_dimensions_for_all_variants():
    a = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    images = to_common_uint8({"a": a, "b": np.flipud(a).copy()})
    assert set(images) == {"a", "b"}
    assert images["a"].shape == (32 * UPSCALE, 32 * UPSCALE)
    assert images["b"].shape == images["a"].shape
    assert images["a"].dtype == np.uint8
