import numpy as np
import pytest

from chandra_align.matching.tiled import iter_tile_windows, mosaic_tile_arrays


def test_tile_windows_cover_image_and_mosaic_reconstructs_exact_pixels():
    image = np.arange(37 * 53, dtype=np.uint16).reshape(37, 53)
    tiles = list(iter_tile_windows(53, 37, tile_size=24, overlap=6))
    mosaic = mosaic_tile_arrays(
        53, 37,
        [(tile, image[tile.y:tile.y + tile.height, tile.x:tile.x + tile.width])
         for tile in tiles],
    )
    assert np.array_equal(mosaic, image)
    assert len(tiles) > 1


def test_tile_math_handles_small_and_exact_size_inputs():
    small = list(iter_tile_windows(11, 9, tile_size=32, overlap=4))
    exact = list(iter_tile_windows(32, 32, tile_size=32, overlap=4))
    assert len(small) == 1 and small[0].core == (0, 0, 11, 9)
    assert len(exact) == 1 and (exact[0].width, exact[0].height) == (32, 32)


def test_tile_math_rejects_invalid_overlap():
    with pytest.raises(ValueError, match="overlap"):
        list(iter_tile_windows(100, 100, tile_size=16, overlap=16))
