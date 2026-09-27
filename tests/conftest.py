"""tests/conftest.py — shared fixture paths and helpers."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_OHRC = os.path.join(ROOT, "config", "ohrc.yaml")

from chandra_align.utils import load_config


@pytest.fixture()
def cfg():
    return load_config(CONFIG_OHRC)


def _pair_dir(tmp_path, name):
    d = os.path.join(str(tmp_path), name)
    os.makedirs(d, exist_ok=True)
    return d


@pytest.fixture()
def shifted_pair(tmp_path):
    """Known sub-pixel shift pair + ground truth, written as GeoTIFFs with a lunar CRS."""
    from chandra_align.testing import make_pair_shift, write_tiff
    d = _pair_dir(tmp_path, "shift")
    ref, mov, M = make_pair_shift(dx=7.3, dy=-3.9, angle_deg=0.4, seed=7)
    a = os.path.join(d, "ref.tif")
    b = os.path.join(d, "mov.tif")
    write_tiff(a, ref, crs="IAU_2015:30100")
    write_tiff(b, mov, crs="IAU_2015:30100")
    return {"ref": a, "mov": b, "M": M, "ref_img": ref, "mov_img": mov}


@pytest.fixture()
def degenerate_pair(tmp_path):
    from chandra_align.testing import make_pair_degenerate, write_tiff
    d = _pair_dir(tmp_path, "degenerate")
    a_img, b_img, _ = make_pair_degenerate(seed=7)
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    write_tiff(a, a_img, crs="IAU_2015:30100")
    write_tiff(b, b_img, crs="IAU_2015:30100")
    return {"ref": a, "mov": b, "ref_img": a_img, "mov_img": b_img}
