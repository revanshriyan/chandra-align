"""CRS guard tests — projection-trap protection (M1.3)."""

import os

import numpy as np
import pytest

from chandra_align.ingest import CRSError, assert_pair_same_crs, assert_same_crs
from chandra_align.testing import make_pair_shift, write_tiff


def test_crs_guard_fires_on_mismatch(tmp_path):
    """Hard-assert identical CRS: mismatched CRS must raise CRSError BEFORE matching."""
    ref, mov, _ = make_pair_shift(seed=1)
    a = os.path.join(tmp_path, "a.tif")
    b = os.path.join(tmp_path, "b.tif")
    write_tiff(a, ref, crs="IAU_2015:30100")
    write_tiff(b, mov, crs="EPSG:4326")
    os.path.getsize(a) > 0
    with pytest.raises(CRSError):
        assert_pair_same_crs(a, b)


def test_crs_guard_fires_on_missing_crs(tmp_path):
    """No CRS (e.g. raw IIRS QUB) is a failure, not a silent pass."""
    ref, mov, _ = make_pair_shift(seed=2)
    a = os.path.join(tmp_path, "a.tif")
    b = os.path.join(tmp_path, "b.tif")
    write_tiff(a, ref, crs="IAU_2015:30100")
    write_tiff(b, mov, crs=None)
    with pytest.raises(CRSError):
        assert_pair_same_crs(a, b)


def test_crs_guard_passes_on_identical_crs(tmp_path):
    """Identical lunar CRS passes the guard."""
    ref, mov, _ = make_pair_shift(seed=3)
    a = os.path.join(tmp_path, "a.tif")
    b = os.path.join(tmp_path, "b.tif")
    write_tiff(a, ref, crs="EPSG:4326")
    write_tiff(b, mov, crs="EPSG:4326")
    *_, crs = assert_pair_same_crs(a, b)  # must not raise
    assert "4326" in crs


def test_assert_same_crs_direct():
    assert assert_same_crs("CRS_A", "CRS_A") == "CRS_A"
    with pytest.raises(CRSError):
        assert_same_crs("CRS_A", "CRS_B")
    with pytest.raises(CRSError):
        assert_same_crs("NONE", "CRS_A")
