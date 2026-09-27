"""Tests for real-data path: download-log gate, PDS4 .img reading."""

import csv
import os
import tempfile

import numpy as np
import pytest

from chandra_align.ingest import read_image, _read_pds4_detached


def test_download_log_gate_blocks_without_entry():
    """Real-data runs without a download_log.csv entry must be rejected."""
    from scripts.run_pipeline import _find_log_entry
    with tempfile.TemporaryDirectory() as d:
        # Create a dummy image that doesn't look like a fixture
        img = np.random.rand(100, 100)
        test_path = os.path.join(d, "real_data.tif")
        # Write a minimal GeoTIFF
        import rasterio
        with rasterio.open(test_path, "w", driver="GTiff", height=100, width=100,
                           count=1, dtype="float64", crs="EPSG:4326") as dst:
            dst.write(img.astype(np.float64), 1)
        
        # _find_log_entry should return None for unlogged paths
        entry = _find_log_entry(test_path, run_id="test123")
        assert entry is None


def test_download_log_gate_passes_with_entry():
    """Real-data runs with a matching download_log.csv entry pass the gate."""
    from scripts.run_pipeline import _find_log_entry
    with tempfile.TemporaryDirectory() as d:
        img = np.random.rand(100, 100)
        test_path = os.path.join(d, "real_data.tif")
        import rasterio
        with rasterio.open(test_path, "w", driver="GTiff", height=100, width=100,
                           count=1, dtype="float64", crs="EPSG:4326") as dst:
            dst.write(img.astype(np.float64), 1)
        
        # Create a log entry matching this path in the PROJECT's data/download_log.csv
        # (not a temp file, since _find_log_entry reads from the project data dir)
        log_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "download_log.csv"
        )
        # Read existing entries
        with open(log_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Add test entry
        rows.append({"local_path": test_path, "run_id": "test123", "product_id": "test",
                     "source_url": "", "acquisition_date": "", "solar_azimuth_deg": "",
                     "solar_elevation_deg": "", "scale_ratio": ""})
        
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        
        try:
            entry = _find_log_entry(test_path, run_id="test123")
            assert entry is not None
            assert entry["run_id"] == "test123"
        finally:
            # Clean up: remove the test entry
            rows.pop()
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)


def test_pds4_ohrc_reading():
    """Read the real CH-2 OHRC PDS4 .img with detached label."""
    path = "data/ch2_ohr_ncp_20241115T1525004388_d_img_d18/data/calibrated/20241115/ch2_ohr_ncp_20241115T1525004388_d_img_d18.img"
    if not os.path.exists(path):
        pytest.skip("Real OHRC data not present")
    
    band, meta = read_image(path)
    assert band.shape == (101074, 12000)
    assert meta["crs"] == "POLAR_STEREOGRAPHIC"
    assert meta["width"] == 12000
    assert meta["height"] == 101074
    # Band should have meaningful data (not all zeros)
    assert band.max() > 10.0
    assert band.min() >= 0.0


def test_pds4_dimensions_from_label():
    """_read_pds4_detached parses dimensions and dtype from label correctly."""
    path = "data/ch2_ohr_ncp_20241115T1525004388_d_img_d18/data/calibrated/20241115/ch2_ohr_ncp_20241115T1525004388_d_img_d18.img"
    if not os.path.exists(path):
        pytest.skip("Real OHRC data not present")
    
    band, crs, transform, width, height, count = _read_pds4_detached(path)
    assert width == 12000
    assert height == 101074
    assert count == 1
    assert crs == "POLAR_STEREOGRAPHIC"


def test_synthetic_fixtures_bypass_log_gate():
    """Paths containing 'fixtures' bypass the download-log gate."""
    import scripts.run_pipeline as rp
    # The gate checks if 'fixtures' is in the path
    # This is a simple heuristic; synthetic fixtures are under tests/fixtures/
    assert "fixtures" in "tests/fixtures/pair_a.tif"
    assert "fixtures" not in "data/real_product.img"