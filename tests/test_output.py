"""Stage 4 output test — warp + COG + GeoJSON sidecar."""

import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from chandra_align.warp import (export_cog, fit_tile_affines, warp_piecewise_affine,
                               write_match_points_geojson, write_metrics_json)
from chandra_align import matcher as mm
from chandra_align.refine import verify_magsac


def test_warp_piecewise_affine_runs(tmp_path, shifted_pair):
    """Piecewise-affine warp must run, produce an output of the right shape, and
    require at least one valid tile affine."""
    rng = np.random.default_rng(0)
    n = 80
    pid = np.random.default_rng(1).uniform(10, 502, (n, 2))
    # with a small global shift so all tiles contain inliers
    pb = pid + 4.0
    aff = fit_tile_affines((512, 512), pid, pb,
                           tile_size=(256, 256), overlap=0.2,
                           cfg_verification={"ransac_reproj_threshold": 3.0,
                                             "max_iters": 2000, "confidence": 0.99})
    assert len(aff) >= 1, "no fittable affine tile"
    out = warp_piecewise_affine(np.asarray(shifted_pair["ref_img"], np.float32),
                                aff, blend_width_px=16)
    assert out.shape == (512, 512)
    assert np.isfinite(out).all()


def test_export_cog_embeds_crs(tmp_path, shifted_pair):
    """COG/GeoTIFF export must carry the CRS we passed in."""
    import rasterio

    arr = np.asarray(shifted_pair["ref_img"], np.float64)
    out = os.path.join(tmp_path, "warped.tif")
    note = export_cog(arr, out, crs="EPSG:4326")
    assert "geotiff" in note.lower()
    with rasterio.open(out) as src:
        assert src.crs is not None
        assert "4326" in str(src.crs.to_string())
        assert src.height == 512 and src.width == 512


def test_geojson_sidecar(tmp_path, shifted_pair):
    """Match-point GeoJSON must be written and valid."""
    from chandra_align.matcher import SIFTMatcher
    m = SIFTMatcher(lowes_ratio=0.75, n_features=4000)
    pa, pb = m.match(shifted_pair["ref_img"], shifted_pair["mov_img"])
    assert pa.shape[0] >= 1
    out = os.path.join(tmp_path, "mp.geojson")
    n = write_match_points_geojson(out, pa, pb, crs="EPSG:4326")
    assert n == pa.shape[0]
    data = json.loads(open(out).read())
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == n
    assert data["crs"]["properties"]["name"] == "EPSG:4326"


def test_live_run_detail_routes_are_available():
    """The live run-detail endpoints used by the viewer must resolve correctly."""
    client = TestClient(app)

    metrics = client.get('/runs/test_run/metrics.json')
    geojson = client.get('/runs/test_run/match_points.geojson')

    assert metrics.status_code == 200
    assert geojson.status_code == 200
    payload = metrics.json()
    assert payload.get('trust_flag') in {"Not Trusted", "UNKNOWN"}
