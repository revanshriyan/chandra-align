"""End-to-end test for POST /register (api/main.py).

Until now the endpoint was only exercised by hand: it could not
complete a single request (the trust evaluate call had the wrong
signature, and a documented CPU install could not load the default
GPU matcher tier). This test drives the real endpoint with a small
deterministic pair, with the GPU stack simulated as unavailable so
the SIFT fallback path is what actually runs — the posture of a
documented CPU install.

Guard tests cover the API resource limits: per-file upload cap,
per-image pixel cap, and uploaded-config range validation.

Hardening tests cover the execution model: the blocking stage work
runs in the threadpool (not the event loop), concurrent executions
are semaphore-bounded with a 503 when saturated, and the implied
tile-count ceiling defaults to 1024 with an env override.
"""

import json
import os

import pytest
import yaml
from fastapi.testclient import TestClient

from api.main import app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_OHRC = os.path.join(ROOT, "config", "ohrc.yaml")


@pytest.fixture()
def gpu_stack_unavailable(monkeypatch):
    """Simulate a deployment without torch/lightglue installed.

    LightGlueMatcher._load is where the missing GPU stack surfaces as
    a RuntimeError; raising it here exercises the exact fallback the
    cascade must take on a CPU-only install.
    """
    from chandra_align import matcher as mm

    def _raise(self):
        raise RuntimeError("GPU stack not installed (simulated)")

    monkeypatch.setattr(mm.LightGlueMatcher, "_load", _raise)


def _post_register(client, ref_path, mov_path, cfg_path):
    with open(ref_path, "rb") as rf, open(mov_path, "rb") as mf, \
            open(cfg_path, "rb") as cf:
        return client.post("/register", files={
            "ref_file": ("ref.tif", rf, "image/tiff"),
            "mov_file": ("mov.tif", mf, "image/tiff"),
            "config_file": ("ohrc.yaml", cf, "application/yaml"),
        })


def _modified_config(tmp_path, **sections):
    with open(CONFIG_OHRC) as f:
        cfg = yaml.safe_load(f)
    for key, value in sections.items():
        if isinstance(value, dict):
            cfg.setdefault(key, {}).update(value)
        else:
            cfg[key] = value
    out = os.path.join(str(tmp_path), "hostile.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f)
    return out


def test_register_completes_via_sift_fallback(tmp_path, monkeypatch,
                                              shifted_pair,
                                              gpu_stack_unavailable):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)
    resp = _post_register(client, shifted_pair["ref"], shifted_pair["mov"],
                          CONFIG_OHRC)
    assert resp.status_code == 200, resp.text[:500]
    body = resp.json()
    assert body["run_id"]
    assert body["trust_flag"] in {"Trusted", "Not Trusted"}
    assert isinstance(body["metrics"], dict)
    assert body["metrics"].get("rmse") is not None

    # The saved bundle must honestly report the matcher that ran.
    metrics_path = tmp_path / "outputs" / body["run_id"] / "metrics.json"
    saved = json.loads(metrics_path.read_text())
    assert saved["matcher"] == "sift_fallback"
    assert saved["matcher_tier_used"] == "sift_fallback"


def test_register_response_reports_sift_fallback_tier(tmp_path, monkeypatch,
                                                     shifted_pair,
                                                     gpu_stack_unavailable):
    """The consumer-facing response must name the fallback, not plain SIFT.

    Regression test for a response/saved-bundle divergence: the saved
    metrics.json recorded "sift_fallback" while the response JSON
    serialized metrics.matcher_tier_used as "sift", because the
    response schema had no value for the fallback name.
    """
    monkeypatch.chdir(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)
    resp = _post_register(client, shifted_pair["ref"], shifted_pair["mov"],
                          CONFIG_OHRC)
    assert resp.status_code == 200, resp.text[:500]
    body = resp.json()
    assert body["metrics"]["matcher_tier_used"] == "sift_fallback"


def test_register_rejects_oversized_upload(tmp_path, monkeypatch,
                                           shifted_pair):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHANDRA_MAX_UPLOAD_MB", "0")
    client = TestClient(app, raise_server_exceptions=False)
    resp = _post_register(client, shifted_pair["ref"], shifted_pair["mov"],
                          CONFIG_OHRC)
    assert resp.status_code == 413


def test_register_rejects_oversize_image(tmp_path, monkeypatch,
                                         shifted_pair):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHANDRA_MAX_IMAGE_PIXELS", "10000")
    client = TestClient(app, raise_server_exceptions=False)
    resp = _post_register(client, shifted_pair["ref"], shifted_pair["mov"],
                          CONFIG_OHRC)
    assert resp.status_code == 413


@pytest.mark.parametrize("sections", [
    {"tile_size": [4, 4]},                          # tiles below the 128 px floor
    {"tile_size": [128, 128], "tile_overlap": 0.99},  # implied tile count explodes
    {"matcher": {"grid": [1000, 4]}},               # grid above the 1..64 range
    {"warp": {"blend_width_px": 9999}},             # blend above the 0..512 range
])
def test_register_rejects_hostile_config(tmp_path, monkeypatch,
                                         shifted_pair, sections):
    monkeypatch.chdir(tmp_path)
    cfg_path = _modified_config(tmp_path, **sections)
    client = TestClient(app, raise_server_exceptions=False)
    resp = _post_register(client, shifted_pair["ref"], shifted_pair["mov"],
                          cfg_path)
    assert resp.status_code == 422, resp.text[:300]


def test_register_offloads_stage_work_to_threadpool(tmp_path, monkeypatch,
                                                    shifted_pair,
                                                    gpu_stack_unavailable):
    """The blocking pipeline must not run in the async handler.

    Regression test for the event-loop freeze: the handler used to run
    ingest/matching/warp inline, so one heavy request stalled every
    other endpoint for its full duration. The stage work now goes
    through run_in_threadpool; the result must still come back whole.
    """
    import api.main as api_main

    calls = []
    real_run_in_threadpool = api_main.run_in_threadpool

    async def spy(func, *args, **kwargs):
        calls.append(func)
        return await real_run_in_threadpool(func, *args, **kwargs)

    monkeypatch.setattr(api_main, "run_in_threadpool", spy)
    monkeypatch.chdir(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)
    resp = _post_register(client, shifted_pair["ref"], shifted_pair["mov"],
                          CONFIG_OHRC)
    assert resp.status_code == 200, resp.text[:500]
    assert calls == [api_main._run_registration]


def test_register_rejects_when_capacity_saturated(tmp_path, monkeypatch,
                                                  shifted_pair,
                                                  gpu_stack_unavailable):
    """With zero execution slots, /register answers 503, not a hang."""
    monkeypatch.setenv("CHANDRA_MAX_CONCURRENT_RUNS", "0")
    monkeypatch.chdir(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)
    resp = _post_register(client, shifted_pair["ref"], shifted_pair["mov"],
                          CONFIG_OHRC)
    assert resp.status_code == 503, resp.text[:300]
    assert "saturated" in resp.json()["detail"]


def test_register_slot_released_after_failed_run(tmp_path, monkeypatch,
                                                 shifted_pair,
                                                 gpu_stack_unavailable):
    """The semaphore must not leak when the worker raises (422 path).

    With a single slot: a hostile config is rejected inside the
    worker, then a valid request must still be admitted — proving the
    slot came back via the finally release.
    """
    monkeypatch.setenv("CHANDRA_MAX_CONCURRENT_RUNS", "1")
    monkeypatch.chdir(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)
    bad_cfg = _modified_config(tmp_path, tile_size=[4, 4])
    resp = _post_register(client, shifted_pair["ref"], shifted_pair["mov"],
                          bad_cfg)
    assert resp.status_code == 422, resp.text[:300]
    resp = _post_register(client, shifted_pair["ref"], shifted_pair["mov"],
                          CONFIG_OHRC)
    assert resp.status_code == 200, resp.text[:300]


def test_register_tile_cap_names_effective_limit(tmp_path, monkeypatch,
                                                 shifted_pair):
    """A config implying >1024 tiles is a 422 naming the 1024 limit."""
    monkeypatch.chdir(tmp_path)
    cfg_path = _modified_config(tmp_path, tile_size=[128, 128],
                                tile_overlap=0.99)
    client = TestClient(app, raise_server_exceptions=False)
    resp = _post_register(client, shifted_pair["ref"], shifted_pair["mov"],
                          cfg_path)
    assert resp.status_code == 422, resp.text[:300]
    assert "1024" in resp.text


def test_validate_run_config_tile_limit_and_override(monkeypatch):
    """Unit-level: the tile ceiling, its env override, and shipped profiles."""
    import api.main as api_main
    from fastapi import HTTPException

    with open(CONFIG_OHRC) as f:
        cfg = yaml.safe_load(f)

    # The shipped OHRC profile on the largest documented strip
    # (~12000 x 90000 px, tile 4096, overlap 0.20) implies 112 tiles:
    # comfortably under the new ceiling.
    api_main._validate_run_config(cfg, (90000, 12000))

    cfg["tile_size"] = [128, 128]
    cfg["tile_overlap"] = 0.5
    with pytest.raises(HTTPException) as excinfo:
        api_main._validate_run_config(cfg, (4096, 4096))
    assert excinfo.value.status_code == 422
    assert "(limit 1024)" in str(excinfo.value.detail)

    # The override restores headroom and is the limit the 422 names.
    monkeypatch.setenv("CHANDRA_MAX_TILES", "10000")
    api_main._validate_run_config(cfg, (4096, 4096))
    cfg["tile_overlap"] = 0.99
    with pytest.raises(HTTPException) as excinfo:
        api_main._validate_run_config(cfg, (4096, 4096))
    assert "(limit 10000)" in str(excinfo.value.detail)


def test_dossier_preserves_sift_fallback_tier():
    """B3: offline dossiers must record sift_fallback, not plain sift.

    src/exporters/dossier.py has its own _parse_matcher_tier; the generic
    'sift' branch used to swallow 'sift_fallback', dropping the honesty
    caveat from exported dossiers even though the API response kept it.
    """
    from src.exporters.dossier import _parse_matcher_tier
    from src.schemas.models import MatcherTier

    assert _parse_matcher_tier("sift_fallback") == MatcherTier.SIFT_FALLBACK
    # The generic branch still applies to plain SIFT.
    assert _parse_matcher_tier("sift") == MatcherTier.TIER2_SIFT
