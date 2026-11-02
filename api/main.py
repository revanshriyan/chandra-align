"""FastAPI demo service (M8.1).

POST /register — upload image pair + config → returns COG path, match points, metrics JSON, trust flag.
GET  /          — serves the built React viewer (web/dist) with SPA fallback.
Graceful degradation: a run ID that exists in the log but has no metrics/COG returns
explicit UNMEASURED states + "Not Trusted" instead of a 500.
Tile streaming: GET /tiles/{run_id}/{z}/{x}/{y}.png for COG tile access.
"""

import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from chandra_align.utils import load_config
from chandra_align.ingest import assert_pair_same_crs
from chandra_align import matcher as mm
from chandra_align.photometry import shadow_saturation_mask
from chandra_align.refine import verify_magsac, refine_subpixel_ncc, uniformity_score
from chandra_align.trust import evaluate, trust_flag, calibration_status
from chandra_align.metrics import inlier_stats, metrics_bundle, residual_map
from chandra_align.warp import (export_cog, fit_tile_affines, warp_piecewise_affine,
                                write_match_points_geojson, write_metrics_json)
from src.exporters.dossier import create_dossier, export_dossier
from src.schemas.models import (
    MetricsBundle,
    RegistrationDossier,
    TrustFlag,
    CalibrationStatus,
    PipelineStatus,
    UNMEASURED,
)

# Resolve the built viewer dist dir from env (relative-robust: falls back to web/dist
# relative to the repo root two levels up from this file)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_WEB_DIST = Path(os.environ.get("CHANDRA_ALIGN_WEB_DIST",
                                str(_REPO_ROOT / "web" / "dist")))


def _list_runs() -> list[dict]:
    """List all run directories with basic info.

    Also includes log entries from download_log.csv that have no run directory yet —
    these appear with explicit UNMEASURED states so the UI can show them honestly.
    """
    runs = {}
    if RUNS_DIR.exists():
        for run_dir in RUNS_DIR.iterdir():
            if not run_dir.is_dir():
                continue
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                try:
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                    runs[run_dir.name] = {
                        "run_id": run_dir.name,
                        "trust_flag": metrics.get("trust_flag", "UNKNOWN"),
                        "matcher": metrics.get("matcher", "UNKNOWN"),
                        "runtime_s": metrics.get("runtime_s"),
                        "has_metrics": True,
                        "has_cog": (run_dir / "registered.tif").exists(),
                    }
                except (json.JSONDecodeError, OSError):
                    pass

    # Merge log entries that have no run directory (un-run / failed pairs)
    log_path = _REPO_ROOT / "data" / "download_log.csv"
    if log_path.exists():
        import csv
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rid = row.get("run_id")
                    if rid and rid not in runs:
                        runs[rid] = {
                            "run_id": rid,
                            "trust_flag": "Not Trusted",
                            "matcher": "UNMEASURED",
                            "runtime_s": "UNMEASURED",
                            "has_metrics": False,
                            "has_cog": False,
                            "product_id": row.get("product_id"),
                            "status": "logged-not-run",
                        }
        except OSError:
            pass
    return sorted(runs.values(), key=lambda x: x.get("run_id", ""), reverse=True)


app = FastAPI(title="Chandra-Align Demo API", version="0.1.0")

# CORS: explicit origins only. A wildcard origin combined with
# credentials lets any website call this API from a visitor's browser.
# Override with CHANDRA_CORS_ORIGINS (comma-separated) when the viewer
# is served from another origin.
_CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CHANDRA_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:7860,http://127.0.0.1:7860",
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

RUNS_DIR = Path("outputs")


def _run_dir_or_none(run_id: str) -> Optional[Path]:
    """Resolve a run directory, or None if run_id escapes RUNS_DIR.

    run_id arrives from the URL, so it is caller-controlled: a value
    like ".." or "a/b" must never resolve outside the runs directory.
    Callers treat None exactly like a run that does not exist.
    """
    base = RUNS_DIR.resolve()
    candidate = (base / run_id).resolve()
    if candidate.parent != base:
        return None
    return candidate


@app.get("/api/runs")
async def list_runs() -> list[dict]:
    return _list_runs()


def _safe_upload_name(filename: Optional[str]) -> str:
    """Reduce an upload filename to a bare file name.

    UploadFile.filename is client-controlled and may carry path
    components ("../../x", Windows-style "..\\..\\x", browser
    fakepaths). Only the final component is ever used, so the write
    can never escape the destination directory. A filename with no
    usable component is rejected rather than silently renamed.
    """
    name = Path((filename or "").replace("\\", "/")).name.strip()
    if not name or name in {".", ".."}:
        raise HTTPException(400, "upload has no usable filename")
    return name


def _max_upload_bytes() -> int:
    """Per-file upload cap in bytes (env CHANDRA_MAX_UPLOAD_MB, default 1024)."""
    try:
        mb = int(os.environ.get("CHANDRA_MAX_UPLOAD_MB", "1024"))
    except ValueError:
        mb = 1024
    return max(0, mb) * 1024 * 1024


def _max_image_pixels() -> int:
    """Per-image pixel cap (env CHANDRA_MAX_IMAGE_PIXELS, default 100_000_000)."""
    try:
        return int(os.environ.get("CHANDRA_MAX_IMAGE_PIXELS", "100000000"))
    except ValueError:
        return 100_000_000


def _max_tiles() -> int:
    """Implied-tile-count cap per run (env CHANDRA_MAX_TILES, default 1024).

    Measured tile scaling is ~0.59 s/tile on a large canvas and a
    1024-tile run already costs ~10 minutes while producing metrics
    identical to a 1-tile run, so the default ceiling sits at the
    measured point of zero benefit rather than the previous 10000.
    """
    try:
        return int(os.environ.get("CHANDRA_MAX_TILES", "1024"))
    except ValueError:
        return 1024


def _max_concurrent_runs() -> int:
    """Concurrent /register executions (env CHANDRA_MAX_CONCURRENT_RUNS, default 2)."""
    try:
        return max(0, int(os.environ.get("CHANDRA_MAX_CONCURRENT_RUNS", "2")))
    except ValueError:
        return 2


_register_slots: Optional[threading.BoundedSemaphore] = None
_register_slots_limit: Optional[int] = None
_register_slots_lock = threading.Lock()


def _register_slot_semaphore() -> threading.BoundedSemaphore:
    """Semaphore bounding concurrent register executions.

    A single registration can hold gigabytes of RAM for minutes, so the
    threadpool offload alone is not enough: without a bound, a burst of
    heavy requests would saturate worker threads and host memory. The
    semaphore is recreated when the configured limit changes so the env
    knob takes effect without bookkeeping; holders of an older
    semaphore still release against it correctly.
    """
    global _register_slots, _register_slots_limit
    limit = _max_concurrent_runs()
    with _register_slots_lock:
        if _register_slots is None or _register_slots_limit != limit:
            _register_slots = threading.BoundedSemaphore(limit)
            _register_slots_limit = limit
        return _register_slots


def _write_upload(upload: UploadFile, tmp_dir: Path) -> str:
    """Stream an upload to disk in chunks under a per-file size cap.

    The previous implementation buffered the whole file in RAM and had
    no size limit, so a single upload could exhaust host memory. Bytes
    are counted as they stream; exceeding the cap aborts with 413 and
    the partial file is removed.
    """
    dest = tmp_dir / _safe_upload_name(upload.filename)
    cap = _max_upload_bytes()
    total = 0
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > cap:
                    raise HTTPException(
                        413,
                        f"upload exceeds the per-file size cap "
                        f"({cap // (1024 * 1024)} MB)",
                    )
                f.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    return str(dest)


def _validate_run_config(cfg: dict, shape) -> None:
    """Reject uploaded-config values that make a run unbounded (422).

    The config file is caller-supplied, so the numeric fields that size
    the compute must be range-checked before any stage runs: tile size
    and the implied tile count (the tile stage is linear in tile
    count), the matcher grid, the tile overlap, and the warp blend
    width. Values from the shipped modality profiles pass unchanged.
    """
    def _fail(msg):
        raise HTTPException(422, f"invalid config: {msg}")

    try:
        tile_h, tile_w = (int(v) for v in cfg["tile_size"])
    except (KeyError, TypeError, ValueError):
        _fail("tile_size must be a [height, width] pair")
    if tile_h < 128 or tile_w < 128:
        _fail("tile_size dimensions must be at least 128 px")

    try:
        overlap = float(cfg.get("tile_overlap", 0.0))
    except (TypeError, ValueError):
        _fail("tile_overlap must be a number")
    if not 0.0 <= overlap < 1.0:
        _fail("tile_overlap must be in [0, 1)")

    img_h, img_w = int(shape[0]), int(shape[1])

    def _tile_count(dim, tile):
        if dim <= tile:
            return 1
        step = max(1, int(tile * (1.0 - overlap)))
        return 1 + -(-(dim - tile) // step)

    n_tiles = _tile_count(img_h, tile_h) * _tile_count(img_w, tile_w)
    tile_limit = _max_tiles()
    if n_tiles > tile_limit:
        _fail(f"tile_size implies {n_tiles} tiles (limit {tile_limit})")

    try:
        grid_h, grid_w = (int(v) for v in cfg["matcher"]["grid"])
    except (KeyError, TypeError, ValueError):
        _fail("matcher.grid must be a [rows, cols] pair")
    if not (1 <= grid_h <= 64 and 1 <= grid_w <= 64):
        _fail("matcher.grid dimensions must be within 1..64")

    try:
        blend = int(cfg["warp"]["blend_width_px"])
    except (KeyError, TypeError, ValueError):
        _fail("warp.blend_width_px must be an integer")
    if not 0 <= blend <= 512:
        _fail("warp.blend_width_px must be within 0..512")


@app.post("/register")
async def register(
    request: Request,
    ref_file: UploadFile = File(..., description="Reference map image (GeoTIFF)"),
    mov_file: UploadFile = File(..., description="CH-2 product image (GeoTIFF)"),
    config_file: UploadFile = File(..., description="Modality profile YAML (ohrc/tmc/iirs)"),
) -> JSONResponse:
    """Register a CH-2 product to a reference map.

    Returns: {run_id, trust_flag, metrics, cog_note, paths...}
    """
    # Uploads trigger compute and disk writes. When CHANDRA_API_KEY is
    # set, require it as an X-API-Key header; when unset (the private
    # demo posture) the endpoint stays open. Full user authentication
    # remains a pre-public-deployment requirement.
    api_key = os.environ.get("CHANDRA_API_KEY")
    if api_key and request.headers.get("x-api-key") != api_key:
        raise HTTPException(401, "missing or invalid API key")
    run_id = uuid.uuid4().hex[:8]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save uploads in the handler (streaming, capped). Everything past
    # this point — ingest, matching, refinement, warp, export — is
    # blocking CPU/IO work, so it runs in the threadpool: executed
    # inline, a single heavy request froze the event loop (and every
    # other endpoint, health checks included) for its full duration.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ref_path = _write_upload(ref_file, tmp_path)
        mov_path = _write_upload(mov_file, tmp_path)
        cfg_path = _write_upload(config_file, tmp_path)

        return await run_in_threadpool(
            _run_registration, run_id, run_dir, ref_path, mov_path, cfg_path)


def _run_registration(run_id: str, run_dir: Path, ref_path: str,
                      mov_path: str, cfg_path: str) -> JSONResponse:
    """Blocking body of POST /register, executed via run_in_threadpool.

    Admission is bounded by a semaphore (CHANDRA_MAX_CONCURRENT_RUNS):
    when all slots are busy the run is rejected with 503 instead of
    queueing unbounded heavy work. HTTPExceptions raised here propagate
    through run_in_threadpool unchanged, so every guard keeps its
    status code (401 is checked earlier, in the handler).
    """
    slots = _register_slot_semaphore()
    if not slots.acquire(blocking=False):
        raise HTTPException(
            503,
            "registration capacity saturated: too many concurrent runs, "
            "retry later",
        )
    try:
        cfg = load_config(cfg_path)

        # Stage 1: ingest + CRS guard
        band_a, band_b, meta_a, meta_b, crs = assert_pair_same_crs(ref_path, mov_path)

        # Resource guards: reject oversize images and unbounded config
        # values before any matching compute starts. Matching and the
        # refinement stage scale super-linearly with pixel count, and
        # the config is caller-supplied.
        pixel_cap = _max_image_pixels()
        if band_a.size > pixel_cap or band_b.size > pixel_cap:
            raise HTTPException(
                413,
                f"image exceeds the per-image pixel cap ({pixel_cap} px)",
            )
        _validate_run_config(cfg, band_a.shape)

        # photometry mask (geometry-from-metadata; approximation flagged downstream)
        mask = shadow_saturation_mask(band_b, **{
            k: v for k, v in cfg["photometry"].items()
            if k in ("shadow_threshold_deg", "saturation_threshold")})

        # Stage 2: matcher cascade
        t0 = time.time()
        pts_a, pts_b, matcher_name, escalated = mm.run_cascade(
            band_a, band_b, cfg["matcher"])
        n_raw = pts_a.shape[0]

        # Stage 3: verify FIRST, then sub-pixel refine
        inl_a, inl_b, M_raw, inlier_ratio = verify_magsac(pts_a, pts_b, cfg["verification"])
        ref_a, ref_b, ref_stats = refine_subpixel_ncc(
            band_a, band_b, inl_a, inl_b,
            ncc_window=int(cfg["refinement"]["ncc_window"]))

        # Stage 4: uniformity + warp + outputs
        shape = band_a.shape
        uni = uniformity_score(ref_a, shape, grid=tuple(cfg["matcher"]["grid"]))
        M_fit, rm = evaluate(M_raw, ref_a, ref_b)  # circularity-guarded held-out RMSE

        tile_affines = fit_tile_affines(shape, ref_a, ref_b,
                                        tile_size=tuple(cfg["tile_size"]),
                                        overlap=float(cfg["tile_overlap"]))
        warped = warp_piecewise_affine(band_a, tile_affines,
                                       blend_width_px=int(cfg["warp"]["blend_width_px"]))
        warped_path = str(run_dir / "registered.tif")
        export_note = export_cog(warped, warped_path, crs=crs)
        mp_json = str(run_dir / "match_points.geojson")
        n_pts = write_match_points_geojson(mp_json, ref_a, ref_b, crs=crs, refined=True, M=M_fit)
        rm_metric = residual_map(ref_a, ref_b, M_fit, grid=tuple(cfg["matcher"]["grid"]),
                                shape=shape) if ref_a.shape[0] else {"residuals_px": []}

        runtime_s = time.time() - t0
        flag = trust_flag(rm.get("rmse_px", "UNMEASURED"), inlier_ratio, uni,
                          cfg["trust"])

        bundle_extra = {
            "matcher_tier_used": matcher_name,
            "escalated_to_tier2": escalated,
            "raw_matches": int(n_raw),
            "export": {"registered": warped_path, "cog_note": export_note,
                       "match_points_geojson": mp_json},
            "residual_map": rm_metric,
            "calibration_status": calibration_status(cfg["trust"]),
            "refinement": ref_stats,
        }
        bundle = metrics_bundle(rm, inlier_stats(n_raw, int(inl_a.shape[0]), inlier_ratio),
                                uni, runtime_s, flag, matcher_name,
                                approximation_flag=True, extra=bundle_extra)
        metrics_path = str(run_dir / "metrics.json")
        write_metrics_json(metrics_path, bundle)

        # Return the validated metrics bundle via model
        # Read back the saved metrics to ensure consistency
        with open(metrics_path, "r") as f:
            saved_metrics = json.load(f)

        # Create a validated MetricsBundle from the saved data
        try:
            validated_bundle = MetricsBundle(
                run_id=run_id,
                trust_flag=TrustFlag(flag),
                calibration_status=CalibrationStatus(bundle.get("calibration_status", "UNCALIBRATED")),
                status=PipelineStatus.COMPLETED,
                inliers={
                    "raw_matches": bundle.get("inliers", {}).get("raw_matches", n_raw),
                    "verified_inliers": bundle.get("inliers", {}).get("inliers", 0),
                    "inlier_ratio": inlier_ratio,
                    "uniformity_score": uni,
                    "quadtree_entropy": bundle.get("uniformity_score", 0.0),  # placeholder
                    "spatial_coverage_pct": 0.0,  # placeholder
                },
                rmse={
                    "held_out": rm.get("held_out", False),
                    "n_check_points": rm.get("n_check_points", 0),
                    "rmse_x_px": rm.get("rmse_x_px", UNMEASURED),
                    "rmse_y_px": rm.get("rmse_y_px", UNMEASURED),
                    "rmse_px": rm.get("rmse_px", UNMEASURED),
                    "rmse_m": rm.get("rmse_m", UNMEASURED),
                },
                condition_number_kappa=UNMEASURED,
                footprint_iou=UNMEASURED,
                quality_flags=[],
                matcher_tier_used=_parse_matcher_tier(matcher_name),
                escalated_to_tier2=escalated,
                transformation_mode=_parse_transformation_mode("piecewise_affine"),
                approximation_flag=True,
                runtime_s=runtime_s,
                registered_cog_path=warped_path,
                match_points_geojson_path=mp_json,
            )
            bundle_dict = validated_bundle.model_dump(mode="json")
        except Exception:
            # Fallback to raw bundle if validation fails
            bundle_dict = saved_metrics

        return JSONResponse({
            "run_id": run_id,
            "trust_flag": flag,
            "metrics": bundle_dict,
            "cog_note": export_note,
            "paths": {
                "registered": warped_path,
                "match_points": mp_json,
                "metrics": metrics_path,
            }
        })
    finally:
        slots.release()


def _parse_matcher_tier(matcher: Optional[str]):
    from src.schemas.models import MatcherTier
    if not matcher:
        return MatcherTier.TIER1_RIFT2
    matcher_lower = matcher.lower()
    if 'lightglue' in matcher_lower and 'aliked' in matcher_lower:
        return MatcherTier.TIER2_LIGHTGLUE_ALIKED
    elif 'lightglue' in matcher_lower and 'disk' in matcher_lower:
        return MatcherTier.TIER2_LIGHTGLUE_DISK
    elif 'sift_fallback' in matcher_lower:
        return MatcherTier.SIFT_FALLBACK
    elif 'sift' in matcher_lower:
        return MatcherTier.TIER2_SIFT
    elif 'rift2' in matcher_lower:
        return MatcherTier.TIER1_RIFT2
    return MatcherTier.TIER1_RIFT2


def _parse_transformation_mode(mode: Optional[str]):
    from src.schemas.models import TransformationMode
    if not mode:
        return TransformationMode.PIECEWISE_AFFINE
    mode_lower = mode.lower()
    if 'piecewise' in mode_lower:
        return TransformationMode.PIECEWISE_AFFINE
    elif 'global' in mode_lower or 'affine' in mode_lower:
        return TransformationMode.GLOBAL_AFFINE
    elif 'similarity' in mode_lower:
        return TransformationMode.SIMILARITY
    elif 'identity' in mode_lower:
        return TransformationMode.IDENTITY
    return TransformationMode.PIECEWISE_AFFINE


@app.get("/runs/{run_id}/registered.tif")
async def download_cog(run_id: str):
    run_dir = _run_dir_or_none(run_id)
    path = run_dir / "registered.tif" if run_dir is not None else None
    if path is None or not path.exists():
        # Graceful: logged but un-run → explicit UNMEASURED payload, not a 500
        return JSONResponse({
            "run_id": run_id,
            "status": "logged-not-run",
            "error": "registered.tif not generated for this run",
            "registered_cog": "UNMEASURED",
            "trust_flag": "Not Trusted",
        }, status_code=200)
    return FileResponse(path, media_type="image/tiff", filename=f"{run_id}_registered.tif")


@app.get("/runs/{run_id}/match_points.geojson")
async def download_geojson(run_id: str):
    run_dir = _run_dir_or_none(run_id)
    path = run_dir / "match_points.geojson" if run_dir is not None else None
    if path is None or not path.exists():
        return JSONResponse({
            "run_id": run_id,
            "status": "logged-not-run",
            "error": "match_points.geojson not generated for this run",
            "match_points": "UNMEASURED",
            "trust_flag": "Not Trusted",
        }, status_code=200)
    return FileResponse(path, media_type="application/geo+json", filename=f"{run_id}_match_points.geojson")


@app.get("/runs/{run_id}/metrics.json", response_model=MetricsBundle)
async def download_metrics(run_id: str) -> MetricsBundle:
    run_dir = _run_dir_or_none(run_id)
    path = run_dir / "metrics.json" if run_dir is not None else None
    if path is None or not path.exists():
        # Return a minimal valid model with UNMEASURED fields
        return MetricsBundle(
            run_id=run_id,
            trust_flag=TrustFlag.NOT_TRUSTED,
            calibration_status=CalibrationStatus.UNCALIBRATED,
            status=PipelineStatus.LOGGED_NOT_RUN,
            inliers={
                "raw_matches": 0,
                "verified_inliers": 0,
                "inlier_ratio": 0.0,
                "uniformity_score": 0.0,
                "quadtree_entropy": None,
                "spatial_coverage_pct": 0.0,
            },
            rmse={
                "held_out": False,
                "n_check_points": 0,
                "rmse_x_px": UNMEASURED,
                "rmse_y_px": UNMEASURED,
                "rmse_px": UNMEASURED,
                "rmse_m": UNMEASURED,
            },
            condition_number_kappa=UNMEASURED,
            footprint_iou=UNMEASURED,
            quality_flags=[],
            matcher_tier_used=_parse_matcher_tier(None),
            escalated_to_tier2=False,
            transformation_mode=_parse_transformation_mode(None),
            approximation_flag=True,
            runtime_s=0.0,
        )
    with open(path, "r") as f:
        data = json.load(f)
    # Ensure run_id is present for validation
    data.setdefault("run_id", run_id)
    # Parse and validate through model
    return MetricsBundle.model_validate(data)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/v1/export-dossier/{run_id}", response_model=RegistrationDossier)
async def export_dossier_endpoint(run_id: str) -> RegistrationDossier:
    """Export registration dossier for a run."""
    run_dir = _run_dir_or_none(run_id)
    if run_dir is None or not run_dir.exists():
        raise HTTPException(404, f"Run {run_id} not found")

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(404, f"Metrics for run {run_id} not found")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Try to load dossier if already generated, otherwise create it
    dossier_path = run_dir / "dossier.json"
    if dossier_path.exists():
        with open(dossier_path, "r") as f:
            dossier_data = json.load(f)
        return RegistrationDossier.model_validate(dossier_data)
    else:
        # Return a minimal valid dossier indicating it needs regeneration
        raise HTTPException(
            404,
            "Full dossier not generated for this run. Re-run pipeline with dossier export enabled."
        )


# --- Tile Streaming Endpoint ---
@app.get("/tiles/{run_id}/{z}/{x}/{y}.png")
async def stream_cog_tile(run_id: str, z: int, x: int, y: int):
    """
    Stream COG tile from registered.tif for interactive Web-GIS viewing.
    
    Converts Web Mercator (z, x, y) to the COG's CRS and reads the appropriate window.
    Returns a 256x256 PNG tile.
    """
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.transform import from_bounds as transform_from_bounds
    import numpy as np
    from io import BytesIO
    
    run_dir = _run_dir_or_none(run_id)
    cog_path = run_dir / "registered.tif" if run_dir is not None else None
    if cog_path is None or not cog_path.exists():
        raise HTTPException(404, f"COG not found for run {run_id}")
    
    try:
        with rasterio.open(cog_path) as src:
            # Get CRS and bounds
            crs = src.crs
            bounds = src.bounds
            
            # Web Mercator projection bounds
            # Earth radius in Web Mercator (EPSG:3857)
            R = 6378137.0
            world_size = 2 * np.pi * R
            tile_size = world_size / (2 ** z)
            
            # Tile bounds in Web Mercator
            minx = x * tile_size - world_size / 2
            maxx = (x + 1) * tile_size - world_size / 2
            miny = world_size / 2 - (y + 1) * tile_size
            maxy = world_size / 2 - y * tile_size
            
            # If COG is in EPSG:4326 (geographic), we need to convert
            # For simplicity, assume COG is in EPSG:4326 and read using geographic bounds
            # A full implementation would use pyproj to convert Web Mercator to COG CRS
            if crs and crs.to_epsg() == 4326:
                # Convert Web Mercator to lat/lon
                def web_mercator_to_lonlat(x_web, y_web):
                    lon = x_web / R * 180.0 / np.pi
                    lat = np.arctan(np.sinh(y_web / R)) * 180.0 / np.pi
                    return lon, lat
                
                lon_min, lat_max = web_mercator_to_lonlat(minx, maxy)
                lon_max, lat_min = web_mercator_to_lonlat(maxx, miny)
                
                # Read window from COG
                window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
            else:
                # For other CRS, use bounds directly (simplified)
                window = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, src.transform)
            
            # Read the tile data
            tile_data = src.read(1, window=window, out_shape=(256, 256))
            
            # Normalize to 0-255 for PNG
            finite = np.isfinite(tile_data)
            if not np.any(finite):
                # Empty tile
                img_data = np.zeros((256, 256), dtype=np.uint8)
            else:
                lo, hi = np.nanpercentile(tile_data[finite], [2, 98])
                if hi <= lo:
                    img_data = np.zeros((256, 256), dtype=np.uint8)
                else:
                    normalized = np.clip((tile_data - lo) / (hi - lo) * 255, 0, 255)
                    img_data = normalized.astype(np.uint8)
            
            # Convert to PNG
            from PIL import Image
            img = Image.fromarray(img_data, mode='L')
            buf = BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            
            return Response(content=buf.read(), media_type="image/png")
            
    except Exception as e:
        raise HTTPException(500, f"Tile streaming error: {str(e)}")


# --- Static React viewer serving (SPA fallback) ---
# Mounted last so all /api and /runs routes take precedence.
if _WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_WEB_DIST), html=True), name="viewer")
else:
    @app.get("/", include_in_schema=False)
    async def index_fallback():
        return HTMLResponse(
            "<html><body><h1>Chandra-Align API</h1>"
            "<p>Viewer not built. Run <code>cd web && npm install && npm run build</code> "
            "or <code>docker compose up --build</code>.</p>"
            '<p><a href="/docs">API docs</a> | <a href="/health">Health</a></p>'
            "</body></html>", status_code=200)