"""FastAPI demo service (M8.1).

POST /register — upload image pair + config → returns COG path, match points, metrics JSON, trust flag.
GET  /          — serves the built React viewer (web/dist) with SPA fallback.
Graceful degradation: a run ID that exists in the log but has no metrics/COG returns
explicit UNMEASURED states + "Not Trusted" instead of a 500.
"""

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNS_DIR = Path("outputs")


@app.get("/api/runs")
async def list_runs() -> list[dict]:
    return _list_runs()


def _write_upload(upload: UploadFile, tmp_dir: Path) -> str:
    dest = tmp_dir / upload.filename
    with open(dest, "wb") as f:
        f.write(upload.file.read())
    return str(dest)


@app.post("/register")
async def register(
    ref_file: UploadFile = File(..., description="Reference map image (GeoTIFF)"),
    mov_file: UploadFile = File(..., description="CH-2 product image (GeoTIFF)"),
    config_file: UploadFile = File(..., description="Modality profile YAML (ohrc/tmc/iirs)"),
) -> JSONResponse:
    """Register a CH-2 product to a reference map.

    Returns: {run_id, trust_flag, metrics, cog_note, paths...}
    """
    run_id = uuid.uuid4().hex[:8]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # save uploads
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ref_path = _write_upload(ref_file, tmp_path)
        mov_path = _write_upload(mov_file, tmp_path)
        cfg_path = _write_upload(config_file, tmp_path)

        cfg = load_config(cfg_path)

        # Stage 1: ingest + CRS guard
        band_a, band_b, meta_a, meta_b, crs = assert_pair_same_crs(ref_path, mov_path)

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
        M_fit, rm = evaluate(ref_a, ref_b)  # circularity-guarded held-out RMSE

        tile_affines = fit_tile_affines(shape, ref_a, ref_b,
                                        tile_size=tuple(cfg["tile_size"]),
                                        overlap=float(cfg["tile_overlap"]))
        warped = warp_piecewise_affine(band_a, tile_affines,
                                       blend_width_px=int(cfg["warp"]["blend_width_px"]))
        warped_path = str(run_dir / "registered.tif")
        export_note = export_cog(warped, warped_path, crs=crs)
        mp_json = str(run_dir / "match_points.geojson")
        n_pts = write_match_points_geojson(mp_json, ref_a, ref_b, crs=crs, refined=True)
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


def _parse_matcher_tier(matcher: Optional[str]):
    from src.schemas.models import MatcherTier
    if not matcher:
        return MatcherTier.TIER1_RIFT2
    matcher_lower = matcher.lower()
    if 'lightglue' in matcher_lower and 'aliked' in matcher_lower:
        return MatcherTier.TIER2_LIGHTGLUE_ALIKED
    elif 'lightglue' in matcher_lower and 'disk' in matcher_lower:
        return MatcherTier.TIER2_LIGHTGLUE_DISK
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
    path = RUNS_DIR / run_id / "registered.tif"
    if not path.exists():
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
    path = RUNS_DIR / run_id / "match_points.geojson"
    if not path.exists():
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
    path = RUNS_DIR / run_id / "metrics.json"
    if not path.exists():
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
    # Parse and validate through model
    return MetricsBundle.model_validate(data)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/v1/export-dossier/{run_id}", response_model=RegistrationDossier)
async def export_dossier_endpoint(run_id: str) -> RegistrationDossier:
    """Export registration dossier for a run."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
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