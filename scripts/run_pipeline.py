"""scripts/run_pipeline.py — end-to-end Stage-1 prototype driver on an image pair.

Usage:
  .venv/Scripts/python scripts/run_pipeline.py \
      config/ohrc.yaml fixtures/pair_ref.tif fixtures/pair_mov.tif outputs/run1 metrics.json

For real-data runs (CH-2 product + reference map), the product must have a matching
entry in data/download_log.csv with the same run_id. Synthetic fixtures bypass the log.
"""

import json
import os
import sys
import time
import uuid
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find_log_entry(product_path: str, run_id: str | None = None) -> dict | None:
    """Find a download_log.csv entry matching the product path and optionally run_id."""
    log_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "download_log.csv"
    )
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            logged_path = row.get("local_path")
            if not logged_path:
                continue
            try:
                matches = os.path.samefile(logged_path, product_path)
            except OSError:
                matches = os.path.normcase(os.path.abspath(logged_path)) == os.path.normcase(os.path.abspath(product_path))
            if matches:
                if run_id is None or row.get("run_id") == run_id:
                    return row
    return None


def main(config_path, ref_path, mov_path, out_dir, metrics_path):
    import numpy as np

    # --- Download-log gate for real data ---
    # Detect if inputs look like real data (not synthetic fixtures under fixtures/)
    is_synthetic = "fixtures" in mov_path or "fixtures" in ref_path
    if not is_synthetic:
        # Use existing run_id from log if available (pass None to match by path only)
        log_entry = _find_log_entry(mov_path, None)
        if log_entry is None:
            print(json.dumps({
                "error": "real-data run requires a matching entry in data/download_log.csv",
                "product_path": mov_path,
            }, indent=1))
            sys.exit(3)
        run_id = log_entry.get("run_id", uuid.uuid4().hex[:8])
    else:
        run_id = uuid.uuid4().hex[:8]

    from chandra_align.utils import load_config
    from chandra_align.ingest import assert_pair_same_crs
    from chandra_align import matcher as mm
    from chandra_align.photometry import shadow_saturation_mask, normalize_photometry
    from chandra_align.refine import verify_magsac, refine_subpixel_ncc, uniformity_score
    from chandra_align.trust import evaluate, trust_flag, calibration_status
    from chandra_align.metrics import inlier_stats, metrics_bundle, residual_map
    from chandra_align.warp import (export_cog, fit_tile_affines, warp_piecewise_affine,
                                    write_match_points_geojson, write_metrics_json)
    from src.exporters.dossier import create_dossier, export_dossier_json, export_dossier

    os.makedirs(out_dir, exist_ok=True)
    cfg = load_config(config_path)
    t0 = time.time()

    # Stage 1: ingest + CRS guard
    band_a, band_b, meta_a, meta_b, crs = assert_pair_same_crs(ref_path, mov_path)

    # Photometric normalization with CLAHE + Lommel-Seeliger
    phot_cfg = cfg["photometry"]
    if phot_cfg.get("enable_clahe", False):
        from chandra_align.photometry import normalize_photometry

        # Get solar angles from metadata
        inc_key = phot_cfg.get("incidence_angle_key", "SOLAR_INCIDENCE")
        eme_key = phot_cfg.get("emission_angle_key", "EMISSION_ANGLE")
        incidence = float(meta_b.get(inc_key, phot_cfg.get("ref_incidence_deg", 60.0)))
        emission = float(meta_b.get(eme_key, phot_cfg.get("ref_emission_deg", 0.0)))

        band_b = normalize_photometry(
            band_b,
            incidence_angle_deg=incidence,
            emission_angle_deg=emission,
            model=phot_cfg.get("model", "lommel_seeliger"),
            enable_clahe=phot_cfg.get("enable_clahe", True),
            clahe_clip_limit=phot_cfg.get("clahe_clip_limit", 2.0),
            clahe_tile_grid=tuple(phot_cfg.get("clahe_tile_grid", [8, 8])),
            ref_incidence_deg=phot_cfg.get("ref_incidence_deg", 60.0),
            ref_emission_deg=phot_cfg.get("ref_emission_deg", 0.0)
        )

    # photometry mask (geometry-from-metadata; approximation flagged downstream)
    mask_a = shadow_saturation_mask(band_b, **{
        k: v for k, v in cfg["photometry"].items()
        if k in ("shadow_threshold_deg", "saturation_threshold")})
    _ = mask_a  # kept for downstream use; lighting model handled elsewhere

    # Stage 2: matcher cascade
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
    warped_path = os.path.join(out_dir, "registered.tif")
    export_note = export_cog(warped, warped_path, crs=crs)
    mp_json = os.path.join(out_dir, "match_points.geojson")
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
    write_metrics_json(metrics_path, bundle)

    # Create and export dossier
    dossier = create_dossier(
        run_id=run_id,
        product_info={
            'product_id': 'moving_image',
            'path': mov_path,
            'crs': meta_b.get('crs', 'UNKNOWN'),
            'gsd_m': cfg.get('gsd_m', 0.25),
            'size': {'width': meta_b.get('width'), 'height': meta_b.get('height')},
            'solar_azimuth_deg': meta_b.get('SOLAR_AZIMUTH', 'UNMEASURED'),
            'sun_elevation_deg': meta_b.get('SUN_ELEVATION', 'UNMEASURED'),
            'incidence_angle_deg': meta_b.get('SOLAR_INCIDENCE', 'UNMEASURED'),
        },
        reference_info={
            'product_id': 'reference_image',
            'path': ref_path,
            'crs': meta_a.get('crs', 'UNKNOWN'),
            'gsd_m': cfg.get('gsd_m', 0.25),
            'size': {'width': meta_a.get('width'), 'height': meta_a.get('height')},
            'solar_azimuth_deg': meta_a.get('SOLAR_AZIMUTH', 'UNMEASURED'),
            'sun_elevation_deg': meta_a.get('SUN_ELEVATION', 'UNMEASURED'),
            'incidence_angle_deg': meta_a.get('SOLAR_INCIDENCE', 'UNMEASURED'),
        },
        match_points=[
            {
                'x_ref': float(pa[0]), 'y_ref': float(pa[1]),
                'x_mov': float(pb[0]), 'y_mov': float(pb[1]),
                'refined': True,
                'residual_px': 0.0,  # placeholder, actual residuals in GeoJSON
            }
            for pa, pb in zip(ref_a.tolist(), ref_b.tolist())
        ] if ref_a.shape[0] > 0 else [],
        metrics=bundle,
        transform_info={
            'matcher': matcher_name,
            'mode': 'piecewise_affine' if len(tile_affines) > 1 else 'global_affine',
            'matrix': M_fit.tolist() if hasattr(M_fit, 'tolist') else M_fit,
            'condition_number': bundle.get('condition_number_kappa', 'UNMEASURED'),
            'matrix_stable': bundle.get('condition_number_kappa', 'UNMEASURED') != 'UNMEASURED' and float(bundle.get('condition_number_kappa', 0)) < 2000,
            'escalated_to_tier2': escalated,
        },
        gsd_product=cfg.get('gsd_m', 0.25),
        gsd_reference=cfg.get('gsd_m', 0.25),
        output_dir=out_dir,
    )
    export_dossier_json(dossier, os.path.join(out_dir, "dossier.json"))
    export_dossier(dossier, out_dir)

    print(json.dumps({"trust_flag": flag, "rmse_px": rm.get("rmse_px"),
                      "runtime_s": round(runtime_s, 2),
                      "export_note": export_note}, indent=1))
    return bundle


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 6:
        print(__doc__)
        sys.exit(2)
    main(*sys.argv[1:])
