#!/usr/bin/env python
"""
PRADAN Dataset Benchmark Script

Batch-processes Chandrayaan-2 PDS4 image pairs from a local folder.
Outputs a summary benchmark_results.csv tracking:
Pair_ID, Inlier_Count, Spatial_Coverage_Pct, RMSE_px, RMSE_m, Trust_Status
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chandra_align.ingest import assert_pair_same_crs
from chandra_align import matcher as mm
from chandra_align.photometry import shadow_saturation_mask
from chandra_align.refine import verify_magsac, refine_subpixel_ncc, uniformity_score
from chandra_align.trust import evaluate, trust_flag, calibration_status
from chandra_align.metrics import inlier_stats, metrics_bundle, residual_map
from chandra_align.warp import (export_cog, fit_tile_affines, warp_piecewise_affine,
                                write_match_points_geojson, write_metrics_json)
from chandra_align.utils import load_config
from chandra_align.testing import write_tiff


def find_image_pairs(data_dir: Path, pattern: str = "*.img") -> list:
    """
    Find matching OHRC/NAC image pairs in a directory.
    
    Args:
        data_dir: Directory containing PDS4 image files
        pattern: File pattern to match
        
    Returns:
        List of (product_path, reference_path) tuples
    """
    pairs = []
    # For now, look for existing pairs in download_log.csv
    log_path = Path("data/download_log.csv")
    if not log_path.exists():
        return []
    
    import csv
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            local_path = Path(row['local_path'])
            if local_path.exists():
                # Look for a matching reference
                run_id = row['run_id']
                # Simple heuristic: pair by run_id
                pairs.append({
                    'product_id': row['product_id'],
                    'run_id': run_id,
                    'product_path': local_path,
                    'solar_azimuth_deg': float(row.get('solar_azimuth_deg', 'UNMEASURED')),
                    'solar_elevation_deg': float(row.get('solar_elevation_deg', 'UNMEASURED')),
                })
    
    return pairs


def process_pair(product_path: Path, ref_path: Path, config_path: str, 
                 output_dir: Path, run_id: str) -> dict:
    """
    Process a single image pair through the full pipeline.
    
    Returns:
        Dictionary with benchmark metrics
    """
    try:
        cfg = load_config(config_path)
        
        # Stage 1: ingest + CRS guard
        band_a, band_b, meta_a, meta_b, crs = assert_pair_same_crs(ref_path, product_path)
        
        # photometry mask
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
        M_fit, rm = evaluate(ref_a, ref_b)
        
        tile_affines = fit_tile_affines(shape, ref_a, ref_b,
                                        tile_size=tuple(cfg["tile_size"]),
                                        overlap=float(cfg["tile_overlap"]))
        warped = warp_piecewise_affine(band_a, tile_affines,
                                       blend_width_px=int(cfg["warp"]["blend_width_px"]))
        
        # Compute metrics
        runtime_s = time.time() - t0
        flag = trust_flag(rm.get("rmse_px", "UNMEASURED"), inlier_ratio, uni,
                          cfg["trust"])
        
        # Get GSD info
        gsd_product = 0.25  # OHRC default
        gsd_reference = 0.5   # NAC default
        
        rmse_pixels = rm.get("rmse_px", "UNMEASURED")
        rmse_meters = "UNMEASURED"
        if rmse_pixels != "UNMEASURED" and isinstance(rmse_pixels, (int, float)):
            rmse_meters = rmse_pixels * gsd_reference
        
        return {
            'pair_id': Path(product_path).stem,
            'product_id': Path(product_path).stem,
            'reference_id': Path(ref_path).stem,
            'acquisition_date': datetime.now().strftime('%Y-%m-%d'),
            'solar_azimuth_delta': 'UNMEASURED',
            'solar_elevation_delta': 'UNMEASURED',
            'gsd_ratio': gsd_product / gsd_reference if gsd_reference > 0 else 'UNMEASURED',
            'inliers': int(inl_a.shape[0]) if inl_a is not None else 0,
            'raw_matches': int(n_raw),
            'inlier_ratio': inlier_ratio,
            'spatial_coverage_pct': uni * 100 if uni != 'UNMEASURED' else 0,
            'uniformity_score': uni if uni != 'UNMEASURED' else 0,
            'rmse_px': rmse_pixels,
            'rmse_m': rmse_meters,
            'trust_flag': flag,
            'matcher': matcher_name,
            'mode': 'piecewise_affine',
            'condition_number': 'UNMEASURED',
            'runtime_s': runtime_s,
            'quality_flags': []
        }
        
    except Exception as e:
        return {
            'pair_id': Path(product_path).stem,
            'product_id': Path(product_path).stem,
            'reference_id': Path(ref_path).stem,
            'acquisition_date': datetime.now().strftime('%Y-%m-%d'),
            'solar_azimuth_delta': 'UNMEASURED',
            'solar_elevation_delta': 'UNMEASURED',
            'gsd_ratio': 'UNMEASURED',
            'inliers': 0,
            'raw_matches': 0,
            'inlier_ratio': 0,
            'spatial_coverage_pct': 0,
            'uniformity_score': 0,
            'rmse_px': 'UNMEASURED',
            'rmse_m': 'UNMEASURED',
            'trust_flag': 'ERROR',
            'matcher': 'ERROR',
            'mode': 'ERROR',
            'condition_number': 'UNMEASURED',
            'runtime_s': 0,
            'quality_flags': [f'ERROR: {str(e)}']
        }


def run_benchmark(data_dir: str, config_path: str, output_dir: str, 
                  max_pairs: int = None) -> list:
    """
    Run benchmark on all pairs in data directory.
    
    Args:
        data_dir: Directory with image pairs
        config_path: Path to modality config YAML
        output_dir: Output directory for results
        max_pairs: Maximum number of pairs to process
        
    Returns:
        List of result dictionaries
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Find pairs
    pairs = find_image_pairs(data_path)
    
    if max_pairs:
        pairs = pairs[:max_pairs]
    
    results = []
    
    for i, pair in enumerate(pairs):
        print(f"\nProcessing pair {i+1}/{len(pairs)}: {pair['product_id']}")
        
        # For now, use synthetic reference for testing
        # In real usage, would find matching reference image
        product_path = pair['product_path']
        
        # For benchmark, use synthetic reference if no real reference found
        ref_path = str(data_path / "synthetic_ref.tif")
        
        # Create synthetic reference if doesn't exist
        if not Path(ref_path).exists():
            from chandra_align.testing import make_pair_shift, write_tiff
            # Create a synthetic reference matching the product
            from chandra_align.ingest import read_image
            band, meta = read_image(pair['product_path'])
            ref, mov, _ = make_pair_shift(dx=3.2, dy=-1.7, seed=42)
            ref = ref.astype(np.float64)
            write_tiff(ref_path, ref, crs="EPSG:4326")
        
        # Process pair
        result = process_pair(
            product_path=product_path,
            ref_path=ref_path,
            config_path=config_path,
            output_dir=Path(output_dir),
            run_id=f"bench_{Path(product_path).stem}"
        )
        
        results.append(result)
        
        print(f"  Inliers: {result['inliers']}, "
              f"RMSE: {result['rmse_px']} px / {result['rmse_m']} m, "
              f"Trust: {result['trust_flag']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Chandra-Align PRADAN Benchmark')
    parser.add_argument('--data-dir', required=True, help='Directory with image data')
    parser.add_argument('--config', default='config/ohrc.yaml', help='Modality config YAML')
    parser.add_argument('--output-dir', default='benchmark_output', help='Output directory')
    parser.add_argument('--max-pairs', type=int, help='Maximum pairs to process')
    parser.add_argument('--output-csv', default='benchmark_results.csv', help='Output CSV path')
    
    args = parser.parse_args()
    
    print(f"Starting Chandra-Align PRADAN Benchmark")
    print(f"Data dir: {args.data_dir}")
    print(f"Config: {args.config}")
    print(f"Output dir: {args.output_dir}")
    print(f"Max pairs: {args.max_pairs or 'all'}")
    
    # Run benchmark
    results = run_benchmark(
        data_dir=args.data_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        max_pairs=args.max_pairs
    )
    
    # Write CSV
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'Pair_ID', 'Product_ID', 'Reference_ID', 'Acquisition_Date',
        'Solar_Azimuth_Delta_deg', 'Solar_Elevation_Delta_deg',
        'GSD_Ratio', 'Inlier_Count', 'Raw_Matches', 'Inlier_Ratio',
        'Spatial_Coverage_Pct', 'Uniformity_Score', 'RMSE_px', 'RMSE_m',
        'Trust_Status', 'Matcher_Used', 'Transformation_Mode',
        'Condition_Number', 'Runtime_Seconds', 'Quality_Flags'
    ]
    
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'Pair_ID': r['pair_id'],
                'Product_ID': r['product_id'],
                'Reference_ID': r['reference_id'],
                'Acquisition_Date': r['acquisition_date'],
                'Solar_Azimuth_Delta_deg': r['solar_azimuth_delta'],
                'Solar_Elevation_Delta_deg': r['solar_elevation_delta'],
                'GSD_Ratio': r['gsd_ratio'],
                'Inlier_Count': r['inliers'],
                'Raw_Matches': r['raw_matches'],
                'Inlier_Ratio': r['inlier_ratio'],
                'Spatial_Coverage_Pct': r['spatial_coverage_pct'],
                'Uniformity_Score': r['uniformity_score'],
                'RMSE_px': r['rmse_px'],
                'RMSE_m': r['rmse_m'],
                'Trust_Status': r['trust_flag'],
                'Matcher_Used': r['matcher'],
                'Transformation_Mode': r['mode'],
                'Condition_Number': r['condition_number'],
                'Runtime_Seconds': r['runtime_s'],
                'Quality_Flags': '; '.join(r['quality_flags'])
            })
    
    print(f"\nBenchmark complete! Results saved to {args.output_csv}")
    print(f"Processed {len(results)} pairs")
    
    # Print summary
    trusted = sum(1 for r in results if r['trust_flag'] == 'Trusted')
    print(f"Trusted: {trusted}/{len(results)}")


def create_benchmark_report(
    results: List[Dict],
    output_path: str
) -> str:
    """
    Create benchmark CSV report from list of results.
    
    Args:
        results: List of benchmark result dictionaries
        output_path: Output CSV path
        
    Returns:
        Path to created CSV
    """
    import csv
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'Pair_ID', 'Product_ID', 'Reference_ID', 'Acquisition_Date',
        'Solar_Azimuth_Delta_deg', 'Solar_Elevation_Delta_deg',
        'GSD_Ratio', 'Inlier_Count', 'Raw_Matches', 'Inlier_Ratio',
        'Spatial_Coverage_Pct', 'Uniformity_Score', 'RMSE_px', 'RMSE_m',
        'Trust_Status', 'Matcher_Used', 'Transformation_Mode',
        'Condition_Number', 'Runtime_Seconds', 'Quality_Flags'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'Pair_ID': r.get('pair_id', ''),
                'Product_ID': r.get('product_id', ''),
                'Reference_ID': r.get('reference_id', ''),
                'Acquisition_Date': r.get('acquisition_date', ''),
                'Solar_Azimuth_Delta_deg': r.get('solar_azimuth_delta', 'UNMEASURED'),
                'Solar_Elevation_Delta_deg': r.get('solar_elevation_delta', 'UNMEASURED'),
                'GSD_Ratio': r.get('gsd_ratio', 'UNMEASURED'),
                'Inlier_Count': r.get('inliers', 0),
                'Raw_Matches': r.get('raw_matches', 0),
                'Inlier_Ratio': r.get('inlier_ratio', 0),
                'Spatial_Coverage_Pct': r.get('spatial_coverage_pct', 0),
                'Uniformity_Score': r.get('uniformity_score', 0),
                'RMSE_px': r.get('rmse_px', 'UNMEASURED'),
                'RMSE_m': r.get('rmse_m', 'UNMEASURED'),
                'Trust_Status': r.get('trust_flag', 'UNMEASURED'),
                'Matcher_Used': r.get('matcher', ''),
                'Transformation_Mode': r.get('mode', ''),
                'Condition_Number': r.get('condition_number', 'UNMEASURED'),
                'Runtime_Seconds': r.get('runtime_s', 0),
                'Quality_Flags': '; '.join(r.get('quality_flags', []))
            })
    
    return str(output_path)


def _main():
    import time
    main()


if __name__ == "__main__":
    _main()