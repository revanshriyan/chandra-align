"""
Dossier & Metrics Exporter

Generates comprehensive registration reports in JSON and GeoJSON formats.
Uses Pydantic v2 models for strict type safety.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

from src.schemas.models import (
    RegistrationDossier,
    BenchmarkRow,
    MatchPoint,
    MatchPointsGeoJSON,
    MatchPointFeature,
    TrustFlag,
    CalibrationStatus,
    NotTrustedReason,
    MatcherTier,
    TransformationMode,
    UNMEASURED,
    unmeasured_to_none,
)


def create_dossier(
    run_id: str,
    product_info: Dict,
    reference_info: Dict,
    match_points: List[Dict],
    metrics: Dict,
    transform_info: Dict,
    gsd_product: float,
    gsd_reference: float,
    output_dir: Optional[str] = None,
) -> RegistrationDossier:
    """
    Generate comprehensive registration dossier as a Pydantic model.

    Args:
        run_id: Unique run identifier
        product_info: Product image metadata
        reference_info: Reference image metadata
        match_points: List of match point dictionaries
        metrics: Metrics dictionary from pipeline
        transform_info: Transformation info from warping
        gsd_product: Product image GSD (meters/pixel)
        gsd_reference: Reference image GSD (meters/pixel)
        output_dir: Optional output directory

    Returns:
        RegistrationDossier Pydantic model instance
    """
    # Physical RMSE conversion
    rmse_pixels = metrics.get('rmse', {}).get('rmse_px', UNMEASURED)
    rmse_meters = UNMEASURED
    if rmse_pixels is not UNMEASURED and isinstance(rmse_pixels, (int, float)):
        rmse_meters = rmse_pixels * gsd_reference

    # Compute GSD ratio
    gsd_ratio = gsd_product / gsd_reference if gsd_reference > 0 else UNMEASURED

    # Convert match points to MatchPoint models
    mp_models = []
    for mp in match_points:
        mp_models.append(MatchPoint(
            x_ref=mp.get('x_ref', 0.0),
            y_ref=mp.get('y_ref', 0.0),
            x_mov=mp.get('x_mov', mp.get('x', 0.0)),
            y_mov=mp.get('y_mov', mp.get('y', 0.0)),
            refined=mp.get('refined', False),
            residual_px=mp.get('residual_px', 0.0),
        ))

    # Get transformation matrix
    matrix = transform_info.get('matrix')
    if isinstance(matrix, np.ndarray):
        matrix_list = matrix.tolist()
    elif isinstance(matrix, list):
        matrix_list = matrix
    else:
        matrix_list = []

    # Quality flags
    quality_flags = _generate_quality_flags(metrics, transform_info)

    dossier = RegistrationDossier(
        run_id=run_id,
        timestamp=datetime.utcnow(),
        product_image={
            'id': product_info.get('product_id', 'UNKNOWN'),
            'path': product_info.get('path', ''),
            'crs': product_info.get('crs', 'UNKNOWN'),
            'gsd_m': gsd_product,
            'size': product_info.get('size', {}),
            'solar_geometry': {
                'sun_azimuth_deg': product_info.get('solar_azimuth_deg', UNMEASURED),
                'sun_elevation_deg': product_info.get('sun_elevation_deg', UNMEASURED),
                'incidence_angle_deg': product_info.get('incidence_angle_deg', UNMEASURED),
            }
        },
        reference_image={
            'id': reference_info.get('product_id', 'UNKNOWN'),
            'path': reference_info.get('path', ''),
            'crs': reference_info.get('crs', 'UNKNOWN'),
            'gsd_m': gsd_reference,
            'size': reference_info.get('size', {}),
            'solar_geometry': {
                'sun_azimuth_deg': reference_info.get('solar_azimuth_deg', UNMEASURED),
                'sun_elevation_deg': reference_info.get('sun_elevation_deg', UNMEASURED),
                'incidence_angle_deg': reference_info.get('incidence_angle_deg', UNMEASURED),
            }
        },
        gsd_ratio=gsd_ratio,
        matcher_used=_parse_matcher_tier(transform_info.get('matcher')),
        transformation_mode=_parse_transformation_mode(transform_info.get('mode')),
        transformation_matrix=matrix_list,
        match_points=mp_models,
        match_count=len(mp_models),
        inlier_count=metrics.get('inliers', {}).get('inliers', 0),
        inlier_ratio=metrics.get('inliers', {}).get('inlier_ratio', 0.0),
        uniformity_score=metrics.get('uniformity_score', 0.0),
        spatial_coverage_pct=metrics.get('spatial_coverage_pct', 0.0),
        quadtree_entropy=metrics.get('quadtree_entropy', {}).get('normalized_entropy', 0.0),
        matrix_condition_number=transform_info.get('condition_number', UNMEASURED) if transform_info.get('condition_number') is not None else UNMEASURED,
        matrix_stable=transform_info.get('matrix_stable', False),
        rmse_pixels=rmse_pixels,
        rmse_meters=rmse_meters,
        rmse_x_pixels=metrics.get('rmse', {}).get('rmse_x_px', UNMEASURED),
        rmse_y_pixels=metrics.get('rmse', {}).get('rmse_y_px', UNMEASURED),
        trust_flag=_parse_trust_flag(metrics.get('trust_flag')),
        calibration_status=_parse_calibration_status(metrics.get('calibration_status')),
        approximation_flag=metrics.get('approximation_flag', True),
        escalated_to_tier2=transform_info.get('escalated_to_tier2', False),
        fallback_reason=transform_info.get('fallback_reason'),
        mesh_info=transform_info.get('mesh_info', {}),
        validation=transform_info.get('validation', {}),
        quality_flags=quality_flags,
    )

    return dossier


def _parse_matcher_tier(matcher: Optional[str]) -> MatcherTier:
    """Parse matcher string to MatcherTier enum."""
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


def _parse_transformation_mode(mode: Optional[str]) -> TransformationMode:
    """Parse mode string to TransformationMode enum."""
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


def _parse_trust_flag(flag: Optional[str]) -> TrustFlag:
    """Parse trust flag string to TrustFlag enum."""
    if flag == 'Trusted':
        return TrustFlag.TRUSTED
    return TrustFlag.NOT_TRUSTED


def _parse_calibration_status(status: Optional[str]) -> CalibrationStatus:
    """Parse calibration status string to CalibrationStatus enum."""
    if status == 'CALIBRATED':
        return CalibrationStatus.CALIBRATED
    return CalibrationStatus.UNCALIBRATED


def _generate_quality_flags(metrics: Dict, transform_info: Dict) -> List[NotTrustedReason]:
    """Generate quality flags based on metrics."""
    flags = []

    # Trust flag
    if metrics.get('trust_flag') != 'Trusted':
        flags.append(NotTrustedReason.LOW_TRUST)

    # Inlier ratio
    inlier_ratio = metrics.get('inliers', {}).get('inlier_ratio', 0)
    if inlier_ratio < 0.15:
        flags.append(NotTrustedReason.LOW_INLIER_RATIO)

    # Uniformity
    uniformity = metrics.get('uniformity_score', 0)
    if uniformity < 0.125:
        flags.append(NotTrustedReason.LOW_UNIFORMITY)

    # Spatial coverage
    coverage = metrics.get('spatial_coverage_pct', 0)
    if coverage < 15:
        flags.append(NotTrustedReason.INSUFFICIENT_COVERAGE)

    # RMSE
    rmse = metrics.get('rmse', {}).get('rmse_px', UNMEASURED)
    if isinstance(rmse, (int, float)) and rmse > 1.0:
        flags.append(NotTrustedReason.HIGH_RMSE)

    # Condition number
    cond = transform_info.get('condition_number', UNMEASURED)
    if isinstance(cond, (int, float)) and cond > 2000:
        flags.append(NotTrustedReason.ILL_CONDITIONED_MATRIX)

    # Fallback used
    if transform_info.get('mode') != 'piecewise_affine':
        flags.append(NotTrustedReason.TRANSFORM_FALLBACK_USED)

    return flags


def export_dossier_json(dossier: RegistrationDossier, output_path: str) -> str:
    """Export dossier to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict with UNMEASURED -> None for JSON
    data = dossier.model_dump()
    data = unmeasured_to_none(data)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    return str(output_path)


def export_match_points_geojson(
    match_points: List[MatchPoint],
    output_path: str,
    crs: str = 'EPSG:4326'
) -> str:
    """Export match points as GeoJSON FeatureCollection."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features = []
    for i, mp in enumerate(match_points):
        features.append(MatchPointFeature.from_match_point(mp, i))

    geojson = MatchPointsGeoJSON(
        features=features,
        crs={'type': 'name', 'properties': {'name': crs}}
    )

    data = geojson.model_dump()
    data = unmeasured_to_none(data)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    return str(output_path)


def export_dossier(
    dossier: RegistrationDossier,
    output_dir: str,
    base_name: str = 'dossier'
) -> Dict[str, str]:
    """
    Export complete dossier package (JSON + GeoJSON).

    Args:
        dossier: Complete dossier model
        output_dir: Output directory
        base_name: Base filename

    Returns:
        Dict of exported file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    run_id = dossier.run_id

    exported = {}

    # JSON dossier
    json_path = output_dir / f"{base_name}_{run_id}_{timestamp}.json"
    export_dossier_json(dossier, json_path)
    exported['json'] = str(json_path)

    # Match points GeoJSON
    if dossier.match_points:
        geojson_path = output_dir / f"match_points_{run_id}_{timestamp}.geojson"
        export_match_points_geojson(dossier.match_points, geojson_path)
        exported['geojson'] = str(geojson_path)

    return exported


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
            # Validate through BenchmarkRow model
            row_model = BenchmarkRow(
                pair_id=r.get('pair_id', ''),
                product_id=r.get('product_id', ''),
                reference_id=r.get('reference_id', ''),
                acquisition_date=r.get('acquisition_date', ''),
                solar_azimuth_delta_deg=r.get('solar_azimuth_delta', UNMEASURED),
                solar_elevation_delta_deg=r.get('solar_elevation_delta', UNMEASURED),
                gsd_ratio=r.get('gsd_ratio', UNMEASURED),
                inlier_count=r.get('inliers', 0),
                raw_matches=r.get('raw_matches', 0),
                inlier_ratio=r.get('inlier_ratio', 0.0),
                spatial_coverage_pct=r.get('spatial_coverage_pct', 0.0),
                uniformity_score=r.get('uniformity_score', 0.0),
                rmse_px=r.get('rmse_px', UNMEASURED),
                rmse_m=r.get('rmse_m', UNMEASURED),
                trust_status=_parse_trust_flag(r.get('trust_flag')),
                matcher_used=_parse_matcher_tier(r.get('matcher')),
                transformation_mode=_parse_transformation_mode(r.get('mode')),
                condition_number=r.get('condition_number', UNMEASURED),
                runtime_seconds=r.get('runtime_s', 0.0),
                quality_flags='; '.join(r.get('quality_flags', []))
            )

            # Convert to dict for CSV
            row = row_model.model_dump()
            row = unmeasured_to_none(row)
            writer.writerow(row)

    return str(output_path)