"""
CHANDRA-ALIGN: Lunar Cross-Sensor Registration Engine
Scientific Export Package Generator

Clean file export helpers:
- GCP Export (CSV)
- Homography Matrix & Metrics (JSON)
- GeoTIFF / PNG Alignment Output
"""

import numpy as np
import cv2
import json
import csv
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from chandra_align.features import bucket_ids


def _to_uint16_image(image: np.ndarray) -> np.ndarray:
    """Convert normalized floats or 8-bit pixels to 16-bit without overflow."""
    image = np.asarray(image)
    if image.dtype == np.uint16:
        return image
    if image.dtype == np.uint8:
        return image.astype(np.uint16) * 257

    values = np.nan_to_num(image.astype(np.float64, copy=False), nan=0.0,
                           posinf=65535.0, neginf=0.0)
    if values.size and values.max() <= 1.0:
        values = values * 65535.0
    elif values.size and values.max() <= 255.0:
        values = values * 257.0
    return np.clip(values, 0.0, 65535.0).astype(np.uint16)


@dataclass
class GCPRecord:
    """Ground Control Point record for CSV export."""
    point_id: int
    ref_x_px: float
    ref_y_px: float
    sec_x_px: float
    sec_y_px: float
    residual_px: float
    residual_m: float
    bucket_id: int


def export_gcp_csv(
    deformation_vectors: List[Any],
    output_path: str,
    pixel_scale_m: float = 0.25,
    include_header: bool = True,
    image_shape: Optional[Tuple[int, int]] = None,
    grid_shape: Tuple[int, int] = (8, 8)
) -> None:
    """
    Export matched inlier keypoints to CSV formatted as:
    point_id, ref_x, ref_y, sec_x, sec_y, residual_px, residual_m, bucket_id
    
    Args:
        deformation_vectors: List of objects with ref_x, ref_y, sec_x, sec_y, dx_px, dy_px, 
                            magnitude_px, magnitude_m, inlier_weight attributes
        output_path: Output CSV file path
        pixel_scale_m: Ground resolution in meters per pixel
        include_header: Whether to include CSV header row
    """
    vectors = list(deformation_vectors)
    if image_shape is None:
        max_x = max((float(v.ref_x) for v in vectors), default=0.0)
        max_y = max((float(v.ref_y) for v in vectors), default=0.0)
        image_shape = (max(1, int(np.ceil(max_y + 1))), max(1, int(np.ceil(max_x + 1))))
    ids = bucket_ids([(v.ref_x, v.ref_y) for v in vectors], image_shape, grid_shape) if vectors else []
    records = []
    for i, v in enumerate(vectors):
        record = GCPRecord(
            point_id=i,
            ref_x_px=float(v.ref_x),
            ref_y_px=float(v.ref_y),
            sec_x_px=float(v.sec_x),
            sec_y_px=float(v.sec_y),
            residual_px=float(v.magnitude_px),
            residual_m=float(v.magnitude_m),
            bucket_id=int(ids[i])
        )
        records.append(record)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        if include_header:
            writer.writerow([
                'point_id', 'ref_x', 'ref_y',
                'sec_x', 'sec_y', 'residual_px', 'residual_m', 'bucket_id'
            ])
        for r in records:
            writer.writerow([
                r.point_id, r.ref_x_px, r.ref_y_px,
                r.sec_x_px, r.sec_y_px, r.residual_px,
                r.residual_m, r.bucket_id
            ])


def export_homography_json(
    H: np.ndarray,
    metrics: Dict,
    output_path: str,
    sensor_name: str = "OHRC",
    pixel_scale_m: float = 0.25,
    spatial_entropy: Optional[float] = None,
    inlier_count: int = 0,
    matcher_name: str = "Unknown",
    trust_flag: str = "UNTRUSTED",
    uniformity_score: Optional[float] = None
) -> None:
    """
    Export homography matrix and metrics to JSON.
    
    Args:
        H: Homography matrix (3, 3) or affine (2, 3)
        metrics: Metrics dictionary
        output_path: Output JSON file path
        sensor_name: Sensor name for metadata
        pixel_scale_m: Ground resolution in meters per pixel
        spatial_entropy: Spatial uniformity score
        inlier_count: Number of inliers
        matcher_name: Feature matcher used
        trust_flag: Trust assessment flag
    """
    H = np.asarray(H, np.float64)
    
    # Ensure 3x3 matrix
    if H.shape == (2, 3):
        H_full = np.eye(3, dtype=np.float64)
        H_full[:2, :] = H
        H = H_full
    
    export_data = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sensor": sensor_name,
            "pixel_scale_m": pixel_scale_m,
            "matcher": matcher_name,
            "trust_flag": trust_flag
        },
        "homography_matrix": H.tolist(),
        "inlier_count": inlier_count,
        "spatial_entropy": float(spatial_entropy) if spatial_entropy is not None else "UNMEASURED",
        "spatial_uniformity": float(uniformity_score) if uniformity_score is not None else "UNMEASURED",
        "metrics_px": {
            "rmse_px": metrics.get("rmse_px", "UNMEASURED"),
            "mae_px": metrics.get("mae_px", "UNMEASURED"),
            "std_px": metrics.get("std_px", "UNMEASURED")
        },
        "metrics_m": {
            "rmse_m": metrics.get("rmse_m", "UNMEASURED"),
            "mae_m": metrics.get("mae_m", "UNMEASURED"),
            "std_m": metrics.get("std_m", "UNMEASURED")
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)


def export_alignment_geotiff(
    warped_secondary: np.ndarray,
    output_path: str,
    reference_geotransform: Optional[Tuple] = None,
    reference_projection: Optional[str] = None,
    pixel_scale_m: float = 0.25,
    compression: str = "LZW",
    preserve_bit_depth: bool = True
) -> None:
    """
    Export the transformed secondary image as GeoTIFF with spatial metadata.
    
    Args:
        warped_secondary: Warped secondary image (H, W) or (H, W, 3)
        output_path: Output GeoTIFF file path
        reference_geotransform: GDAL geotransform tuple (6 elements) from reference
        reference_projection: WKT projection string from reference
        pixel_scale_m: Ground resolution in meters per pixel (used if no geotransform)
        compression: TIFF compression method
        preserve_bit_depth: Preserve original bit depth (16-bit for lunar data)
    """
    # Ensure 2D grayscale
    if warped_secondary.ndim == 3:
        if warped_secondary.shape[2] == 3:
            # Convert RGB to grayscale using luminance
            warped_secondary = cv2.cvtColor(warped_secondary, cv2.COLOR_RGB2GRAY)
        else:
            warped_secondary = warped_secondary[:, :, 0]
    
    # Preserve bit depth if requested
    if preserve_bit_depth and warped_secondary.dtype != np.uint16:
        warped_secondary = _to_uint16_image(warped_secondary)
    elif warped_secondary.dtype != np.uint8:
        # Normalize to 8-bit
        warped_secondary = cv2.normalize(warped_secondary, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Build geotransform
    if reference_geotransform is not None:
        geotransform = reference_geotransform
    else:
        # Default: upper-left corner at (0, 0), pixel size = pixel_scale_m
        h, w = warped_secondary.shape[:2]
        geotransform = (0.0, pixel_scale_m, 0.0, 0.0, 0.0, -pixel_scale_m)
    
    # Build projection WKT
    if reference_projection is not None:
        projection = reference_projection
    else:
        # Default: Moon spherical projection (simplified)
        projection = (
            'PROJCS["Moon_Equirectangular",'
            'GEOGCS["GCS_Moon",DATUM["D_Moon",'
            'SPHEROID["Moon",1737400.0,0.0]],'
            'PRIMEM["Greenwich",0.0],'
            'UNIT["Degree",0.0174532925199433]],'
            'PROJECTION["Equirectangular"],'
            'PARAMETER["False_Easting",0.0],'
            'PARAMETER["False_Northing",0.0],'
            'PARAMETER["Central_Meridian",0.0],'
            'PARAMETER["Standard_Parallel_1",0.0],'
            'UNIT["Meter",1.0]]'
        )
    
    # Prefer tifffile, but keep TIFF downloads available in minimal installs.
    try:
        import tifffile
        px_x = abs(float(geotransform[1])) or float(pixel_scale_m)
        px_y = abs(float(geotransform[5])) or float(pixel_scale_m)
        tie_x, tie_y = float(geotransform[0]), float(geotransform[3])
        extratags = [
            (33550, 'd', 3, (px_x, px_y, 0.0), False),  # ModelPixelScaleTag
            (33922, 'd', 6, (0.0, 0.0, 0.0, tie_x, tie_y, 0.0), False),  # ModelTiepointTag
            (34735, 'H', 20, (1, 1, 0, 4, 1024, 0, 1, 1, 1025, 0, 1, 1,
                              2048, 0, 1, 32767, 3072, 0, 1, 32767), False),
        ]
        options = dict(photometric='minisblack', metadata={'geotransform': geotransform, 'projection': projection}, extratags=extratags)
        try:
            tifffile.imwrite(output_path, warped_secondary, compression=compression, **options)
        except (ValueError, KeyError, RuntimeError):
            # Some tifffile builds do not include optional compression codecs.
            tifffile.imwrite(output_path, warped_secondary, compression=None, **options)
    except ImportError:
        if not cv2.imwrite(str(output_path), warped_secondary):
            raise RuntimeError("Neither tifffile nor OpenCV could write the GeoTIFF fallback")


def export_alignment_png(
    warped_secondary: np.ndarray,
    output_path: str,
    bit_depth: int = 16
) -> None:
    """
    Export the transformed secondary image as PNG preserving bit depth.
    
    Args:
        warped_secondary: Warped secondary image
        output_path: Output PNG file path
        bit_depth: 8 or 16
    """
    warped_secondary = np.asarray(warped_secondary)
    if warped_secondary.size == 0:
        raise ValueError("Cannot export an empty image")
    if warped_secondary.ndim == 3:
        if warped_secondary.shape[2] == 3:
            warped_secondary = cv2.cvtColor(warped_secondary, cv2.COLOR_RGB2GRAY)
        else:
            warped_secondary = warped_secondary[:, :, 0]
    
    if bit_depth == 16:
        if warped_secondary.dtype != np.uint16:
            warped_secondary = _to_uint16_image(warped_secondary)
    else:
        warped_secondary = cv2.normalize(warped_secondary, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    if not cv2.imwrite(output_path, warped_secondary):
        raise OSError(f"OpenCV could not write PNG output: {output_path}")


def export_composite_visualization(
    ref_image: np.ndarray,
    warped_secondary: np.ndarray,
    diff_map: np.ndarray,
    output_path: str,
    layout: str = "horizontal"
) -> None:
    """
    Export side-by-side composite visualization.
    
    Args:
        ref_image: Reference image
        warped_secondary: Aligned secondary image
        diff_map: Absolute difference map
        output_path: Output file path
        layout: "horizontal" or "vertical"
    """
    # Normalize all to same range
    def normalize_img(img):
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img
    
    ref = normalize_img(ref_image)
    warped = normalize_img(warped_secondary)
    diff = normalize_img(diff_map)
    
    # Apply colormap to diff
    diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO)
    
    if layout == "horizontal":
        composite = np.hstack([ref, warped, diff_colored])
    else:
        composite = np.vstack([ref, warped, diff_colored])
    
    cv2.imwrite(output_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))


def export_full_package(
    ref_image: np.ndarray,
    warped_secondary: np.ndarray,
    diff_map: np.ndarray,
    deformation_vectors: List[Any],
    H: np.ndarray,
    metrics: Dict,
    output_dir: str,
    base_name: str = "chandra_align",
    pixel_scale_m: float = 0.25,
    sensor_name: str = "OHRC",
    spatial_entropy: Optional[float] = None,
    inlier_count: int = 0,
    total_matches: int = 0,
    matcher_name: str = "Unknown",
    trust_flag: str = "UNTRUSTED",
    reference_geotransform: Optional[Tuple] = None,
    reference_projection: Optional[str] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    grid_shape: Tuple[int, int] = (8, 8),
    spatial_uniformity: Optional[float] = None
) -> Dict[str, str]:
    """
    Export complete scientific package with all artifacts.
    
    Returns:
        Dictionary mapping artifact names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # 1. GCP CSV
    csv_path = output_dir / f"{base_name}_gcps.csv"
    export_gcp_csv(deformation_vectors, str(csv_path), pixel_scale_m,
                   image_shape=image_shape, grid_shape=grid_shape)
    paths["gcp_csv"] = str(csv_path)
    
    # 2. Homography + Metrics JSON
    json_path = output_dir / f"{base_name}_transform.json"
    export_homography_json(
        H, metrics, str(json_path), sensor_name, pixel_scale_m,
        spatial_entropy, inlier_count, matcher_name, trust_flag, spatial_uniformity
    )
    paths["transform_json"] = str(json_path)
    
    # 3. Warped secondary GeoTIFF
    tiff_path = output_dir / f"{base_name}_warped.tif"
    export_alignment_geotiff(
        warped_secondary, str(tiff_path), reference_geotransform,
        reference_projection, pixel_scale_m
    )
    paths["warped_geotiff"] = str(tiff_path)
    
    # 4. Warped secondary PNG (16-bit)
    png_path = output_dir / f"{base_name}_warped.png"
    export_alignment_png(warped_secondary, str(png_path), bit_depth=16)
    paths["warped_png"] = str(png_path)
    
    # 5. Composite visualization
    comp_path = output_dir / f"{base_name}_composite.png"
    export_composite_visualization(ref_image, warped_secondary, diff_map, str(comp_path))
    paths["composite_png"] = str(comp_path)
    
    return paths


__all__ = [
    "GCPRecord",
    "export_gcp_csv",
    "export_homography_json",
    "export_alignment_geotiff",
    "export_alignment_png",
    "export_composite_visualization",
    "export_full_package"
]
