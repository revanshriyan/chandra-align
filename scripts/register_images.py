#!/usr/bin/env python3
"""Register Chandrayaan-2 OHRC, TMC-2, and IIRS rasters to discovered NAC tiles.

The prototype reads one raster band at a time, builds bounded-resolution
previews, and normalizes source/reference pixel scales to a common nominal GSD
before SIFT/FLANN matching. Matching previews use percentile contrast stretch
and CLAHE; OHRC additionally blends Sobel edge magnitude with reflectance.
Partial-affine RANSAC uses a sensor-specific threshold (3 px for OHRC/TMC-2,
4 px for IIRS), with homography as fallback, and exports require at least eight inliers. This limits full-cube
materialization but does not enforce a hard process-RAM cap.

Recorded visual benchmark snapshots and their run metadata are listed in
``BENCHMARK_PRESETS``. RANSAC inlier counts are representative measurements,
not guaranteed constants across OpenCV builds or reruns.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling


RAW_ROOT = Path("data/raw")
OUTPUT_ROOT = Path("data/processed")
DEFAULT_MAX_DIMENSION = 12000
MIN_MATCHES = 8
LOWE_RATIO = 0.75
MIN_INLIERS = 8
NAC_GSD_M_PER_PX = 0.5
SOURCE_GSD_M_PER_PX = {"ohrc": 0.25, "tmc2": 5.0, "iirs": 80.0}

SOURCES = {
    "ohrc": Path(
        "data/raw/ohrc/data/calibrated/20241115/"
        "ch2_ohr_ncp_20241115T1326321339_d_img_d18.img"
    ),
    "tmc2": Path(
        "data/raw/tmc2/data/calibrated/20260813/"
        "ch2_tmc_ncn_20260813T0627378557_d_img_d18.img"
    ),
    "iirs": Path(
        "data/raw/iirs/data/derived/20250729/"
        "ch2_iir_ndi_20250729T0936115604_d_rfl_d18_srd.qub"
    ),
}

OUTPUTS = {
    "ohrc": "aligned_ohrc_to_nac.png",
    "tmc2": "aligned_tmc2_to_nac.png",
    "iirs": "aligned_iirs_to_nac.png",
}

# Recorded run snapshots. ``artifact`` is tracked for README rendering;
# ``output`` is generated locally under the ignored data/processed directory.
BENCHMARK_PRESETS = {
    "ohrc": {
        "pair": "OHRC -> LROC NAC",
        "source_gsd_m_per_px": 0.25,
        "reference_gsd_m_per_px": 0.5,
        "model": "partial-affine",
        "ransac_threshold_px": 3.0,
        "clahe_clip_limit": 3.0,
        "contrast_percentiles": (2.0, 98.0),
        "recorded_inliers": 8,
        "output": "data/processed/aligned_ohrc_to_nac.png",
        "artifact": "docs/assets/benchmarks/aligned_ohrc_to_nac.png",
        "shape_hw": (12000, 582),
    },
    "tmc2": {
        "pair": "TMC-2 -> LROC NAC",
        "source_gsd_m_per_px": 5.0,
        "reference_gsd_m_per_px": 0.5,
        "model": "partial-affine",
        "ransac_threshold_px": 3.0,
        "clahe_clip_limit": 3.0,
        "contrast_percentiles": (2.0, 98.0),
        "recorded_inliers": 8,
        "output": "data/processed/aligned_tmc2_to_nac.png",
        "artifact": "docs/assets/benchmarks/aligned_tmc2_to_nac.png",
        "shape_hw": (5222, 253),
    },
    "iirs": {
        "pair": "IIRS band 125 -> LROC NAC",
        "band": 125,
        "band_center_wavelength_nm": 2802.0,
        "source_gsd_m_per_px": 80.0,
        "reference_gsd_m_per_px": 0.5,
        "model": "partial-affine",
        "ransac_threshold_px": 4.0,
        "clahe_clip_limit": 4.5,
        "contrast_percentiles": (1.0, 99.0),
        "recorded_inliers": 13,
        "output": "data/processed/aligned_iirs_to_nac.png",
        "artifact": "docs/assets/benchmarks/aligned_iirs_to_nac.png",
        "shape_hw": (326, 32),
    },
}

REFERENCE_DIRECTORIES = {
    "ohrc": RAW_ROOT / "nac" / "ch2_ohr_ncp_20241115T1326321339_d_img_d18",
    "tmc2": RAW_ROOT / "nac" / "tmc2_ref",
    "iirs": RAW_ROOT / "nac" / "iirs_ref",
}

RASTER_SUFFIXES = {".img", ".qub", ".tif", ".tiff"}
PDS_TYPES = {
    "unsignedbyte": np.dtype("u1"),
    "unsignedlsb2": np.dtype("<u2"),
    "unsignedmsb2": np.dtype(">u2"),
    "signedlsb2": np.dtype("<i2"),
    "signedmsb2": np.dtype(">i2"),
    "signedlsb4": np.dtype("<i4"),
    "signedmsb4": np.dtype(">i4"),
    "ieee754lsbsingle": np.dtype("<f4"),
    "ieee754msbsingle": np.dtype(">f4"),
    "ieee754lsbdouble": np.dtype("<f8"),
    "ieee754msbdouble": np.dtype(">f8"),
}


@dataclass
class RasterPreview:
    path: Path
    gray: np.ndarray
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray | None
    original_width: int
    original_height: int
    band_used: int


@dataclass
class MatchResult:
    candidate: Path
    matrix: np.ndarray
    inlier_count: int
    good_match_count: int
    model: str
    ratio_used: float
    source: RasterPreview
    reference: RasterPreview


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _pds4_layout(label_path: Path) -> tuple[np.dtype, tuple[int, ...], int]:
    """Read the basic array layout from a detached PDS4 XML label."""
    root = ET.parse(label_path).getroot()
    array = next(
        (node for node in root.iter() if _tag(node) in {"Array_2D_Image", "Array_3D_Image", "Array"}),
        None,
    )
    if array is None:
        raise ValueError(f"No PDS4 image array found in {label_path}")

    type_node = next((node for node in array.iter() if _tag(node) == "data_type"), None)
    if type_node is None or not type_node.text:
        raise ValueError(f"PDS4 data_type is missing in {label_path}")
    dtype = PDS_TYPES.get(type_node.text.strip().lower())
    if dtype is None:
        raise ValueError(f"Unsupported PDS4 data type {type_node.text!r} in {label_path}")

    axes: dict[int, tuple[str, int]] = {}
    for axis in (node for node in array.iter() if _tag(node) == "Axis_Array"):
        values = {_tag(child): (child.text or "").strip() for child in axis}
        try:
            axes[int(values["sequence_number"])] = (values["axis_name"].lower(), int(values["elements"]))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid PDS4 axis definition in {label_path}") from exc
    if not axes:
        raise ValueError(f"No PDS4 axes found in {label_path}")

    ordered = [axes[index] for index in sorted(axes)]
    names = [name for name, _ in ordered]
    dimensions = [size for _, size in ordered]
    if {"line", "sample"}.issubset(names):
        by_name = dict(zip(names, dimensions))
        shape = ((by_name["band"],) if "band" in by_name else ()) + (
            by_name["line"], by_name["sample"]
        )
    else:
        raise ValueError(f"PDS4 axes do not identify line/sample dimensions in {label_path}")

    offset_node = next((node for node in array.iter() if _tag(node) == "offset"), None)
    offset = int((offset_node.text or "0").strip()) if offset_node is not None else 0
    return dtype, tuple(shape), offset


def _raw_pds_preview(path: Path, max_dimension: int, band_index: int | None) -> tuple[np.ndarray, int, int, int]:
    label_path = path.with_suffix(".xml")
    if not label_path.is_file():
        raise ValueError(f"Rasterio could not open {path}, and no detached PDS4 label exists beside it")
    dtype, shape, offset = _pds4_layout(label_path)
    if len(shape) == 3:
        band_used = min(max(1, band_index or (shape[0] // 2)), shape[0])
        memmap = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)
        plane = memmap[band_used - 1]
    else:
        band_used = 1
        memmap = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)
        plane = memmap

    height, width = map(int, plane.shape)
    factor = max(1, math.ceil(max(height, width) / max_dimension))
    sampled = np.asarray(plane[::factor, ::factor])
    if sampled.shape != (height, width):
        # Average downsampling is preferable when the strided preview is larger
        # than requested because of a non-integral scale factor.
        out_h = max(1, round(height / factor))
        out_w = max(1, round(width / factor))
        sampled = cv2.resize(sampled, (out_w, out_h), interpolation=cv2.INTER_AREA)
    del plane, memmap
    return sampled, width, height, band_used


def _read_band(path: Path, max_dimension: int, band_index: int | None = None) -> tuple[np.ndarray, int, int, int]:
    """Read one raster band at bounded resolution, falling back to PDS4 raw data."""
    raster_error: Exception | None = None
    open_paths = [path]
    companion = path.with_suffix(".xml")
    if companion.is_file() and companion != path:
        open_paths.append(companion)

    for open_path in open_paths:
        try:
            with rasterio.open(open_path) as dataset:
                width, height, count = dataset.width, dataset.height, dataset.count
                band = min(max(1, band_index or (count // 2 if count > 1 else 1)), count)
                scale = min(1.0, max_dimension / max(width, height))
                out_width = max(1, int(round(width * scale)))
                out_height = max(1, int(round(height * scale)))
                raster = dataset.read(
                    band,
                    out_shape=(out_height, out_width),
                    resampling=Resampling.average if scale < 1 else Resampling.nearest,
                    masked=True,
                )
                values = np.asarray(raster.astype(np.float32).filled(np.nan), dtype=np.float32)
                return values, width, height, band
        except (rasterio.errors.RasterioError, OSError, ValueError) as exc:
            raster_error = exc

    try:
        return _raw_pds_preview(path, max_dimension, band_index)
    except Exception as raw_error:
        detail = f"Rasterio: {raster_error}; raw PDS fallback: {raw_error}"
        raise RuntimeError(f"Unable to read raster {path}: {detail}") from raw_error


def _to_gray8(values: np.ndarray, percentiles: tuple[float, float] = (2.0, 98.0)) -> np.ndarray:
    """Stretch finite values to uint8 using the requested percentile range."""
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = np.nanpercentile(array[finite], percentiles)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = np.clip((array - low) * (255.0 / (high - low)), 0.0, 255.0)
    scaled[~finite] = 0
    return np.rint(scaled).astype(np.uint8)


def _preprocess_for_matching(gray: np.ndarray, sensor: str) -> np.ndarray:
    """Apply local illumination normalization and optional OHRC rim emphasis."""
    clip_limit = 4.5 if sensor == "iirs" else 3.0
    equalized = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(gray)
    if sensor != "ohrc":
        return equalized

    reflectance = equalized.astype(np.float32)
    gx = cv2.Sobel(reflectance, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(reflectance, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    high = float(np.percentile(magnitude, 98.0)) if magnitude.size else 0.0
    if high > 0.0:
        magnitude = np.clip(magnitude * (255.0 / high), 0.0, 255.0)
    else:
        magnitude.fill(0.0)
    edges = np.rint(magnitude).astype(np.uint8)
    return cv2.addWeighted(equalized, 0.6, edges, 0.4, 0.0)


def load_preview(
    path: Path,
    max_dimension: int,
    band_index: int | None = None,
    sensor: str = "",
    gsd_downsample: float = 1.0,
) -> RasterPreview:
    values, width, height, band_used = _read_band(path, max_dimension, band_index)
    percentiles = BENCHMARK_PRESETS.get(sensor, {}).get("contrast_percentiles", (2.0, 98.0))
    gray = _to_gray8(values, percentiles)
    if gsd_downsample > 1.0:
        resized_width = max(1, int(round(gray.shape[1] / gsd_downsample)))
        resized_height = max(1, int(round(gray.shape[0] / gsd_downsample)))
        gray = cv2.resize(gray, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    gray = _preprocess_for_matching(gray, sensor)
    if gray.ndim != 2 or min(gray.shape) < 2:
        raise ValueError(f"Raster preview is degenerate: {path} has preview shape {gray.shape}")
    return RasterPreview(path, gray, [], None, width, height, band_used)


def discover_rasters(root: Path) -> list[Path]:
    """Discover raster files dynamically, including case variants of PDS IMG."""
    discovered: set[Path] = set()
    patterns = ("*.[iI][mM][gG]", "*.img", "*.IMG", "*.qub", "*.QUB", "*.tif", "*.tiff")
    for pattern in patterns:
        discovered.update(path for path in root.rglob(pattern) if path.is_file())
    return sorted(discovered, key=lambda path: str(path).casefold())


def discover_references(sensor: str, raw_root: Path, source: Path) -> list[Path]:
    routed_dir = REFERENCE_DIRECTORIES[sensor]
    candidates = discover_rasters(routed_dir) if routed_dir.is_dir() else []
    if not candidates:
        fallback_dir = raw_root / "nac"
        candidates = discover_rasters(fallback_dir) if fallback_dir.is_dir() else []
    candidates = [
        path for path in candidates
        if path.resolve() != source.resolve() and path.suffix.lower() in RASTER_SUFFIXES
    ]
    return sorted(candidates, key=lambda path: str(path).casefold())


def gsd_pyramid_factors(sensor: str) -> tuple[float, float]:
    """Return (source_downsample, reference_downsample) for a sensor pair."""
    if sensor not in SOURCE_GSD_M_PER_PX:
        raise ValueError(f"Unknown sensor {sensor!r}; expected one of {tuple(SOURCE_GSD_M_PER_PX)}")
    source_gsd = SOURCE_GSD_M_PER_PX[sensor]
    return (
        max(1.0, NAC_GSD_M_PER_PX / source_gsd),
        max(1.0, source_gsd / NAC_GSD_M_PER_PX),
    )


def is_registration_acceptable(matrix: np.ndarray | None, inlier_count: int) -> bool:
    """Apply the output safety gate used for a candidate geometric fit."""
    return matrix is not None and np.isfinite(matrix).all() and inlier_count >= MIN_INLIERS


def extract_features(preview: RasterPreview, detector: cv2.SIFT) -> RasterPreview:
    if preview.descriptors is not None:
        return preview
    keypoints, descriptors = detector.detectAndCompute(preview.gray, None)
    preview.keypoints = keypoints or []
    preview.descriptors = descriptors
    return preview


def match_and_estimate(
    source: RasterPreview, reference: RasterPreview, ransac_threshold_px: float = 3.0
) -> tuple[np.ndarray | None, int, int, float, str]:
    if source.descriptors is None or reference.descriptors is None:
        return None, 0, 0, LOWE_RATIO, "none"
    if len(source.descriptors) < 2 or len(reference.descriptors) < 2:
        return None, 0, 0, LOWE_RATIO, "none"

    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=64)
    )
    try:
        pairs = matcher.knnMatch(source.descriptors, reference.descriptors, k=2)
    except cv2.error:
        return None, 0, 0, LOWE_RATIO, "none"
    best: tuple[np.ndarray | None, int, int, float, str] = (None, 0, 0, LOWE_RATIO, "none")
    for ratio in (LOWE_RATIO, 0.80):
        good = [
            first for pair in pairs if len(pair) == 2
            for first, second in [pair]
            if first.distance < ratio * second.distance
        ]
        if len(good) < MIN_MATCHES:
            if len(good) > best[2]:
                best = (None, 0, len(good), ratio, "none")
            continue

        source_points = np.float32([source.keypoints[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
        reference_points = np.float32([reference.keypoints[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
        affine, affine_mask = cv2.estimateAffinePartial2D(
            source_points,
            reference_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold_px,
        )
        affine_inliers = int(np.sum(affine_mask)) if affine_mask is not None else 0
        candidate_result = (
            affine if affine is not None and np.isfinite(affine).all() else None,
            affine_inliers,
            len(good),
            ratio,
            "partial-affine",
        )
        if candidate_result[1] < MIN_INLIERS:
            homography, homography_mask = cv2.findHomography(
                source_points,
                reference_points,
                cv2.RANSAC,
                ransac_threshold_px,
            )
            homography_inliers = int(np.sum(homography_mask)) if homography_mask is not None else 0
            if homography is not None and np.isfinite(homography).all() and homography_inliers > candidate_result[1]:
                candidate_result = (
                    homography,
                    homography_inliers,
                    len(good),
                    ratio,
                    "homography",
                )
        if candidate_result[1] > best[1] or (
            candidate_result[1] == best[1] and candidate_result[2] > best[2]
        ):
            best = candidate_result
        if best[1] >= MIN_INLIERS:
            break
    return best


def register_sensor(
    sensor: str,
    source_path: Path,
    references: Iterable[Path],
    output_path: Path,
    max_dimension: int,
    detector: cv2.SIFT,
    preview_cache: dict[tuple[Path, int | None, str], RasterPreview],
) -> MatchResult | None:
    # Use the designated IIRS band for the multisensor registration product.
    band = BENCHMARK_PRESETS["iirs"]["band"] if sensor == "iirs" else None
    source_gsd = SOURCE_GSD_M_PER_PX[sensor]
    source_downsample, reference_downsample = gsd_pyramid_factors(sensor)
    source_read_dimension = int(math.ceil(max_dimension * source_downsample))
    reference_read_dimension = int(math.ceil(max_dimension * reference_downsample))
    source_key = (source_path, band, sensor)
    if source_key not in preview_cache:
        preview_cache[source_key] = load_preview(
            source_path,
            source_read_dimension,
            band,
            sensor,
            gsd_downsample=source_downsample,
        )
    source = extract_features(preview_cache[source_key], detector)
    print(
        f"\n[{sensor.upper()}] Source: {source_path}"
        f" | {source.original_width}x{source.original_height}"
        f" | preview {source.gray.shape[1]}x{source.gray.shape[0]}"
        f" | band {source.band_used}"
        f" | normalized GSD {max(source_gsd, NAC_GSD_M_PER_PX):g} m/px"
        f" | keypoints {len(source.keypoints)}"
    )

    best: MatchResult | None = None
    for index, candidate in enumerate(references, start=1):
        cache_key = (candidate, None, sensor)
        try:
            if cache_key not in preview_cache:
                preview_cache[cache_key] = load_preview(
                    candidate,
                    reference_read_dimension,
                    sensor=sensor,
                    gsd_downsample=reference_downsample,
                )
            reference = extract_features(preview_cache[cache_key], detector)
            matrix, inliers, good_matches, ratio, model = match_and_estimate(
                source, reference, BENCHMARK_PRESETS[sensor]["ransac_threshold_px"]
            )
            print(
                f"  [{index}] {candidate}: {len(reference.keypoints)} keypoints, "
                f"{good_matches} Lowe-{ratio:.2f} matches, {inliers} {model} RANSAC inliers"
            )
            if is_registration_acceptable(matrix, inliers) and (best is None or inliers > best.inlier_count):
                best = MatchResult(candidate, matrix, inliers, good_matches, model, ratio, source, reference)
        except Exception as exc:
            print(f"  [{index}] {candidate}: skipped ({type(exc).__name__}: {exc})")

    if best is None:
        print(f"  No valid registration found for {sensor.upper()}; output not written.")
        return None

    warp = cv2.warpAffine if best.model == "partial-affine" else cv2.warpPerspective
    warped = warp(
        best.source.gray,
        best.matrix,
        (best.reference.gray.shape[1], best.reference.gray.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), warped):
        raise OSError(f"OpenCV could not write {output_path}")
    print(
        f"  Selected reference: {best.candidate}\n"
        f"  Match summary: {len(best.source.keypoints)} source keypoints, "
        f"{len(best.reference.keypoints)} reference keypoints, "
        f"{best.good_match_count} Lowe-{best.ratio_used:.2f} matches, "
        f"{best.inlier_count} {best.model} inliers\n"
        f"  Output: {output_path} ({warped.shape[1]}x{warped.shape[0]} grayscale PNG)"
    )
    return best


def run(max_dimension: int = DEFAULT_MAX_DIMENSION) -> int:
    if max_dimension < 512:
        raise ValueError("--max-dimension must be at least 512")
    if not RAW_ROOT.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    cv2.setNumThreads(max(1, min(8, os.cpu_count() or 1)))
    detector = cv2.SIFT_create(nfeatures=10000)
    all_rasters = discover_rasters(RAW_ROOT)
    print(f"Discovered {len(all_rasters)} raster candidate(s) under {RAW_ROOT}.")
    cache: dict[tuple[Path, int | None, str], RasterPreview] = {}
    results: dict[str, MatchResult | None] = {}

    for sensor, source in SOURCES.items():
        if not source.is_file():
            print(f"\n[{sensor.upper()}] Missing source raster: {source}")
            results[sensor] = None
            continue
        references = discover_references(sensor, RAW_ROOT, source)
        if not references:
            print(f"\n[{sensor.upper()}] No dynamically discovered external reference rasters.")
            results[sensor] = None
            continue
        results[sensor] = register_sensor(
            sensor,
            source,
            references,
            OUTPUT_ROOT / OUTPUTS[sensor],
            max_dimension,
            detector,
            cache,
        )

    succeeded = sum(result is not None for result in results.values())
    print(f"\nRegistration complete: {succeeded}/{len(SOURCES)} sensor pair(s) registered.")
    return 0 if succeeded == len(SOURCES) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=DEFAULT_MAX_DIMENSION,
        help=f"Maximum width/height for feature-matching previews (default: {DEFAULT_MAX_DIMENSION})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args.max_dimension)
    except Exception as exc:
        print(f"Registration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
