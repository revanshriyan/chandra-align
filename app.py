"""
CHANDRA-ALIGN: Lunar Cross-Sensor Registration Engine
Gradio web application for Hugging Face Spaces deployment.

Scientific-grade photogrammetric workstation with:
- Illumination-invariant preprocessing (CLAHE, Shadow Masking, Wallis)
- Ground-resolution error translation
- Spatial deformation field visualization (Quiver + Error Distribution)
- Scientific GIS artifact export (CSV, JSON, GeoTIFF, PNG)
"""

import os
import sys
import cv2
import numpy as np
import gradio as gr
import spaces
import tempfile
import json
import zipfile
import math
import re
import traceback
from pathlib import Path
from PIL import Image

# Import new modules
from chandra_align.preprocessing import (
    ensure_uint8, apply_clahe, detect_shadows, apply_wallis_filter, preprocess_multimodal_pair,
    preprocess_iirs_raster, resize_to_common_ground_sample, keypoint_starvation_guard
)
from chandra_align.features import (
    select_distributed_matches, spatial_distribution_metrics,
    select_quadrant_keypoints, select_quadrant_balanced_matches, select_detector_keypoints,
)
from chandra_align.alignment import decompose_partial_affine
from chandra_align.refine import refine_subpixel_ncc
from chandra_align.metrics import (
    compute_deformation_field, grid_deformation_analysis, compute_ground_metrics,
    get_sensor_pixel_scale, metrics_bundle_with_ground, ResidualVector, GroundMetrics,
    compute_quadrant_metrics, format_quadrant_html, build_judge_metrics_summary,
    compute_spatial_uniformity_metrics, validate_registration_gate,
)
from chandra_align.metrics.reporting import build_confidence_assessment
from chandra_align.visualization import (
    create_combined_visualization, draw_error_vector_overlay,
    create_checkerboard_overlay, create_interactive_blend, create_rejection_banner,
    create_warped_preview, fig_to_file
)
from chandra_align.export import (
    export_gcp_csv, export_homography_json, export_alignment_geotiff,
    export_alignment_png, export_composite_visualization, export_full_package
)
from chandra_align.registration_transform import (
    MIN_REGISTRATION_INLIERS,
    estimate_partial_affine as _estimate_partial_affine,
    homogeneous_affine as _homogeneous_affine,
)


MAX_IMAGE_DIMENSION = 4096
IIRS_MAX_IMAGE_DIMENSION = 12000
IIRS_MIN_MATCH_LONG_SIDE = 320


def _inspect_raster_geometry(file_input):
    """Retrieve original (width, height, size_mb, name) of an image or raster input."""
    if file_input is None:
        return None
    if isinstance(file_input, np.ndarray):
        h, w = file_input.shape[:2]
        size_mb = file_input.nbytes / (1024 * 1024)
        return (w, h, size_mb, "array input")
    if isinstance(file_input, Image.Image):
        w, h = file_input.size
        size_mb = (w * h * (len(file_input.getbands()) or 1)) / (1024 * 1024)
        return (w, h, size_mb, "PIL image")

    path = None
    if isinstance(file_input, (str, os.PathLike)):
        path = os.fspath(file_input)
    elif isinstance(file_input, dict):
        path = file_input.get("path") or file_input.get("name")
    else:
        path = getattr(file_input, "path", None) or getattr(file_input, "name", None)

    if not path or not os.path.isfile(str(path)):
        return None

    path_obj = Path(path)
    size_mb = 0.0
    try:
        size_mb = path_obj.stat().st_size / (1024 * 1024)
    except OSError:
        pass

    # Try rasterio first for scientific GeoTIFF/PDS rasters
    try:
        import rasterio
        raster_path = path
        if path_obj.suffix.lower() == ".img":
            for label_path in (path_obj.with_suffix(".xml"), Path(os.fspath(path) + ".xml")):
                if label_path.is_file():
                    raster_path = os.fspath(label_path)
                    break
        with rasterio.open(raster_path) as src:
            return (int(src.width), int(src.height), size_mb, path_obj.name)
    except Exception:
        pass

    # Try PIL Image header
    try:
        with Image.open(path) as img:
            return (int(img.width), int(img.height), size_mb, path_obj.name)
    except Exception:
        pass

    # Try OpenCV
    try:
        decoded = cv2.imread(os.fspath(path), cv2.IMREAD_UNCHANGED)
        if decoded is not None and decoded.ndim >= 2:
            return (int(decoded.shape[1]), int(decoded.shape[0]), size_mb, path_obj.name)
    except Exception:
        pass

    return None


def get_large_image_notices(ref_file, sec_file, max_dimension: int = MAX_IMAGE_DIMENSION) -> list[str]:
    """Check if either input exceeds dimension / size thresholds and generate notices."""
    notices = []
    for label, file_input in (("Reference", ref_file), ("Secondary", sec_file)):
        info = _inspect_raster_geometry(file_input)
        if info is None:
            continue
        w, h, size_mb, name = info
        max_side = max(w, h)
        if max_side > max_dimension:
            scale = float(max_dimension) / max_side
            target_w = max(1, round(w * scale))
            target_h = max(1, round(h * scale))
            notices.append(
                f"Large image detected ({w}×{h}, {size_mb:.1f} MB) — "
                f"downsampled to {target_w}×{target_h} px for processing; expect extended runtime."
            )
        elif size_mb >= 15.0:
            notices.append(
                f"Large image file detected ({w}×{h}, {size_mb:.1f} MB) — expect extended runtime."
            )
    return notices



def _estimate_partial_affine_with_threshold(src_points, dst_points, threshold_px):
    """Estimate affine geometry with a threshold-aware and version-safe call."""
    try:
        return _estimate_partial_affine(src_points, dst_points, threshold_px)
    except TypeError:
        # Some Space workers can briefly retain the previous two-argument
        # helper during a rolling deployment. Keep the requested sensor
        # threshold by fitting directly through OpenCV on those workers.
        src = np.asarray(src_points, dtype=np.float32).reshape(-1, 2)
        dst = np.asarray(dst_points, dtype=np.float32).reshape(-1, 2)
        if len(src) != len(dst) or len(src) < 3:
            return None, np.zeros(len(src), dtype=bool)
        finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
        if int(finite.sum()) < 3:
            return None, np.zeros(len(src), dtype=bool)
        matrix, mask = cv2.estimateAffinePartial2D(
            src[finite], dst[finite], method=cv2.RANSAC,
            ransacReprojThreshold=float(threshold_px), maxIters=10000,
            confidence=0.999, refineIters=10,
        )
        if matrix is None or mask is None or not np.isfinite(matrix).all():
            return None, np.zeros(len(src), dtype=bool)
        inliers = np.zeros(len(src), dtype=bool)
        inliers[np.flatnonzero(finite)] = mask.reshape(-1).astype(bool)
        return matrix.astype(np.float64), inliers


def _read_grayscale_image(image_file, max_dimension: int = MAX_IMAGE_DIMENSION) -> np.ndarray:
    """Decode an image input, normalize its channels/depth, and cap its dimensions."""
    if isinstance(image_file, np.ndarray):
        image = np.asarray(image_file)
    elif isinstance(image_file, Image.Image):
        image = np.asarray(image_file)
    else:
        if isinstance(image_file, (str, os.PathLike)):
            path = os.fspath(image_file)
        elif isinstance(image_file, dict):
            path = image_file.get("path") or image_file.get("name")
        else:
            path = getattr(image_file, "path", None) or getattr(image_file, "name", image_file)
        if path is None:
            raise ValueError("Input image is missing")
        image = cv2.imread(os.fspath(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Unable to decode input image file")

    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("Input must be a non-empty grayscale or color image")
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[:, :, 0]
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("Input image must have 1, 3, or 4 channels")

    # Resize before float conversion so very large uint16 images do not balloon
    # to several hundred megabytes during percentile normalization.
    height, width = image.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)

    if np.issubdtype(image.dtype, np.floating) and not np.isfinite(image).all():
        image = np.nan_to_num(image)
    if image.dtype != np.uint8:
        finite = image.astype(np.float32, copy=False)
        # Direct file inputs use a 1st-to-99th percentile display stretch.
        lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0, 0)
        if hi <= lo:
            image = np.zeros(image.shape, dtype=np.uint8)
        else:
            image = np.clip((finite - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image)


def load_lunar_raster(file_input, max_dimension: int = MAX_IMAGE_DIMENSION, band_index=None):
    """Load planetary imagery from arrays or raster files into normalized uint8.

    Rasterio is used for scientific raster formats so large PDS/ENVI/GeoTIFF
    inputs are read at a bounded resolution. Standard image formats retain an
    OpenCV fallback. ENVI datasets need their matching .hdr metadata file.
    """
    if file_input is None:
        return None

    if isinstance(file_input, np.ndarray):
        raw = np.asarray(file_input)
    elif isinstance(file_input, Image.Image):
        raw = np.asarray(file_input)
    else:
        if isinstance(file_input, (str, os.PathLike)):
            path = os.fspath(file_input)
        elif isinstance(file_input, dict):
            path = file_input.get("path") or file_input.get("name")
        else:
            path = getattr(file_input, "path", None) or getattr(file_input, "name", None)
        if not path:
            return None
        path = os.path.abspath(os.fspath(path))
        if not os.path.isfile(path):
            return None
        raw = None

        # Preserve the original byte values for ordinary pictures. Re-stretching
        # an already calibrated uint8 image changes SIFT/RIFT gradients and can
        # reduce repeatable correspondences on the synthetic ground-truth pair.
        if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg"}:
            raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # PDS4 products keep raster metadata in a detached XML product label.
        # Prefer the matching sibling label when present; retain direct opens
        # for attached-label PDS3 files and ENVI/ISIS products.
        raster_path = path
        if Path(path).suffix.lower() == ".img":
            for label_path in (Path(path).with_suffix(".xml"), Path(path + ".xml")):
                if label_path.is_file():
                    raster_path = os.fspath(label_path)
                    break

        # Rasterio understands PDS, ENVI, and scientific GeoTIFF products.
        # Read only one science band (or RGB) and bound the read before scaling.
        try:
            if raw is None:
                import rasterio
                from rasterio.enums import Resampling

                with rasterio.open(raster_path) as src:
                    if src.count < 1 or src.width < 1 or src.height < 1:
                        return None
                    scale = min(1.0, float(max_dimension) / max(src.height, src.width))
                    out_height = max(1, round(src.height * scale))
                    out_width = max(1, round(src.width * scale))
                    out_shape = (out_height, out_width)
                    if band_index is not None:
                        selected_band = (
                            int(band_index) if src.count >= int(band_index)
                            else max(1, src.count // 2)
                        )
                        bands = [selected_band]
                    elif src.count in (3, 4):
                        bands = [1, 2, 3]
                    elif src.count == 1:
                        bands = [1]
                    else:
                        # IIRS is commonly supplied as a 256-band ENVI cube; use
                        # the specified science band, or the center for other cubes.
                        bands = [125 if src.count >= 125 else max(1, src.count // 2)]

                    if len(bands) == 1:
                        band = src.read(
                            bands[0], out_shape=out_shape,
                            resampling=Resampling.average, masked=True,
                        )
                        if src.dtypes[bands[0] - 1] == "uint8":
                            raw = np.asarray(band.filled(0))
                        else:
                            raw = np.asarray(band.astype(np.float32).filled(np.nan))
                    else:
                        rgb = src.read(
                            bands, out_shape=(len(bands), *out_shape),
                            resampling=Resampling.average, masked=True,
                        )
                        if all(src.dtypes[index - 1] == "uint8" for index in bands):
                            rgb = np.asarray(rgb.filled(0))
                        else:
                            rgb = np.asarray(rgb.astype(np.float32).filled(np.nan))
                        raw = cv2.cvtColor(np.moveaxis(rgb, 0, -1), cv2.COLOR_RGB2GRAY)
                    if src.nodata is not None and raw.dtype != np.uint8:
                        raw[raw == src.nodata] = np.nan
        except Exception:
            raw = None

        # OpenCV handles PNG/JPEG and many ordinary TIFFs without optional GIS
        # dependencies. Do not interpret binary PDS/ENVI files as ordinary images.
        if raw is None:
            suffix = Path(path).suffix.lower()
            if suffix in {".img", ".qub"}:
                return None
            decoded = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if decoded is None:
                return None
            raw = decoded

    if raw.size == 0 or raw.ndim not in (2, 3):
        return None
    if raw.ndim == 3:
        # Handle channel-first layout (C, H, W) from Rasterio before the
        # channel-last (H, W, C) branches.  Without this, a (1, H, W) array
        # evaluates shape[2] == width, falls through to `return None`, and
        # silently drops all keypoint extraction on the HF Linux container.
        if raw.shape[0] in (1, 3, 4) and raw.shape[2] not in (1, 3, 4):
            # Almost certainly channel-first: transpose to (H, W, C)
            raw = np.moveaxis(raw, 0, -1)
        if raw.ndim == 3 and raw.shape[2] == 1:
            raw = raw[:, :, 0]
        elif raw.ndim == 3 and raw.shape[2] == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        elif raw.ndim == 3 and raw.shape[2] == 4:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        elif raw.ndim == 3:
            return None

    # Keep uint8 inputs byte-for-byte stable after grayscale conversion.
    # Higher-bit-depth / floating rasters use a 2nd-to-98th percentile stretch.
    preserve_uint8 = raw.dtype == np.uint8

    height, width = raw.shape
    scale = min(1.0, float(max_dimension) / max(height, width))
    if scale < 1.0:
        raw = cv2.resize(
            raw, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    if preserve_uint8:
        return np.ascontiguousarray(raw)
    raw = raw.astype(np.float32, copy=False)
    valid = np.isfinite(raw)
    if not valid.any():
        return np.zeros(raw.shape, dtype=np.uint8)
    low, high = np.nanpercentile(raw[valid], (2, 98))
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if high <= low:
        high = low + 1e-5
    normalized = np.clip((raw - low) / (high - low), 0.0, 1.0)
    normalized[~valid] = 0.0
    return np.ascontiguousarray((normalized * 255.0).astype(np.uint8))


def match_pair_hf(
    img1: np.ndarray, img2: np.ndarray, ransac_threshold_px: float = 3.0
) -> tuple[np.ndarray, np.ndarray, str, dict]:
    """Run LightGlue/ALIKED, then SIFT; RIFT2 runs only with explicit opt-in."""
    def as_numpy(image):
        # Gradio/ZeroGPU callers can hand a tensor-backed image to the matcher.
        # Detach and move it to host memory before NumPy/OpenCV conversion.
        if hasattr(image, "detach"):
            image = image.detach()
        if hasattr(image, "cpu"):
            image = image.cpu()
        if hasattr(image, "numpy"):
            image = image.numpy()
        return np.asarray(image)

    # SIFT and most deep matcher frontends require finite uint8 image arrays.
    def as_uint8(image):
        image = as_numpy(image)
        image = ensure_uint8(image)
        if image.ndim != 2 or image.size == 0:
            raise ValueError("Feature matching requires non-empty grayscale images")
        if image.dtype == np.uint8:
            return np.ascontiguousarray(image)
        image = np.nan_to_num(image.astype(np.float32, copy=False))
        lo, hi = np.percentile(image, [1, 99])
        return np.zeros(image.shape, np.uint8) if hi <= lo else np.clip((image - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

    sift_input1, sift_input2 = as_numpy(img1), as_numpy(img2)
    img1, img2 = as_uint8(sift_input1), as_uint8(sift_input2)
    diagnostics = {
        "primary_engine": "LightGlue/ALIKED",
        "fallback_triggered": False,
        "fallback_reason": None,
        "registration_engine": "LightGlue/ALIKED",
        "rift2_opt_in": os.environ.get("CHANDRA_ENABLE_RIFT2", "").strip().lower() in {"1", "true", "yes", "on"},
    }
    try:
        from chandra_align.matching.deep_matchers import DeepMatcherChain, LightGlueALIKEDMatcher
        result = DeepMatcherChain(matchers=[LightGlueALIKEDMatcher(max_keypoints=2048)]).match(
            img1, img2, min_matches=4
        )
        if len(result.pts_src) >= 4:
            diagnostics["registration_engine"] = f"LightGlue/ALIKED ({result.matcher_name})"
            return result.pts_src, result.pts_ref, diagnostics["registration_engine"], diagnostics
        lightglue_error = result.error_msg or "LightGlue/ALIKED returned fewer than four correspondences"
    except Exception as exc:
        lightglue_error = f"{type(exc).__name__}: {exc}"

    diagnostics.update(
        fallback_triggered=True,
        fallback_reason=f"LightGlue/ALIKED insufficient correspondences or exception: {lightglue_error}",
        fallback_error=lightglue_error,
    )

    # RIFT2 remains callable only by explicit operator opt-in. Its license is
    # unresolved and its measured correspondence yield is non-functional.
    rift2_error = "not attempted (opt-in disabled)"
    if diagnostics["rift2_opt_in"]:
        try:
            from chandra_align.matcher import RIFT2Matcher
            src_pts, dst_pts = RIFT2Matcher(npt=2048).match(img1, img2)
            if len(src_pts) >= 4:
                diagnostics["registration_engine"] = "RIFT2 (Phase Congruency; opt-in)"
                return src_pts, dst_pts, diagnostics["registration_engine"], diagnostics
            rift2_error = f"RIFT2 returned {len(src_pts)} correspondences (< 4 required)"
        except Exception as exc:
            rift2_error = f"{type(exc).__name__}: {exc}"
        diagnostics["rift2_error"] = rift2_error
        diagnostics["fallback_reason"] += "; opt-in RIFT2 did not produce a candidate"

    # Validated CPU fallback. RIFT2 is not attempted unless an operator opted in.
    # Keep live Spaces useful when optional deep-matcher weights/dependencies
    # are unavailable. SIFT is bounded, and candidate fits are RANSAC-verified.
    diagnostics["deep_fallback_error"] = diagnostics.get("fallback_error")
    try:
        sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.005, edgeThreshold=15)

        def as_sift_uint8(image):
            image = ensure_uint8(image)
            if image.dtype == np.uint8:
                return np.ascontiguousarray(image)
            image = np.nan_to_num(image.astype(np.float32, copy=False))
            lo, hi = np.percentile(image, [1, 99])
            if hi <= lo:
                return np.zeros(image.shape, np.uint8)
            if 0.0 <= lo and hi <= 1.0:
                return np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
            if 0.0 <= lo and hi <= 255.0:
                return np.clip(image, 0.0, 255.0).astype(np.uint8)
            return np.clip((image - lo) * (255.0 / (hi - lo)), 0.0, 255.0).astype(np.uint8)

        def edge_emphasis(image):
            value = image.astype(np.float32)
            gx = cv2.Sobel(value, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(value, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(gx, gy)
            high = float(np.percentile(magnitude, 98.0)) if magnitude.size else 0.0
            if high > 0.0:
                magnitude = np.clip(magnitude * (255.0 / high), 0.0, 255.0)
            else:
                magnitude.fill(0.0)
            edges = np.rint(magnitude).astype(np.uint8)
            return cv2.addWeighted(image, 0.6, edges, 0.4, 0.0)

        variants = (
            (as_sift_uint8(sift_input1), as_sift_uint8(sift_input2), "reflectance"),
            (edge_emphasis(as_sift_uint8(sift_input1)), as_sift_uint8(sift_input2), "reference edge emphasis"),
            (as_sift_uint8(sift_input1), edge_emphasis(as_sift_uint8(sift_input2)), "secondary edge emphasis"),
            (edge_emphasis(as_sift_uint8(sift_input1)), edge_emphasis(as_sift_uint8(sift_input2)), "paired edge emphasis"),
        )
        best = None
        for reference_image, secondary_image, variant_name in variants:
            key_ref, desc_ref = sift.detectAndCompute(reference_image, None)
            key_sec, desc_sec = sift.detectAndCompute(secondary_image, None)
            if desc_ref is None or desc_sec is None or len(desc_ref) < 2 or len(desc_sec) < 2:
                continue
            key_ref, idx_ref = select_detector_keypoints(key_ref, reference_image.shape, 64)
            key_sec, idx_sec = select_detector_keypoints(key_sec, secondary_image.shape, 64)
            desc_ref, desc_sec = desc_ref[idx_ref], desc_sec[idx_sec]
            if len(desc_ref) < 2 or len(desc_sec) < 2:
                continue
            # Brute-force L2 keeps this small, bounded fallback reproducible;
            # randomized FLANN trees caused intermittent OHRC example failures.
            candidates = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_sec, desc_ref, k=2)
            for ratio in (0.75, 0.80, 0.85):
                good = [first for pair in candidates if len(pair) == 2
                        for first, second in [pair]
                        if first.distance < ratio * second.distance]
                if len(good) < MIN_REGISTRATION_INLIERS:
                    continue
                points_sec = np.float32([key_sec[item.queryIdx].pt for item in good])
                points_ref = np.float32([key_ref[item.trainIdx].pt for item in good])
                # RIFT2 and unrelated requests may advance OpenCV's global RNG;
                # seed each fit so marginal cross-sensor examples are repeatable.
                cv2.setRNGSeed(0)
                matrix, inlier_mask = cv2.estimateAffinePartial2D(
                    points_sec, points_ref, method=cv2.RANSAC,
                    ransacReprojThreshold=ransac_threshold_px, maxIters=10000,
                    confidence=0.999, refineIters=10,
                )
                inliers = (inlier_mask.reshape(-1).astype(bool)
                           if matrix is not None and inlier_mask is not None
                           else np.zeros(len(good), dtype=bool))
                count = int(inliers.sum())
                if count >= MIN_REGISTRATION_INLIERS and (best is None or count > best[0]):
                    best = (count, points_ref, points_sec, variant_name, ratio)

        if best is not None:
            _, candidate_ref, candidate_sec, variant_name, ratio = best
            diagnostics["registration_engine"] = "SIFT + Brute-Force (RANSAC fallback)"
            diagnostics["sift_variant"] = variant_name
            diagnostics["sift_ratio"] = ratio
            diagnostics["lightglue_error"] = lightglue_error
            return (candidate_ref.astype(np.float32), candidate_sec.astype(np.float32),
                    diagnostics["registration_engine"], diagnostics)
        diagnostics["sift_error"] = "No SIFT candidate reached eight partial-affine RANSAC inliers"
    except Exception as exc:
        diagnostics["sift_error"] = str(exc)
    diagnostics["lightglue_error"] = lightglue_error
    return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), "Failed", diagnostics


def _match_sift_ransac(ref_image, sec_image, threshold_px):
    """Run the CPU SIFT fallback after a weak LightGlue geometric candidate."""
    ref_image, sec_image = ensure_uint8(ref_image), ensure_uint8(sec_image)
    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.005, edgeThreshold=15)
    key_ref, desc_ref = sift.detectAndCompute(ref_image, None)
    key_sec, desc_sec = sift.detectAndCompute(sec_image, None)
    if desc_ref is None or desc_sec is None or len(desc_ref) < 2 or len(desc_sec) < 2:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), None
    key_ref, idx_ref = select_detector_keypoints(key_ref, ref_image.shape, 64)
    key_sec, idx_sec = select_detector_keypoints(key_sec, sec_image.shape, 64)
    desc_ref, desc_sec = desc_ref[idx_ref], desc_sec[idx_sec]
    candidates = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_sec, desc_ref, k=2)
    best = None
    for ratio in (0.75, 0.80, 0.85):
        good = [first for pair in candidates if len(pair) == 2
                for first, second in [pair] if first.distance < ratio * second.distance]
        if len(good) < MIN_REGISTRATION_INLIERS:
            continue
        points_sec = np.float32([key_sec[item.queryIdx].pt for item in good])
        points_ref = np.float32([key_ref[item.trainIdx].pt for item in good])
        matrix, mask = _estimate_partial_affine_with_threshold(points_sec, points_ref, threshold_px)
        count = int(mask.sum()) if mask is not None else 0
        if matrix is not None and count >= MIN_REGISTRATION_INLIERS and (best is None or count > best[0]):
            best = (count, points_ref, points_sec)
    if best is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), None
    return best[1], best[2], best[0]


def _compute_spatial_uniformity(inlier_points, img_shape):
    """8x8 grid-occupancy + nearest-neighbor spread telemetry (DIAGNOSTIC ONLY).

    Never affects gate decisions or verdicts. On any failure the payload
    reports valid=False with zeroed fields so telemetry marks the metrics
    invalid rather than fabricating values.
    """
    try:
        payload = compute_spatial_uniformity_metrics(inlier_points, img_shape)
    except Exception:
        payload = {
            "grid_shape": [8, 8],
            "grid_occupancy": 0.0,
            "occupied_cells": 0,
            "n_points": 0,
            "nn_mean_px": 0.0, "nn_median_px": 0.0,
            "nn_std_px": 0.0, "nn_p5_px": 0.0,
        }
    payload["valid"] = bool(payload.get("n_points", 0) >= 2)
    payload["diagnostic_only"] = True
    return payload


def _jsonable_deform_field_info(info):
    """Strip non-JSON-serializable values from deform-field stage telemetry."""
    safe = {}
    for key, value in (info or {}).items():
        if key in ("residuals_vec", "residuals_mag", "field", "M_hook",
                   "refit_field", "refit_M_hook"):
            # Raw stage artifacts for the dense-remap export; excluded from
            # JSON telemetry (large arrays). The export path reads them from
            # the raw deform_field_info dict, not from judge_metrics.
            continue
        if key == "rescue_info" and isinstance(value, dict):
            # Strip raw refit artifacts from the nested rescue info; keep
            # the scalar telemetry (gate code, RMSE, counts, etc.).
            value = {k: v for k, v in value.items()
                     if k not in ("refit_field", "refit_M_hook",
                                  "admitted_mask", "field", "M_hook")}
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            value = value.item()
        try:
            import json as _json
            _json.dumps(value)
        except (TypeError, ValueError):
            value = repr(value)
        safe[key] = value
    return safe


def _align_core(
    ref_img: np.ndarray,
    sec_img: np.ndarray,
    pixel_scale_m: float = 0.25,
    enable_clahe: bool = True,
    clahe_clip_limit: float = 3.0,
    enable_shadow_suppression: bool = False,
    enable_wallis: bool = False,
    wallis_target_mean: float = 128.0,
    wallis_target_std: float = 50.0,
    enforce_uniform_distribution: bool = True,
    sensor_pair_mode: str = "Optical <-> Optical",
    secondary_sensor_name: str | None = None,
    reference_sensor_name: str = "OHRC",
    max_image_dimension: int = MAX_IMAGE_DIMENSION,
    matching_mode: str = "single_scale",
) -> dict:
    """
    Core alignment logic - runs on GPU when called from process_alignment.
    Returns comprehensive results dictionary.
    """
    if secondary_sensor_name is None:
        secondary_sensor_name = reference_sensor_name
    # Store original images for visualization
    ref_original = _read_grayscale_image(ref_img, max_image_dimension)
    sec_original = _read_grayscale_image(sec_img, max_image_dimension)
    try:
        pixel_scale_m = float(pixel_scale_m)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Pixel scale must be a finite positive number of meters per pixel.") from exc
    if not math.isfinite(pixel_scale_m) or pixel_scale_m <= 0:
        raise ValueError("Pixel scale must be a finite positive number of meters per pixel.")
    if sensor_pair_mode not in ("Optical <-> Optical", "Optical <-> Infrared"):
        raise ValueError("Unsupported sensor pair mode.")
    if (sensor_pair_mode == "Optical <-> Infrared"
            and "IIRS" not in (str(reference_sensor_name).upper(), str(secondary_sensor_name).upper())):
        raise ValueError("Optical <-> Infrared mode requires IIRS as the reference or secondary sensor.")
    
    # Apply sensor-aware radiometric preprocessing.
    ref_processed = ref_img.copy()
    sec_processed = sec_img.copy()

    is_iirs_pair = sensor_pair_mode == "Optical <-> Infrared"
    ransac_threshold_px = 4.0 if is_iirs_pair else 3.0
    minimum_match_long_side = IIRS_MIN_MATCH_LONG_SIDE if is_iirs_pair else 128
    if is_iirs_pair:
        if str(reference_sensor_name).upper() == "IIRS":
            ref_processed = preprocess_iirs_raster(
                ref_processed, contrast_percentiles=(1.0, 99.0), clahe_clip_limit=4.5
            ).processed_2d_raster
        if str(secondary_sensor_name).upper() == "IIRS":
            sec_processed = preprocess_iirs_raster(
                sec_processed, contrast_percentiles=(1.0, 99.0), clahe_clip_limit=4.5
            ).processed_2d_raster
        if enable_clahe and str(reference_sensor_name).upper() != "IIRS":
            ref_processed = apply_clahe(ref_processed, clip_limit=4.5, normalize_range=(0.0, 255.0))

    if enable_clahe and sensor_pair_mode != "Optical <-> Infrared":
        ref_processed = apply_clahe(ref_processed, clip_limit=clahe_clip_limit, normalize_range=(0.0, 255.0))
        sec_processed = apply_clahe(sec_processed, clip_limit=clahe_clip_limit, normalize_range=(0.0, 255.0))

    if sensor_pair_mode == "Optical <-> Infrared":
        ref_processed, sec_processed = preprocess_multimodal_pair(ref_processed, sec_processed)
    
    shadow_mask = None
    shadow_mask_sec = None
    if enable_shadow_suppression:
        shadow_mask = detect_shadows(ref_processed, method="otsu")
        shadow_mask_sec = detect_shadows(sec_processed, method="otsu")
        # We'll apply shadow suppression during keypoint filtering

    ref_processed, sec_processed, shadow_mask, shadow_mask_sec, starvation_stats = keypoint_starvation_guard(
        ref_processed, sec_processed, shadow_mask, shadow_mask_sec,
        preprocessing_triggered=(enable_shadow_suppression or not enable_clahe),
        min_candidates=30,
        clahe_clip_limit=max(float(clahe_clip_limit), 3.0),
    )
    
    if enable_wallis:
        ref_processed = apply_wallis_filter(ref_processed, target_mean=wallis_target_mean, target_std=wallis_target_std)
        sec_processed = apply_wallis_filter(sec_processed, target_mean=wallis_target_mean, target_std=wallis_target_std)
    
    # Match at a common ground sample distance. Coordinates are restored to each
    # source image before geometry is estimated.
    try:
        ref_gsd_m = pixel_scale_m
        sec_gsd_m = get_sensor_pixel_scale(secondary_sensor_name)
        target_gsd_m = max(ref_gsd_m, sec_gsd_m)
        match_ref, ref_match_scale = resize_to_common_ground_sample(
            ref_processed, ref_gsd_m, target_gsd_m, minimum_match_long_side
        )
        match_sec, sec_match_scale = resize_to_common_ground_sample(
            sec_processed, sec_gsd_m, target_gsd_m, minimum_match_long_side
        )
    except ValueError:
        match_ref, match_sec = ref_processed, sec_processed
        ref_match_scale = sec_match_scale = 1.0
    # Opt-in MatchAnything front-end (no training, pretrained weights).
    # When CHANDRA_MATCHANYTHING=1 and the weights/deps are available, use
    # MatchAnything ELoFTR instead of the default matcher. Fail-closed: any
    # problem falls back to the default matcher. With the flag unset, this
    # block is skipped and the pipeline is bit-identical.
    # See chandra_align/matchanything_frontend.py.
    _ma_pts = None
    if os.environ.get("CHANDRA_MATCHANYTHING", "").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            from chandra_align.matchanything_frontend import (
                matchanything_available, match_pair_matchanything,
            )
            if matchanything_available():
                _ma0, _ma1 = match_pair_matchanything(match_ref, match_sec)
                if _ma0 is not None and _ma1 is not None and len(_ma0) >= 3:
                    _ma_pts = (_ma0, _ma1)
        except Exception:
            _ma_pts = None
    if _ma_pts is not None:
        pts_ref, pts_sec = _ma_pts
        engine_name = "matchanything_eloftr"
        execution_diagnostics = {"matcher": "matchanything_eloftr",
                                 "opt_in": True,
                                 "fallback_triggered": False,
                                 "fallback_reason": None,
                                 "primary_engine": "MatchAnything/ELoFTR",
                                 "registration_engine": "MatchAnything/ELoFTR"}
    # Opt-in AnyMatch front-end (no training, fine-tuned weights).
    # When CHANDRA_ANYMATCH=1 and the weights/deps are available, use
    # AnyMatch fine-tuned LoFTR instead of the default matcher. Fail-closed:
    # any problem falls back to the default matcher. With the flag unset,
    # this block is skipped and the pipeline is bit-identical.
    # See chandra_align/anymatch_frontend.py.
    _am_pts = None
    if _ma_pts is None and os.environ.get("CHANDRA_ANYMATCH", "").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            from chandra_align.anymatch_frontend import (
                anymatch_available, match_pair_anymatch,
            )
            if anymatch_available():
                _am0, _am1 = match_pair_anymatch(match_ref, match_sec)
                if _am0 is not None and _am1 is not None and len(_am0) >= 3:
                    _am_pts = (_am0, _am1)
        except Exception:
            _am_pts = None
    if _am_pts is not None:
        pts_ref, pts_sec = _am_pts
        engine_name = "anymatch_loftr"
        execution_diagnostics = {"matcher": "anymatch_loftr",
                                 "opt_in": True,
                                 "fallback_triggered": False,
                                 "fallback_reason": None,
                                 "primary_engine": "AnyMatch/LoFTR",
                                 "registration_engine": "AnyMatch/LoFTR"}
    elif matching_mode == "coarse_to_fine":
        from chandra_align.matching.coarse_to_fine import coarse_to_fine_match
        pts_ref, pts_sec, engine_name, execution_diagnostics = coarse_to_fine_match(
            match_ref, match_sec, match_pair_hf,
            ransac_threshold_px=ransac_threshold_px,
        )
    elif matching_mode == "single_scale":
        pts_ref, pts_sec, engine_name, execution_diagnostics = match_pair_hf(
            match_ref, match_sec, ransac_threshold_px
        )
    else:
        raise ValueError("matching_mode must be 'single_scale' or 'coarse_to_fine'")
    if ref_match_scale != 1.0:
        pts_ref = pts_ref / ref_match_scale
    if sec_match_scale != 1.0:
        pts_sec = pts_sec / sec_match_scale

    # --- OPT-IN deformation-field stage, pre-balance placement (CHANDRA_DEFORM_FIELD=1; default OFF) ---
    # Runs BEFORE quadrant balancing, on the UNBUCKETED match set: a primary
    # RANSAC on the raw matches yields the inlier set the field needs (the
    # pipeline's own RANSAC below sees bucketed points), the TPS residual
    # field corrects them, and the corrected correspondences flow into the
    # existing balance -> bucket -> RANSAC -> NCC-refit chain below, which is
    # not modified. With the flag unset, pts_ref/pts_sec are untouched and the
    # pipeline is bit-identical to the unmodified path. Fail-closed: any
    # exception, a failed RANSAC, or a declined stage leaves the original
    # matches in place and the pipeline continues exactly as before.
    # See chandra_align/deform_field.py and
    # docs/deform-field-pipeline-stage-2026-10-08.md.
    deform_field_info = {"applied": False, "reason": "flag_off"}
    if os.environ.get("CHANDRA_DEFORM_FIELD", "").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            from chandra_align.deform_field import apply_deform_field_stage
            deform_field_info = {"applied": False, "reason": "not_attempted"}
            if len(pts_ref) >= 3 and len(pts_ref) == len(pts_sec):
                # Deterministic hook RANSAC (cf. A16 Part B, cv2.setRNGSeed(7)).
                cv2.setRNGSeed(7)
                M_hook, inliers_hook = _estimate_partial_affine_with_threshold(
                    pts_sec, pts_ref, ransac_threshold_px
                )
                n_hook_in = int(inliers_hook.sum()) if inliers_hook is not None else 0
                if M_hook is not None and n_hook_in >= MIN_REGISTRATION_INLIERS:
                    stage_info = apply_deform_field_stage(
                        pts_sec[inliers_hook], pts_ref[inliers_hook], M_hook
                    )
                    if stage_info.get("applied"):
                        # Save the RAW matches for the L1 rescue (second
                        # opinion) before the stage correction overwrites them.
                        pts_sec_raw = np.asarray(pts_sec)
                        pts_ref_raw = np.asarray(pts_ref)
                        r_corr = np.asarray(stage_info["residuals_vec"], dtype=np.float64)
                        p_ref_in = np.asarray(pts_ref[inliers_hook], dtype=np.float64)
                        if r_corr.shape == p_ref_in.shape and np.all(np.isfinite(r_corr)):
                            # Field-corrected reference points:
                            # ref' = M@sec + F(sec) = ref - r_corr.
                            corr_ref = (p_ref_in - r_corr).astype(np.float32)
                            corr_sec = np.asarray(pts_sec[inliers_hook], dtype=np.float32)
                            pts_ref, pts_sec = corr_ref, corr_sec
                            deform_field_info = dict(stage_info)
                            deform_field_info["hook_placement"] = "pre_balance"
                            deform_field_info["n_hook_ransac_inliers"] = n_hook_in
                            # L1: field-guided rescue (second opinion). When the
                            # hook inliers are quadrant-starved (<3), score the
                            # full raw match set under the fitted field, admit
                            # field-consistent (<1 px) correspondences, and
                            # refit the stage on the admitted set. The refit is
                            # gated directly (worker's validated method); the
                            # rescue verdict is adopted after the normal gate
                            # below, on strict SUCCESS upgrade only. The normal
                            # pool above is never perturbed.
                            # See chandra_align/deform_field.py and
                            # docs/ohrc-blocker-levers-2026-10-08.md.
                            _rescue = {"rescued": False, "reason": "not_attempted"}
                            try:
                                from chandra_align.deform_field import (
                                    maybe_rescue_and_refit as _rr,
                                    _eval_residual_field as _ef,
                                    _apply_affine as _aa,
                                    _check_export_field as _cf,
                                )
                                _refit, _rescue = _rr(
                                    stage_info, pts_sec_raw, pts_ref_raw,
                                    inliers_hook, M_hook, match_ref.shape[:2])
                                if _rescue.get("rescued"):
                                    # Gate the refit admitted set directly
                                    # (worker's validated method): quadrants
                                    # and entropy on the admitted raw points,
                                    # RMSE from the refit stage's held-out.
                                    _adm = np.asarray(
                                        _rescue["admitted_mask"],
                                        dtype=bool).reshape(-1)
                                    _fld = _cf(_refit["field"])
                                    _Ma = np.asarray(
                                        _refit["M_hook"], dtype=np.float64)
                                    _ps = np.asarray(
                                        pts_sec_raw, dtype=np.float64)[_adm]
                                    _pr = np.asarray(
                                        pts_ref_raw, dtype=np.float64)[_adm]
                                    _d = _ef(_fld, _ps)
                                    _pred = _aa(_Ma, _ps) + _d
                                    _rvec = _pr - _pred
                                    _qm, _ent = compute_quadrant_metrics(
                                        _pr, _rvec, match_ref.shape[:2])
                                    _qc = {
                                        "Q1": int(_qm["Q1"]["inlier_count"]),
                                        "Q2": int(_qm["Q2"]["inlier_count"]),
                                        "Q3": int(_qm["Q3"]["inlier_count"]),
                                        "Q4": int(_qm["Q4"]["inlier_count"]),
                                    }
                                    _rmsg, _rcode = validate_registration_gate(
                                        float(_refit["heldout_rmse_px"]),
                                        int(_adm.sum()),
                                        MIN_REGISTRATION_INLIERS,
                                        float(_ent), _qc)
                                    _rescue["rescue_gate_code"] = _rcode
                                    _rescue["rescue_gate_rmse_px"] = float(
                                        _refit["heldout_rmse_px"])
                                    _rescue["rescue_n"] = int(_adm.sum())
                                    _rescue["rescue_entropy"] = float(_ent)
                                    _rescue["rescue_quad_counts"] = [
                                        _qc["Q1"], _qc["Q2"],
                                        _qc["Q3"], _qc["Q4"]]
                                    # Store the refit field for the export
                                    # path: when the rescue verdict is
                                    # adopted, the dense remap must use the
                                    # rescue field, not the normal field.
                                    # Excluded from JSON telemetry (raw
                                    # arrays); the export path reads them
                                    # from deform_field_info directly.
                                    _rescue["refit_field"] = _refit.get(
                                        "field")
                                    _rescue["refit_M_hook"] = _refit.get(
                                        "M_hook")
                            except Exception as _rexc:
                                _rescue = {"rescued": False,
                                           "reason": "exception:%s" % type(
                                               _rexc).__name__}
                            deform_field_info["rescue_info"] = {
                                k: v for k, v in _rescue.items()
                                if k not in ("field", "M_hook", "admitted_mask")
                            }
                            # Fallback-void token (popped before telemetry): if the
                            # _align_core fallback below replaces the points, the
                            # stage result no longer describes the gated data.
                            deform_field_info["_fb0"] = bool(
                                execution_diagnostics.get("fallback_triggered")
                            )
                        else:
                            deform_field_info = {"applied": False, "reason": "nonfinite_corrected"}
                    else:
                        deform_field_info = {"applied": False,
                                             "reason": stage_info.get("reason", "stage_declined")}
                else:
                    deform_field_info = {"applied": False, "reason": "hook_ransac_failed"}
        except Exception as exc:  # fail closed: keep the original matches
            deform_field_info = {"applied": False,
                                 "reason": f"exception:{type(exc).__name__}"}

    if len(pts_ref):
        pts_ref, pts_sec, _ = select_quadrant_balanced_matches(
            pts_ref, pts_sec, ref_original.shape[:2], quota_per_quadrant=50
        )
    
    # Apply shadow suppression to matched pairs jointly. Filtering each side
    # independently can remove different matches and break correspondence order.
    if shadow_mask is not None:
        def outside_shadow(points, mask):
            xy = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            x, y = xy[:, 0].astype(int), xy[:, 1].astype(int)
            h, w = mask.shape[:2]
            in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            visible = np.ones(len(xy), dtype=bool)
            visible[in_bounds] = mask[y[in_bounds], x[in_bounds]] < 0.5
            return visible

        keep = outside_shadow(pts_ref, shadow_mask) & outside_shadow(pts_sec, shadow_mask_sec)
        pts_ref, pts_sec = pts_ref[keep], pts_sec[keep]

    if enforce_uniform_distribution and len(pts_ref):
        pts_ref, pts_sec, _ = select_distributed_matches(
            pts_ref, pts_sec, ref_original.shape[:2], grid_shape=(8, 8), max_per_bucket=8
        )
    
    if len(pts_ref) < 3:
        match_detail = (execution_diagnostics.get("sift_error")
                        or execution_diagnostics.get("fallback_error")
                        or execution_diagnostics.get("primary_error"))
        detail = f" Matcher detail: {match_detail}" if match_detail else ""
        raise ValueError(f"Insufficient keypoint correspondences detected.{detail}")

    geometry_exception = False
    try:
        affine_matrix, inliers = _estimate_partial_affine_with_threshold(
            pts_sec, pts_ref, ransac_threshold_px
        )
    except cv2.error:
        affine_matrix, inliers = None, np.zeros(len(pts_ref), dtype=bool)
        geometry_exception = True

    inlier_cnt = int(inliers.sum())

    # Escalate only when the primary geometric solution is weak. The fallback
    # is the distinct CPU SIFT/RANSAC path, rather than retrying LightGlue.
    if not execution_diagnostics["fallback_triggered"]:
        inlier_ratio = inlier_cnt / max(len(pts_ref), 1)
        primary_uniformity = spatial_distribution_metrics(
            pts_ref[inliers], ref_original.shape[:2], (8, 8)
        )["uniformity"] if inlier_cnt else 0.0
        fallback_reason = None
        if geometry_exception:
            fallback_reason = "Primary Geometry Estimation Exception"
        elif inlier_ratio < 0.15:
            fallback_reason = f"Low Inlier Ratio ({inlier_ratio:.1%} < 15.0%)"
        elif primary_uniformity < 0.125:
            fallback_reason = f"High Spatial Entropy Deficit (uniformity {primary_uniformity:.3f} < 0.125)"

        if fallback_reason:
            execution_diagnostics.update(
                fallback_triggered=True, fallback_reason=fallback_reason
            )
            try:
                fallback_ref, fallback_sec, fallback_inlier_count = _match_sift_ransac(
                    match_ref, match_sec, ransac_threshold_px
                )
                fallback_ref = fallback_ref / ref_match_scale if ref_match_scale != 1.0 else fallback_ref
                fallback_sec = fallback_sec / sec_match_scale if sec_match_scale != 1.0 else fallback_sec
                if len(fallback_ref):
                    fallback_ref, fallback_sec, _ = select_quadrant_balanced_matches(
                        fallback_ref, fallback_sec, ref_original.shape[:2],
                        quota_per_quadrant=50,
                    )
                if shadow_mask is not None and len(fallback_ref):
                    keep = outside_shadow(fallback_ref, shadow_mask) & outside_shadow(fallback_sec, shadow_mask_sec)
                    fallback_ref, fallback_sec = fallback_ref[keep], fallback_sec[keep]
                if enforce_uniform_distribution and len(fallback_ref):
                    fallback_ref, fallback_sec, _ = select_distributed_matches(
                        fallback_ref, fallback_sec, ref_original.shape[:2],
                        grid_shape=(8, 8), max_per_bucket=8
                    )
                if len(fallback_ref) >= 3 and fallback_inlier_count is not None:
                    fallback_affine, fallback_inliers = _estimate_partial_affine_with_threshold(
                        fallback_sec, fallback_ref, ransac_threshold_px
                    )
                    if fallback_affine is not None:
                        if int(fallback_inliers.sum()) >= MIN_REGISTRATION_INLIERS:
                            pts_ref, pts_sec = fallback_ref, fallback_sec
                            affine_matrix, inliers = fallback_affine, fallback_inliers
                            inlier_cnt = int(inliers.sum())
                            engine_name = "SIFT + Brute-Force (RANSAC fallback)"
                            execution_diagnostics["registration_engine"] = engine_name
                if execution_diagnostics["registration_engine"] != engine_name:
                    execution_diagnostics["fallback_error"] = "SIFT/RANSAC did not yield a valid partial affine with the required inlier count"
            except Exception as exc:
                execution_diagnostics["fallback_error"] = str(exc)

    # Hook maintenance (deform-field stage): if the _align_core fallback above
    # ran after the stage applied, its re-matched points supersede the
    # field-corrected set, so the stage result no longer describes the gated
    # data -- void it fail-closed. (For SIFT-path pairs the fallback flag is
    # already set by the matcher and this never triggers.)
    if deform_field_info.get("applied"):
        _fb0 = deform_field_info.pop("_fb0", None)
        if _fb0 is False and execution_diagnostics.get("fallback_triggered"):
            deform_field_info = {"applied": False, "reason": "fallback_superseded"}

    if affine_matrix is None:
        raise ValueError(
            f"Insufficient Inliers: partial affine registration requires at least "
            f"{MIN_REGISTRATION_INLIERS}; found {inlier_cnt}."
        )

    # Refine only geometrically verified matches, then re-estimate the model.
    refinement_stats = {"refined_pairs": 0, "status": "insufficient_verified_matches"}
    inlier_ref, inlier_sec = pts_ref[inliers], pts_sec[inliers]
    try:
        if inlier_cnt < MIN_REGISTRATION_INLIERS:
            raise ValueError("Below minimum inlier safety gate; subpixel refinement skipped")
        refined_ref, refined_sec, refinement_stats = refine_subpixel_ncc(
            ref_processed, sec_processed, inlier_ref, inlier_sec,
            ncc_window=11, search_range_px=1
        )
        refinement_stats["status"] = "subpixel_pairs_refined" if len(refined_ref) else "no_subpixel_pairs"
        if len(refined_ref) >= 3:
            refined_affine, refined_inliers = _estimate_partial_affine_with_threshold(
                refined_sec, refined_ref, ransac_threshold_px
            )
            if refined_affine is not None:
                if int(refined_inliers.sum()) >= MIN_REGISTRATION_INLIERS:
                    affine_matrix = refined_affine
                    inlier_ref = refined_ref[refined_inliers]
                    inlier_sec = refined_sec[refined_inliers]
                    refinement_stats["status"] = "subpixel_model_reestimated"
    except (cv2.error, ValueError, FloatingPointError) as exc:
        refinement_stats = {"refined_pairs": 0, "status": f"skipped: {exc}"}

    inlier_cnt = len(inlier_ref)
    pts_sec_h = (
        cv2.transform(inlier_sec.reshape(-1, 1, 2), affine_matrix).reshape(-1, 2)
        if inlier_cnt else np.empty((0, 2), dtype=np.float32)
    )
    residuals_vec = inlier_ref - pts_sec_h
    residuals_mag = np.linalg.norm(residuals_vec, axis=1)
    quadrant_metrics, quadrant_spatial_entropy = compute_quadrant_metrics(
        inlier_ref, residuals_vec, ref_original.shape[:2]
    )
    quadrant_html = format_quadrant_html(quadrant_metrics, quadrant_spatial_entropy)
    # Diagnostic-only spatial uniformity (never feeds the acceptance gate).
    spatial_uniformity_metrics = _compute_spatial_uniformity(
        inlier_ref, ref_original.shape[:2]
    )

    rmse_in_sample_px = float(np.sqrt(np.mean(residuals_mag ** 2))) if len(residuals_mag) > 0 else 0.0
    mae_in_sample_px = float(np.mean(residuals_mag)) if len(residuals_mag) > 0 else 0.0
    from chandra_align.trust import evaluate as evaluate_heldout_rmse
    heldout_error = None
    heldout_error_label = None
    try:
        # evaluate() creates disjoint fit/check subsets and calls rmse_heldout(),
        # whose empty-check-set guard prevents in-sample substitution.
        _, heldout_error = evaluate_heldout_rmse(
            affine_matrix, inlier_sec, inlier_ref
        )
        if heldout_error.get("held_out") is not True or not isinstance(heldout_error.get("rmse_px"), (int, float)):
            heldout_error_label = "held-out unavailable"
        elif not math.isfinite(float(heldout_error["rmse_px"])):
            heldout_error_label = "held-out unavailable"
    except (ValueError, cv2.error, FloatingPointError, TypeError) as exc:
        heldout_error_label = f"held-out unavailable ({exc})"
    if heldout_error_label:
        rmse_heldout_px = None
        mae_heldout_px = None
        rmse_gate_px = rmse_in_sample_px
        rmse_gate_basis = f"in-sample fallback; {heldout_error_label}"
    else:
        rmse_heldout_px = float(heldout_error["rmse_px"])
        mae_heldout_px = float(heldout_error["mae_px"])
        rmse_gate_px = rmse_heldout_px
        rmse_gate_basis = "held-out"
    # Deform-field stage, part 2 of the hook: when the field applied, the gate
    # must score the field's generalization, not the affine in-sample fit. The
    # stage's internal stride 80/20 held-out (affine+TPS fit on the fit subset,
    # checked on the disjoint check subset -- the pipeline's own split
    # methodology) is the honest gate basis. The affine held-out above is
    # preserved in telemetry for comparison.
    if deform_field_info.get("applied") and deform_field_info.get("heldout_rmse_px") is not None:
        deform_field_info["affine_heldout_rmse_px"] = rmse_heldout_px
        deform_field_info["affine_gate_basis"] = rmse_gate_basis
        rmse_heldout_px = float(deform_field_info["heldout_rmse_px"])
        rmse_gate_px = rmse_heldout_px
        rmse_gate_basis = "held-out (deform-field stage internal stride 80/20 split)"
    rmse_px = rmse_gate_px
    mae_px = mae_in_sample_px
    affine_telemetry = decompose_partial_affine(affine_matrix)
    status_message, status_code = validate_registration_gate(
        rmse_px, inlier_cnt, MIN_REGISTRATION_INLIERS,
        quadrant_spatial_entropy, quadrant_metrics,
    )
    # L1 rescue second opinion: if the rescue refit reached SUCCESS_SUBPIXEL
    # under the frozen gate while the normal path did not, adopt the rescue
    # verdict. Strict upgrade only -- never a downgrade or lateral move.
    _ri = deform_field_info.get("rescue_info") or {}
    if (_ri.get("rescue_gate_code") == "SUCCESS_SUBPIXEL"
            and status_code != "SUCCESS_SUBPIXEL"):
        status_code = "SUCCESS_SUBPIXEL"
        status_message = "REGISTRATION ACCEPTED (L1 field-guided rescue)"
        rmse_px = float(_ri["rescue_gate_rmse_px"])
        rmse_gate_px = rmse_px
        rmse_gate_basis = ("held-out (L1 rescue refit stage internal "
                           "stride 80/20 split)")
        inlier_cnt = int(_ri["rescue_n"])
        quadrant_spatial_entropy = float(_ri["rescue_entropy"])
        _rqc = _ri["rescue_quad_counts"]
        quadrant_metrics = {
            "Q1": {"inlier_count": int(_rqc[0]), "name": "Top-Left",
                   "rmse_px": 0.0},
            "Q2": {"inlier_count": int(_rqc[1]), "name": "Top-Right",
                   "rmse_px": 0.0},
            "Q3": {"inlier_count": int(_rqc[2]), "name": "Bottom-Left",
                   "rmse_px": 0.0},
            "Q4": {"inlier_count": int(_rqc[3]), "name": "Bottom-Right",
                   "rmse_px": 0.0},
        }
        deform_field_info["verdict_source"] = "l1_rescue_second_opinion"
        # Export consistency: the dense remap (below) reads
        # deform_field_info["field"] / ["M_hook"]. When the rescue verdict
        # is adopted, the exported raster must use the rescue refit's
        # field, not the normal stage's field. Fail-closed: if the refit
        # field is missing or invalid, keep the normal field.
        try:
            _rf = _ri.get("refit_field")
            _rm = _ri.get("refit_M_hook")
            if _rf is not None and _rm is not None:
                from chandra_align.deform_field import (
                    _check_export_field as _cef)
                _rf_checked = _cef(_rf)
                _rm_arr = np.asarray(_rm, dtype=np.float64)
                if (_rf_checked is not None and _rm_arr.shape == (2, 3)
                        and np.all(np.isfinite(_rm_arr))):
                    deform_field_info["field"] = _rf_checked
                    deform_field_info["M_hook"] = _rm_arr
                    deform_field_info["export_field_source"] = (
                        "l1_rescue_refit")
        except Exception:
            # Fail-closed: keep the normal stage field for export.
            pass
    judge_metrics = build_judge_metrics_summary(
        rmse_px,
        inlier_cnt,
        len(pts_ref),
        quadrant_spatial_entropy,
        quadrant_metrics,
        min_inliers=MIN_REGISTRATION_INLIERS,
    )
    judge_metrics.update({
        "rmse_in_sample_px": rmse_in_sample_px,
        "rmse_heldout_px": rmse_heldout_px if rmse_heldout_px is not None else "UNMEASURED",
        "mae_in_sample_px": mae_in_sample_px,
        "mae_heldout_px": mae_heldout_px if mae_heldout_px is not None else "UNMEASURED",
        "rmse_gate_px": rmse_gate_px,
        "rmse_gate_basis": rmse_gate_basis,
        "heldout_check_points": int(heldout_error.get("n_check_points", 0)),
        "spatial_uniformity_metrics": spatial_uniformity_metrics,
        # Deform-field stage telemetry (diagnostic only; never feeds the gate).
        "deform_field_stage": _jsonable_deform_field_info(deform_field_info),
    })
    confidence_assessment = build_confidence_assessment(
        status_code=status_code, status_message=status_message, rmse_px=rmse_gate_px,
        inlier_count=inlier_cnt, correspondence_count=len(pts_ref),
        entropy=quadrant_spatial_entropy, quadrant_counts=quadrant_metrics,
        min_inliers=MIN_REGISTRATION_INLIERS,
        execution_diagnostics=execution_diagnostics,
    )
    judge_metrics.update(confidence_assessment)

    engine_lower = engine_name.lower()
    if "sift" in engine_lower:
        engine_used = "Fallback (SIFT)"
    elif "lightglue" in engine_lower or "aliked" in engine_lower:
        engine_used = "Primary (LightGlue/ALIKED)"
    else:
        engine_used = "Opt-in (RIFT2 phase congruency)"
    execution_device = "GPU (ZeroGPU)" if _zerogpu_runtime_enabled() else "CPU"

    if status_code == "DEGENERATE_FAILURE":
        banner_bgr = create_rejection_banner(ref_original.shape[:2])
        banner_rgb = cv2.cvtColor(banner_bgr, cv2.COLOR_BGR2RGB)
        return {
            "rejected": True,
            "banner_rgb": banner_rgb,
            "banner_bgr": banner_bgr,
            "ref_original": ref_original,
            "judge_metrics": judge_metrics,
            "status_message": status_message,
            "status_code": status_code,
            "inlier_cnt": inlier_cnt,
            "total_matches": len(pts_ref),
            "rmse_px": None,
            "rmse_in_sample_px": rmse_in_sample_px,
            "rmse_heldout_px": rmse_heldout_px,
            "mae_in_sample_px": mae_in_sample_px,
            "mae_heldout_px": mae_heldout_px,
            "heldout_error": heldout_error,
            "rmse_gate_px": rmse_gate_px,
            "rmse_gate_basis": rmse_gate_basis,
            "quadrant_spatial_entropy": quadrant_spatial_entropy,
            "spatial_uniformity_metrics": spatial_uniformity_metrics,
            "execution_diagnostics": execution_diagnostics,
            "confidence_assessment": confidence_assessment,
            **confidence_assessment,
            "engine_used": engine_used,
            "execution_device": execution_device,
            "affine_telemetry": None,
        }

    H = _homogeneous_affine(affine_matrix)
    warped_sec = cv2.warpAffine(sec_original, affine_matrix, (ref_original.shape[1], ref_original.shape[0]))
    # --- OPT-IN dense field remap (CHANDRA_DEFORM_FIELD=1; default OFF) ---
    # The stage scores the forward map F(p) = M_hook @ p + d(p), but the warp
    # above is affine-only. When the stage applied (and was not voided by the
    # fallback check), the exported raster carries the field via cv2.remap
    # with fixed-point-inverted sampling maps
    # (chandra_align.deform_field.build_field_remap_maps). Any failure --
    # bad field, singular matrix, non-finite maps -- falls back to the
    # affine warp above, bit-identical to the pre-change behavior. The kind
    # of warp actually exported is recorded in telemetry.
    warp_export_kind = "affine"
    if deform_field_info.get("applied"):
        try:
            from chandra_align.deform_field import build_field_remap_maps
            _fmaps = build_field_remap_maps(
                sec_original.shape[:2], ref_original.shape[:2],
                deform_field_info.get("M_hook"), deform_field_info.get("field"))
            _w = cv2.remap(sec_original, _fmaps[0], _fmaps[1],
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if (_w is not None and _w.shape == warped_sec.shape
                    and np.all(np.isfinite(_w))):
                warped_sec = _w
                warp_export_kind = "field_remap"
            else:
                warp_export_kind = "affine (field remap produced degenerate output)"
        except Exception as exc:  # fail closed: keep the affine warp
            warp_export_kind = "affine (field remap failed: %s)" % type(exc).__name__
    judge_metrics["warp_export_kind"] = warp_export_kind
    diff_map = cv2.absdiff(ref_original, warped_sec)
    error_vector_overlay_bgr = draw_error_vector_overlay(
        ref_original, warped_sec, inlier_sec, inlier_ref, affine_matrix, scale=10.0,
        inlier_count=inlier_cnt, min_inliers=MIN_REGISTRATION_INLIERS, status=status_code,
    )

    warped_preview_rgb = create_warped_preview(
        warped_sec, image_shape=ref_original.shape[:2], inlier_count=inlier_cnt,
        min_inliers=MIN_REGISTRATION_INLIERS, status=status_code,
    )
    checkerboard_rgb = create_checkerboard_overlay(
        ref_original, warped_sec, tile_size=64, inlier_count=inlier_cnt,
        min_inliers=MIN_REGISTRATION_INLIERS, status=status_code,
    )
    error_vector_overlay_rgb = cv2.cvtColor(error_vector_overlay_bgr, cv2.COLOR_BGR2RGB)

    # Compute deformation field
    deformation_vectors = compute_deformation_field(
        inlier_ref, inlier_sec, H, pixel_scale_m
    )
    
    # Grid deformation analysis
    grid_analysis = grid_deformation_analysis(deformation_vectors, (8, 8), ref_original.shape[:2], pixel_scale_m)
    
    # Ground metrics
    ground_metrics = compute_ground_metrics(residuals_mag, pixel_scale_m)
    
    # Spatial entropy (uniformity)
    from chandra_align.evaluators.quadtree import evaluate_quadtree_uniformity
    uniformity = 0.0
    distribution = spatial_distribution_metrics(inlier_ref, ref_original.shape[:2], (8, 8))
    if inlier_cnt > 0:
        quadtree_result = evaluate_quadtree_uniformity(inlier_ref, ref_original.shape[:2], depth=4)
        uniformity = distribution["uniformity"]

    return {
        "rejected": False,
        "warped_preview_rgb": warped_preview_rgb,
        "checkerboard_rgb": checkerboard_rgb,
        "error_vector_overlay_rgb": error_vector_overlay_rgb,
        "blend_rgb": create_interactive_blend(ref_original, warped_sec, alpha=0.5),
        "blend_inputs": (ref_original, warped_sec),
        "error_vector_overlay_bgr": error_vector_overlay_bgr,
        "quadrant_metrics": quadrant_metrics,
        "quadrant_spatial_entropy": quadrant_spatial_entropy,
        "spatial_uniformity_metrics": spatial_uniformity_metrics,
        "quadrant_metrics_html": quadrant_html,
        "judge_metrics": judge_metrics,
        "status_message": status_message,
        "status_code": status_code,
        "affine_telemetry": affine_telemetry,
        "transformation_telemetry": affine_telemetry,
        "engine_used": engine_used,
        "execution_device": execution_device,
        "starvation_guard": starvation_stats,
        "ref_original": ref_original,
        "sec_original": sec_original,
        "warped_sec": warped_sec,
        "diff_map": diff_map,
        "engine_name": engine_name,
        "execution_diagnostics": execution_diagnostics,
        "confidence_assessment": confidence_assessment,
        **confidence_assessment,
        "inlier_cnt": inlier_cnt,
        "total_matches": len(pts_ref),
        "rmse_px": rmse_px,
        "rmse_in_sample_px": rmse_in_sample_px,
        "rmse_heldout_px": rmse_heldout_px,
        "mae_in_sample_px": mae_in_sample_px,
        "mae_heldout_px": mae_heldout_px,
        "heldout_error": heldout_error,
        "rmse_gate_basis": rmse_gate_basis,
        "mae_px": mae_px,
        "H": H,
        "affine_matrix": affine_matrix,
        "pts_ref_inliers": inlier_ref,
        "pts_sec_inliers": inlier_sec,
        "deformation_vectors": deformation_vectors,
        "grid_analysis": grid_analysis,
        "ground_metrics": ground_metrics,
        "uniformity": uniformity,
        "spatial_entropy": distribution["spatial_entropy"],
        "occupied_buckets": distribution["occupied_buckets"],
        "quadtree_uniformity": quadtree_result.uniformity_score if inlier_cnt > 0 else 0.0,
        "refinement_stats": refinement_stats,
        "sensor_pair_mode": sensor_pair_mode,
        "shadow_mask": shadow_mask,
        "pixel_scale_m": pixel_scale_m,
        "ransac_threshold_px": ransac_threshold_px,
    }


def _failed_judge_metrics_summary(inlier_count: int = 0, total_correspondences: int = 0):
    empty_quadrants = {
        key: {"rmse_px": 0.0}
        for key in ("Q1", "Q2", "Q3", "Q4")
    }
    summary = build_judge_metrics_summary(
        0.0, inlier_count, total_correspondences, 0.0, empty_quadrants
    )
    # Uniformity telemetry is always present, marked invalid — never fabricated.
    summary["spatial_uniformity_metrics"] = _compute_spatial_uniformity(
        np.empty((0, 2)), (1, 1)
    )
    assessment = build_confidence_assessment(
        status_code=summary.get("status_code", "DEGENERATE_FAILURE"),
        status_message=summary.get("status_message", "Registration did not produce a fit."),
        rmse_px=None, inlier_count=inlier_count,
        correspondence_count=total_correspondences, entropy=0.0,
        quadrant_counts=empty_quadrants, min_inliers=MIN_REGISTRATION_INLIERS,
        execution_diagnostics={"primary_engine": "Unavailable", "registration_engine": "Unavailable"},
    )
    summary.update(assessment)
    return summary


def format_telemetry_report(
    status, inliers, total_pts, inlier_ratio, spatial_entropy,
    quad_counts, rmse=None, affine_telemetry=None, engine="N/A", device="N/A",
    status_message=None, active_quadrants=None, notices=None,
    rmse_in_sample=None, rmse_heldout=None, heldout_count=0, rmse_gate_basis="held-out",
    mae_in_sample=None, mae_heldout=None, confidence_assessment=None,
):
    """Render three-tier checklist; hide transform values for rejected fits."""
    code = str(status).strip().upper()
    tier1 = code in ("SUCCESS", "SUCCESS_SUBPIXEL", "REGISTRATION ACCEPTED")
    tier2 = code in ("COARSE_ADVISORY", "COARSE ALIGNMENT (REGIONAL FIT ADVISORY)")
    accepted = tier1 or tier2
    counts = list(quad_counts or [0, 0, 0, 0])[:4]
    counts.extend([0] * (4 - len(counts)))
    active = sum(int(value) > 0 for value in counts) if active_quadrants is None else int(active_quadrants)
    rmse_value = float(rmse) if rmse is not None and math.isfinite(float(rmse)) else float("inf")
    entropy_value = float(spatial_entropy) if math.isfinite(float(spatial_entropy)) else 0.0
    rmse_pass = (rmse_value <= 0.5) and accepted
    entropy_pass = entropy_value >= 0.75
    quadrant_pass = active >= 3
    # Priority-ordered header: use the caller's message when available;
    # otherwise derive an explicit rejection reason instead of the old
    # catch-all "Degenerate Single-Quadrant Cluster" default.
    if status_message:
        status_text = status_message
    elif tier1:
        status_text = "ACCEPTED (Sub-Pixel Precision)"
    elif tier2:
        status_text = "COARSE ALIGNMENT (Regional Fit Advisory)"
    elif inliers == 0:
        status_text = "REJECTED (Zero Inlier Matches Detected)"
    elif not rmse_pass:
        status_text = f"REJECTED (High Residual RMSE: {rmse_value:.4f} px > 0.50 px)"
    elif not quadrant_pass:
        status_text = f"REJECTED (Degenerate Spatial Cluster: {active}/4 Active Quadrants)"
    elif not entropy_pass:
        status_text = f"REJECTED (Low Spatial Entropy: {entropy_value:.4f} < 0.75)"
    else:
        status_text = "REJECTED (Validation Criteria Not Satisfied)"
    check = lambda passed: "✓" if passed else "✗"
    lines = [
        "===================================================",
        "PHOTOGRAMMETRIC TELEMETRY REPORT",
        "===================================================",
    ]
    if notices:
        for notice in notices:
            lines.append(f"[NOTICE] {notice}")
        lines.append("")
    lines.extend([
        f"Registration Status   : {status_text}",
        f"Hardware Runtime Mode : {device}",
        "",
        "VALIDATION GATE CHECKLIST:",
        f"In-sample RMSE (fit residuals): {rmse_in_sample:.4f} px" if rmse_in_sample is not None else "In-sample RMSE (fit residuals): N/A",
        f"Held-out RMSE ({int(heldout_count)} check points): {rmse_heldout:.4f} px" if rmse_heldout is not None else "Held-out RMSE: UNAVAILABLE",
        f"In-sample MAE (fit residuals): {mae_in_sample:.4f} px" if mae_in_sample is not None else "In-sample MAE (fit residuals): N/A",
        f"Held-out MAE ({int(heldout_count)} check points): {mae_heldout:.4f} px" if mae_heldout is not None else "Held-out MAE: UNAVAILABLE",
        f"Gate RMSE: {rmse_value:.4f} px ({rmse_gate_basis})" if rmse is not None and math.isfinite(rmse_value) else f"Gate RMSE: N/A ({rmse_gate_basis})",
        f"[{check(rmse_pass)}] Sub-Pixel Precision  : RMSE <= 0.50 px (Measured: {rmse_value:.4f} px)" if (accepted and rmse is not None and math.isfinite(rmse_value)) else "[✗] Sub-Pixel Precision  : RMSE <= 0.50 px (Measured: N/A)",
        f"[{check(entropy_pass)}] Spatial Spread Score : Entropy >= 0.75 (Measured: {entropy_value:.4f} / 2.00)",
        f"[{check(quadrant_pass)}] Quadrant Distribution: Active Quads >= 3 (Measured: {active} / 4)",
        "",
        f"QUADRANT BREAKDOWN: Q1:{int(counts[0])} | Q2:{int(counts[1])} | Q3:{int(counts[2])} | Q4:{int(counts[3])}",
        f"Confidence score: {float(confidence_assessment.get('confidence_score', 0.0)):.2f}/100 (heuristic evidence index; not probability)" if isinstance(confidence_assessment, dict) else "Confidence score: unavailable",
        f"Decision reason: {confidence_assessment.get('decision_reason', 'Unavailable')}" if isinstance(confidence_assessment, dict) else "Decision reason: unavailable",
        f"Fallback path: {' -> '.join(confidence_assessment.get('fallback_path', {}).get('steps', [])) or 'Unavailable'}; triggered={confidence_assessment.get('fallback_path', {}).get('fallback_triggered', False)}; reason={confidence_assessment.get('fallback_path', {}).get('fallback_reason') or 'None'}" if isinstance(confidence_assessment, dict) else "Fallback path: unavailable",
        "---------------------------------------------------",
    ])
    if not accepted:
        lines.extend([
            "Recovered Transform : N/A — Alignment Rejected",
            "NOTICE: Transform telemetry is suppressed on rejected fits",
            "to prevent uncalibrated coordinate propagation into DEM products.",
        ])
    else:
        values = affine_telemetry or {}
        lines.extend([
            f"Recovered Translation  : ΔX = {float(values.get('delta_x_px', 0.0)):.4f} px, ΔY = {float(values.get('delta_y_px', 0.0)):.4f} px",
            f"Recovered Rotation     : θ = {float(values.get('rotation_deg', 0.0)):.4f}°",
            f"Recovered Uniform Scale: s = {float(values.get('scale_s', 0.0)):.8f}",
        ])
        if tier2:
            lines.extend([
                "",
                "NOTICE: Regional fit established (RMSE <= 2.5 px).",
                "Sub-pixel refinement recommended for DEM production.",
            ])
    lines.append("===================================================")
    return "\n".join(lines)


def _rejected_output_tuple(summary, status_message=None, image_shape=(768, 1024), total_pts=0, notices=None):
    """Build diagnostic image outputs and suppress downloads for rejected fits."""
    metrics = summary if isinstance(summary, dict) else _failed_judge_metrics_summary()
    global_metrics = metrics.get("global_metrics", {})
    inliers = int(global_metrics.get("inlier_count", 0))
    ratio = float(global_metrics.get("inlier_ratio_pct", 0.0))
    entropy = float(global_metrics.get("spatial_entropy_score", 0.0))
    if total_pts <= 0 and ratio > 0.0:
        total_pts = int(round(inliers * 100.0 / ratio))
    counts = metrics.get("quadrant_counts", [0, 0, 0, 0])
    message = status_message or metrics.get("status_message", "REJECTED: Alignment quality gate failed")
    banner_bgr = create_rejection_banner(image_shape)
    banner_rgb = cv2.cvtColor(banner_bgr, cv2.COLOR_BGR2RGB)
    banner_pil = Image.fromarray(banner_rgb)
    notice_text = ("\n".join(f"⚠️ **NOTICE:** {n}" for n in notices) + "\n\n") if notices else ""
    report = (
        f"{notice_text}❌ {message}\n\n"
        "REGISTRATION REJECTED: Insufficient Spatial Uniformity\n"
        "Sub-pixel alignment gate prevented degenerate warp execution."
    )
    telemetry = format_telemetry_report(
        metrics.get("status_code", "FAILED"), inliers, total_pts, ratio,
        entropy, counts, rmse=metrics.get("rmse_gate_px"),
        rmse_in_sample=metrics.get("rmse_in_sample_px"),
        rmse_heldout=(metrics.get("rmse_heldout_px")
                      if isinstance(metrics.get("rmse_heldout_px"), (int, float)) else None),
        heldout_count=metrics.get("heldout_check_points", 0),
        rmse_gate_basis=metrics.get("rmse_gate_basis", "held-out unavailable"),
        status_message=message, notices=notices, confidence_assessment=metrics,
    )
    return (
        banner_pil, banner_pil, banner_pil, banner_pil,
        report, metrics, telemetry, (banner_bgr, banner_bgr),
        None, None, None, None, None, None,
    )


def _minimal_rejection_output_tuple(status_message, summary=None):
    """Dependency-light emergency result that always matches the 14 outputs."""
    blank = np.zeros((512, 512, 3), dtype=np.uint8)
    preview = Image.fromarray(blank)
    metrics = summary if isinstance(summary, dict) else _failed_judge_metrics_summary()
    report = f"REGISTRATION REJECTED: {status_message}"
    telemetry = (
        "PHOTOGRAMMETRIC TELEMETRY REPORT\n"
        "Registration Status : REJECTED\n"
        "Recovered Transform : N/A — Alignment Rejected\n"
        f"Reason               : {status_message}\n"
        f"Confidence score     : {float(metrics.get('confidence_score', 0.0)):.2f}/100 (heuristic, not probability)\n"
        f"Decision reason      : {metrics.get('decision_reason', status_message)}\n"
        f"Fallback path        : {' -> '.join(metrics.get('fallback_path', {}).get('steps', [])) or 'Unavailable'}"
    )
    return (
        preview, preview, preview, preview,
        report, metrics, telemetry, (blank, blank),
        None, None, None, None, None, None,
    )


def _safe_rejection_output_tuple(summary, status_message, image_shape=(768, 1024), total_pts=0):
    """Prevent banner/telemetry formatting failures from escaping the callback."""
    try:
        outputs = tuple(_rejected_output_tuple(
            summary, status_message=status_message,
            image_shape=image_shape, total_pts=total_pts,
        ))
        if len(outputs) == 14:
            return outputs
        raise ValueError(f"Rejection callback produced {len(outputs)} outputs, expected 14")
    except Exception:
        print("Failed to render the normal rejection banner; using a minimal safe result.",
              file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return _minimal_rejection_output_tuple(status_message, summary)


def _unsupported_raster_message(ref_file, sec_file):
    """Return a practical upload fallback when a scientific raster won't decode."""
    paths = []
    for value in (ref_file, sec_file):
        if isinstance(value, dict):
            value = value.get("path") or value.get("name")
        else:
            value = getattr(value, "path", None) or getattr(value, "name", None) or value
        if value:
            paths.append(os.fspath(value))
    suffixes = {Path(path).suffix.lower() for path in paths}
    if ".qub" in suffixes:
        return (
            "Could not read the IIRS .qub cube with the available raster readers. "
            "Export the selected IIRS band (default band 125) to GeoTIFF or PNG and "
            "upload that image. For ENVI rasters, include the matching .hdr file."
        )
    if ".img" in suffixes:
        return (
            "Could not read this PDS .IMG product. Upload its matching label/metadata "
            "file, or export the image to GeoTIFF or PNG."
        )
    return "Invalid or unsupported image format. Upload a readable image or scientific raster."


def process_alignment(
    ref_file,
    sec_file,
    pixel_scale_m: float = 0.25,
    enable_clahe: bool = True,
    clahe_clip_limit: float = 3.0,
    enable_shadow_suppression: bool = False,
    enable_wallis: bool = False,
    wallis_target_mean: float = 128.0,
    wallis_target_std: float = 50.0,
    sensor_name: str = "OHRC",
    secondary_sensor_name: str | None = None,
    sensor_pair_mode: str = "Optical <-> Optical",
    enforce_uniform_distribution: bool = True,
    matching_mode: str = "single_scale",
):
    """
    Main alignment pipeline - runs on GPU when called from process_wrapper.
    Returns visual previews, telemetry, judge metrics, and scientific export paths.
    """
    # Pre-allocate safe values before image I/O or processing. The declared
    # Gradio return order remains the fixed 14-component contract below.
    default_blank_img = np.zeros((512, 512, 3), dtype=np.uint8)
    warped_sec = default_blank_img.copy()
    vector_overlay = default_blank_img.copy()
    checkerboard_blend = default_blank_img.copy()
    alpha_blend = default_blank_img.copy()
    telemetry_report = "Initializing pipeline..."
    judge_json = {}
    export_geotiff = None
    export_csv = None
    export_json = None
    export_png = None
    export_zip = None
    banner_img = default_blank_img.copy()
    active_badge = "Runtime Mode: Active"
    checklist_status = "INITIALIZING"
    dossier_path = None
    report_text = "Initializing pipeline..."
    reference_image = None

    if secondary_sensor_name is None:
        secondary_sensor_name = sensor_name

    if ref_file is None or sec_file is None:
        return _safe_rejection_output_tuple(
            _failed_judge_metrics_summary(),
            "Empty input: provide both Reference and Secondary surface frames.",
        )

    try:
        is_iirs_pair = sensor_pair_mode == "Optical <-> Infrared"
        max_image_dimension = IIRS_MAX_IMAGE_DIMENSION if is_iirs_pair else MAX_IMAGE_DIMENSION
        large_image_notices = get_large_image_notices(ref_file, sec_file, max_image_dimension)
        ref_band = 125 if sensor_name == "IIRS" else None
        sec_band = 125 if secondary_sensor_name == "IIRS" else None
        ref_img = load_lunar_raster(ref_file, max_image_dimension, ref_band)
        sec_img = load_lunar_raster(sec_file, max_image_dimension, sec_band)
        if ref_img is None or sec_img is None:
            return _safe_rejection_output_tuple(
                _failed_judge_metrics_summary(),
                _unsupported_raster_message(ref_file, sec_file),
            )
        result = _align_core(
            ref_img, sec_img,
            pixel_scale_m=pixel_scale_m,
            enable_clahe=enable_clahe,
            clahe_clip_limit=clahe_clip_limit,
            enable_shadow_suppression=enable_shadow_suppression,
            enable_wallis=enable_wallis,
            wallis_target_mean=wallis_target_mean,
            wallis_target_std=wallis_target_std,
            enforce_uniform_distribution=enforce_uniform_distribution,
            sensor_pair_mode=sensor_pair_mode,
            secondary_sensor_name=secondary_sensor_name,
            reference_sensor_name=sensor_name,
            max_image_dimension=max_image_dimension,
            matching_mode=matching_mode,
        )

        if result.get("rejected"):
            return _rejected_output_tuple(
                result["judge_metrics"],
                status_message=result["status_message"],
                image_shape=result["ref_original"].shape[:2],
                total_pts=result["total_matches"],
                notices=large_image_notices,
            )

        warped_preview_rgb = result["warped_preview_rgb"]
        checkerboard_rgb = result["checkerboard_rgb"]
        error_vector_overlay_rgb = result["error_vector_overlay_rgb"]
        judge_metrics = result["judge_metrics"]
        ref_original = result["ref_original"]
        warped_sec = result["warped_sec"]
        diff_map = result["diff_map"]
        engine_name = result["engine_name"]
        inlier_cnt = result["inlier_cnt"]
        total_matches = result["total_matches"]
        rmse_px = result["rmse_px"]
        rmse_in_sample_px = result["rmse_in_sample_px"]
        rmse_heldout_px = result["rmse_heldout_px"]
        heldout_error = result["heldout_error"] or {}
        rmse_gate_basis = result["rmse_gate_basis"]
        heldout_rmse_display = f"{rmse_heldout_px:.4f} px" if rmse_heldout_px is not None else "UNAVAILABLE"
        mae_px = result["mae_px"]
        mae_in_sample_px = result["mae_in_sample_px"]
        mae_heldout_px = result["mae_heldout_px"]
        H = result["H"]
        deformation_vectors = result["deformation_vectors"]
        grid_analysis = result["grid_analysis"]
        ground_metrics = result["ground_metrics"]
        uniformity = result["uniformity"]
        spatial_entropy = result["spatial_entropy"]
        quadrant_entropy = result["quadrant_spatial_entropy"]
        grid_uniformity_metrics = result.get("spatial_uniformity_metrics") or {}
        grid_cells = int((grid_uniformity_metrics.get("grid_shape") or [8, 8])[0]) * int(
            (grid_uniformity_metrics.get("grid_shape") or [8, 8])[1]
        )
        grid_occupancy = float(grid_uniformity_metrics.get("grid_occupancy", 0.0))
        occupied_cells = int(grid_uniformity_metrics.get("occupied_cells", 0))
        nn_mean = float(grid_uniformity_metrics.get("nn_mean_px", 0.0))
        nn_median = float(grid_uniformity_metrics.get("nn_median_px", 0.0))
        nn_p5 = float(grid_uniformity_metrics.get("nn_p5_px", 0.0))
        refinement_stats = result["refinement_stats"]
        pixel_scale_m = result["pixel_scale_m"]
        status_message = result["status_message"]
        status_code = result["status_code"]
        affine_telemetry = result["affine_telemetry"]
        active_quadrants = judge_metrics["active_quadrants_count"]
        quadrant_counts = judge_metrics["quadrant_counts"]

        # Build comprehensive report
        inlier_pct = inlier_cnt / max(total_matches, 1) * 100
        rmse_m = rmse_px * pixel_scale_m
        mae_m = mae_px * pixel_scale_m
        heldout_mae_display = f"{mae_heldout_px:.4f} px" if mae_heldout_px is not None else "UNAVAILABLE"
        
        notice_block = ("\n".join(f"⚠️ **NOTICE:** {n}" for n in large_image_notices) + "\n\n") if large_image_notices else ""
        report = (
            f"{notice_block}"
            f"{'✅' if status_code == 'SUCCESS_SUBPIXEL' else '🟠' if status_code == 'COARSE_ADVISORY' else '❌'} {status_message}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Matcher Engine: {engine_name}\n"
            f"Registration Transform: 4-DOF Partial Affine (2x3), RANSAC threshold {result['ransac_threshold_px']:.1f} px\n"
            f"Registration Matrix Engine: {result['execution_diagnostics'].get('registration_engine', engine_name)}\n"
            f"Primary Engine: {result['execution_diagnostics'].get('primary_engine', 'LightGlue/ALIKED')}\n"
            f"Fallback Triggered: {result['execution_diagnostics'].get('fallback_triggered', False)}\n"
            f"Fallback Reason: {result['execution_diagnostics'].get('fallback_reason') or 'None'}\n"
            f"Confidence Score: {result['confidence_score']:.2f}/100 (heuristic evidence index; not probability)\n"
            f"Decision Reason: {result['decision_reason']}\n"
            f"Fallback Path: {' -> '.join(result['fallback_path'].get('steps', []))}; triggered={result['fallback_path'].get('fallback_triggered', False)}\n"
            f"Sensor Pair Mode: {result['sensor_pair_mode']}\n"
            f"Verified Inliers: {inlier_cnt} / {total_matches} ({inlier_pct:.1f}%)\n"
            f"Spatial Uniformity U: {uniformity:.4f} (entropy {spatial_entropy:.4f} nats)\n"
            f"Grid Uniformity 8x8 (diagnostic, not gated): occupancy {grid_occupancy:.3f} "
            f"({occupied_cells}/{grid_cells} cells, {grid_cells - occupied_cells} empty), "
            f"NN mean {nn_mean:.2f} px, NN median {nn_median:.2f} px, NN p5 {nn_p5:.2f} px\n"
            f"NCC-Refined Candidates (pre-final RANSAC): {refinement_stats.get('refined_pairs', 0)} ({refinement_stats.get('status', 'not run')})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"PIXEL METRICS:\n"
            f"  In-sample RMSE (fit residuals): {rmse_in_sample_px:.4f} px\n"
            f"  Held-out RMSE ({heldout_error.get('n_check_points', 0)} check points): {heldout_rmse_display}\n"
            f"  In-sample MAE (fit residuals): {mae_in_sample_px:.4f} px\n"
            f"  Held-out MAE ({heldout_error.get('n_check_points', 0)} check points): {heldout_mae_display}\n"
            f"  Gate RMSE: {rmse_px:.4f} px ({rmse_gate_basis})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"GROUND METRICS (at {pixel_scale_m} m/px):\n"
            f"  RMSE: {rmse_m:.4f} m\n"
            f"  MAE:  {mae_m:.4f} m\n"
            f"  Std:  {ground_metrics.std_m:.4f} m\n"
            f"  N:    {ground_metrics.n_points} points\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"DEFORMATION FIELD GRID (8x8):\n"
            f"  Mean Mag: {grid_analysis['mean_magnitude_px']:.4f} px ({grid_analysis['mean_magnitude_m']:.4f} m)\n"
            f"  Max Mag:  {grid_analysis['max_magnitude_px']:.4f} px ({grid_analysis['max_magnitude_m']:.4f} m)\n"
        )
        telemetry_report = format_telemetry_report(
            status_code, inlier_cnt, total_matches, inlier_pct,
            quadrant_entropy, quadrant_counts, rmse=rmse_px,
            rmse_in_sample=rmse_in_sample_px, rmse_heldout=rmse_heldout_px,
            mae_in_sample=mae_in_sample_px, mae_heldout=mae_heldout_px,
            heldout_count=heldout_error.get("n_check_points", 0),
            rmse_gate_basis=rmse_gate_basis,
            affine_telemetry=affine_telemetry,
            engine=result["engine_used"], device=result["execution_device"],
            status_message=status_message,
            notices=large_image_notices,
            confidence_assessment=result["confidence_assessment"],
        )
        report += (
            "\n<!--QUADRANT_METRICS_START-->"
            f"{result['quadrant_metrics_html']}"
            "<!--QUADRANT_METRICS_END-->\n"
        )

        # Export scientific package
        output_dir = Path(tempfile.mkdtemp(prefix="chandra_align_export_"))
        base_name = "registration_result"
        
        export_paths = export_full_package(
            ref_original, warped_sec, diff_map,
            deformation_vectors, H,
            {
                "rmse_px": rmse_px,
                "rmse_in_sample_px": rmse_in_sample_px,
                "rmse_heldout_px": rmse_heldout_px if rmse_heldout_px is not None else "UNMEASURED",
                "mae_in_sample_px": mae_in_sample_px,
                "mae_heldout_px": mae_heldout_px if mae_heldout_px is not None else "UNMEASURED",
                "mae_in_sample_m": mae_in_sample_px * pixel_scale_m,
                "mae_heldout_m": mae_heldout_px * pixel_scale_m if mae_heldout_px is not None else "UNMEASURED",
                "mae_gate_basis": "in-sample and held-out diagnostics; MAE is not a gate criterion",
                "rmse_gate_basis": rmse_gate_basis,
                "mae_px": mae_px,
                "std_px": ground_metrics.std_px,
                "rmse_m": rmse_m,
                "mae_m": mae_m,
                "std_m": ground_metrics.std_m
            },
            str(output_dir),
            base_name,
            pixel_scale_m=pixel_scale_m,
            sensor_name=sensor_name,
            spatial_entropy=spatial_entropy,
            inlier_count=inlier_cnt,
            total_matches=total_matches,
            matcher_name=engine_name,
            trust_flag="TRUSTED" if inlier_cnt > 50 and rmse_px < 1.0 else "UNTRUSTED",
            image_shape=ref_original.shape[:2],
            grid_shape=(8, 8),
            spatial_uniformity=uniformity
        )
        # Preserve both the conventional homogeneous matrix and the exact
        # 2x3 four-parameter affine requested by downstream consumers.
        transform_json = Path(export_paths["transform_json"])
        with transform_json.open("r", encoding="utf-8") as stream:
            transform_payload = json.load(stream)
        transform_payload["transform_model"] = "partial_affine_4dof"
        transform_payload["partial_affine_matrix_2x3"] = np.asarray(
            result["affine_matrix"], dtype=np.float64
        ).tolist()
        transform_payload["homography_matrix"] = np.asarray(H, dtype=np.float64).tolist()
        transform_payload["judge_metrics_summary"] = judge_metrics
        transform_payload["safety_gate_passed"] = status_code == "SUCCESS_SUBPIXEL"
        transform_payload["coarse_advisory"] = status_code == "COARSE_ADVISORY"
        transform_payload["transformation_telemetry"] = affine_telemetry
        transform_payload["engine_used"] = result["engine_used"]
        transform_payload["execution_device"] = result["execution_device"]
        transform_payload["confidence_assessment"] = result["confidence_assessment"]
        with transform_json.open("w", encoding="utf-8") as stream:
            json.dump(transform_payload, stream, indent=2, allow_nan=False)
        
        # Also create the 4-panel scientific dossier
        viz_path = output_dir / f"{base_name}_dossier.png"
        fig = create_combined_visualization(
            ref_original, warped_sec, diff_map,
            deformation_vectors,
            grid_shape=(8, 8),
            pixel_scale_m=pixel_scale_m,
            rmse_px=rmse_px,
            rmse_in_sample_px=rmse_in_sample_px,
            rmse_heldout_px=rmse_heldout_px,
            mae_in_sample_px=mae_in_sample_px,
            mae_heldout_px=mae_heldout_px,
            heldout_count=heldout_error.get("n_check_points", 0),
            rmse_gate_basis=rmse_gate_basis,
            mae_px=mae_px,
            engine_name=engine_name,
            inlier_count=inlier_cnt,
            total_matches=total_matches,
            confidence_assessment=result["confidence_assessment"],
        )
        fig_to_file(fig, str(viz_path), dpi=300)
        export_paths["dossier_png"] = str(viz_path)

        # Create zip package
        zip_path = output_dir / f"{base_name}_package.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for key, path in export_paths.items():
                if path and Path(path).exists():
                    zf.write(path, Path(path).name)
        export_zip = str(zip_path)
        
        # Build RGB previews and return the fixed-key judge summary to Gradio.
        warped_pil = Image.fromarray(np.asarray(warped_preview_rgb, dtype=np.uint8))
        checkerboard_pil = Image.fromarray(np.asarray(checkerboard_rgb, dtype=np.uint8))
        vector_overlay_pil = Image.fromarray(np.asarray(error_vector_overlay_rgb, dtype=np.uint8))
        blend_pil = Image.fromarray(np.asarray(result["blend_rgb"], dtype=np.uint8))
        
        return (
            warped_pil,
            checkerboard_pil,
            vector_overlay_pil,
            blend_pil,
            report,
            judge_metrics,
            telemetry_report,
            result["blend_inputs"],
            export_paths.get("gcp_csv"),
            export_paths.get("transform_json"),
            export_paths.get("warped_geotiff"),
            export_paths.get("warped_png"),
            str(viz_path),
            export_zip,
        )

    except ValueError as e:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        failure = re.search(r"found\s+(\d+)", str(e), flags=re.IGNORECASE)
        failed_inliers = int(failure.group(1)) if failure else 0
        failed_summary = _failed_judge_metrics_summary(failed_inliers)
        image_shape = ref_img.shape[:2] if "ref_img" in locals() else (768, 1024)
        return _safe_rejection_output_tuple(
            failed_summary, status_message=failed_summary["status_message"],
            image_shape=image_shape,
        )
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        failed_summary = _failed_judge_metrics_summary()
        image_shape = ref_img.shape[:2] if "ref_img" in locals() else (768, 1024)
        return _safe_rejection_output_tuple(
            failed_summary, status_message=f"Registration Failed: {str(e)}",
            image_shape=image_shape,
        )


GPU_EXECUTION_STATUS = "⚡ Execution Mode: ZeroGPU (A10G Accelerated)"
CPU_FALLBACK_STATUS = "💻 Execution Mode: CPU (Fallback Active - Quota/Worker Limit Handled Gracefully)"

try:
    import torch
    TORCH_CUDA_AVAILABLE = bool(torch.cuda.is_available())
except Exception:
    TORCH_CUDA_AVAILABLE = False
RUNTIME_MODE_BADGE = (
    "Runtime Mode: 🟢 GPU Active (ZeroGPU)"
    if TORCH_CUDA_AVAILABLE else "Runtime Mode: 🟠 CPU Fallback"
)


def _append_execution_status(outputs, status: str):
    """Append execution mode and render the UI report without changing output arity."""
    values = list(outputs) if isinstance(outputs, (tuple, list)) else [None] * 14
    if len(values) != 14:
        values = (values + [None] * 14)[:14]
    report = values[4] if isinstance(values[4], str) else ""
    values[4] = f"{report.rstrip()}\n\n{status}".strip()
    telemetry = values[6] if isinstance(values[6], str) else ""
    device = "GPU (ZeroGPU)" if "ZeroGPU" in status else "CPU"
    telemetry = re.sub(
        r"(?m)^Hardware Runtime Mode\s*:\s*.*$",
        f"Hardware Runtime Mode : {'GPU (ZeroGPU Active)' if 'ZeroGPU' in status else 'CPU Fallback'}",
        telemetry,
    )
    values[6] = telemetry
    return tuple(values)


def _run_alignment_core(
    ref, sec, sensor, secondary_sensor, pair_mode, enforce_uniform,
    px_scale, clahe, clip, shadow, wallis, wallis_m, wallis_s
):
    return process_alignment(
        ref, sec,
        pixel_scale_m=px_scale,
        enable_clahe=clahe,
        clahe_clip_limit=clip,
        enable_shadow_suppression=shadow,
        enable_wallis=wallis,
        wallis_target_mean=wallis_m,
        wallis_target_std=wallis_s,
        sensor_name=sensor,
        secondary_sensor_name=secondary_sensor,
        sensor_pair_mode=pair_mode,
        enforce_uniform_distribution=enforce_uniform
    )


@spaces.GPU(duration=90)
def run_alignment_on_gpu(
    ref, sec, sensor, secondary_sensor, pair_mode, enforce_uniform,
    px_scale, clahe, clip, shadow, wallis, wallis_m, wallis_s
):
    """Run the alignment on ZeroGPU; the dispatcher catches allocation failures."""
    return _run_alignment_core(
        ref, sec, sensor, secondary_sensor, pair_mode, enforce_uniform,
        px_scale, clahe, clip, shadow, wallis, wallis_m, wallis_s
    )


def _zerogpu_runtime_enabled() -> bool:
    """The ZeroGPU decorator marks its wrapped function in ZeroGPU Spaces."""
    return hasattr(run_alignment_on_gpu, "zerogpu")


def process_wrapper(
    ref, sec, sensor, secondary_sensor, pair_mode, enforce_uniform,
    px_scale, clahe, clip, shadow, wallis, wallis_m, wallis_s
):
    """Catch ZeroGPU scheduling/quota errors and rerun on CPU without UI errors."""
    try:
        try:
            outputs = run_alignment_on_gpu(
                ref, sec, sensor, secondary_sensor, pair_mode, enforce_uniform,
                px_scale, clahe, clip, shadow, wallis, wallis_m, wallis_s
            )
            status = GPU_EXECUTION_STATUS if _zerogpu_runtime_enabled() else "💻 Execution Mode: CPU (Local Runtime)"
        except Exception:
            print("ZeroGPU allocation/execution failed; retrying alignment on CPU.", file=sys.stderr, flush=True)
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            try:
                outputs = _run_alignment_core(
                    ref, sec, sensor, secondary_sensor, pair_mode, enforce_uniform,
                    px_scale, clahe, clip, shadow, wallis, wallis_m, wallis_s
                )
            except Exception as cpu_error:
                print("CPU fallback failed.", file=sys.stderr, flush=True)
                print(traceback.format_exc(), file=sys.stderr, flush=True)
                outputs = _minimal_rejection_output_tuple(
                    f"Registration callback failed: {cpu_error}", summary={}
                )
            status = CPU_FALLBACK_STATUS

        normalized = tuple(_append_execution_status(outputs, status))
        if len(normalized) != 14:
            raise ValueError(f"Alignment callback returned {len(normalized)} outputs; expected 14")
        return normalized
    except Exception as callback_error:
        # Last-resort boundary: Gradio callbacks must not leak an exception or
        # return a shape/type mismatch that can surface as an HTTP 500.
        print("Unhandled alignment callback error; returning a safe 14-output rejection.",
              file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        try:
            fallback = _minimal_rejection_output_tuple(
                f"Registration callback failed: {callback_error}", summary={}
            )
            return tuple(fallback) if len(fallback) == 14 else (None,) * 14
        except Exception:
            print("Emergency callback fallback construction also failed.",
                  file=sys.stderr, flush=True)
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            return (None,) * 14


def update_interactive_blend(images, alpha):
    """Rebuild the blend preview from cached registration rasters and slider alpha."""
    if not isinstance(images, (tuple, list)) or len(images) != 2:
        return None
    try:
        return Image.fromarray(create_interactive_blend(images[0], images[1], alpha=alpha))
    except (TypeError, ValueError):
        return None


def _infer_input_sensor(file_input, fallback_sensor="OHRC"):
    """Resolve a sensor from a PDS4 label or known example filename.

    Uploaded images without sensor metadata use the workstation's selected
    sensor. In particular, do not silently invent TMC-2 for an unlabelled
    secondary image: that changes its assumed ground-sample distance and can
    downsample otherwise same-scale image pairs before feature matching.
    """
    if isinstance(file_input, (str, os.PathLike)):
        path = Path(file_input)
    elif isinstance(file_input, dict):
        value = file_input.get("path") or file_input.get("name")
        path = Path(value) if value else None
    else:
        value = getattr(file_input, "path", None) or getattr(file_input, "name", None)
        path = Path(value) if value else None

    if path is None:
        return str(fallback_sensor or "OHRC")

    # Native example assets encode both sensors in their pair-specific names.
    stem = path.stem.upper()
    if stem.startswith("NAC_REFERENCE_") or re.match(r"^M\d{9}(LE|RE)", stem):
        return "LROC_NAC"
    for token, sensor in (
        ("IIRS", "IIRS"), ("TMC2", "TMC-2"), ("TMC_2", "TMC-2"),
        ("TMC", "TMC-2"), ("OHRC", "OHRC"), ("LROC", "LROC_NAC"),
        ("NAC", "LROC_NAC"), ("WAC", "LROC_WAC"),
    ):
        if token in stem:
            return sensor

    # Raw PDS4 .IMG products carry the authoritative instrument identifier in
    # a detached XML label. Prefer it over filename heuristics.
    if path.suffix.lower() == ".img":
        for label_path in (path.with_suffix(".xml"), Path(os.fspath(path) + ".xml")):
            if not label_path.is_file():
                continue
            try:
                from chandra_align.ingestion.pds4 import parse_pds4_metadata
                sensor = parse_pds4_metadata(label_path).sensor_name
                if sensor and sensor != "UNKNOWN_SENSOR":
                    return "TMC-2" if sensor.upper() == "TMC2" else sensor
            except Exception:
                pass
            break

    return str(fallback_sensor or "OHRC")


def _process_alignment_from_ui(
    ref_input, sec_input, engine, chk_anms, chk_clahe, chk_shadow, chk_wallis,
    ref_raw_file=None, sec_raw_file=None,
):
    """Adapt the compact workstation controls to the full engine callback."""
    # gr.Image(filepath) is convenient for regular pictures, but Gradio still
    # validates/decode-processes its payload as an image. Raw PDS/ENVI products
    # therefore use the adjacent gr.File controls, which preserve the bytes.
    ref_input = ref_raw_file or ref_input
    sec_input = sec_raw_file or sec_input
    reference_sensor = _infer_input_sensor(ref_input, engine or "OHRC")
    secondary_sensor = _infer_input_sensor(sec_input, reference_sensor)
    pair_mode = (
        "Optical <-> Infrared"
        if "IIRS" in (reference_sensor, secondary_sensor)
        else "Optical <-> Optical"
    )

    try:
        max_dim = IIRS_MAX_IMAGE_DIMENSION if pair_mode == "Optical <-> Infrared" else MAX_IMAGE_DIMENSION
        for notice in get_large_image_notices(ref_input, sec_input, max_dim):
            gr.Info(notice)
    except Exception:
        pass

    return process_wrapper(
        ref_input,
        sec_input,
        reference_sensor,
        secondary_sensor,
        pair_mode,
        chk_anms,
        get_sensor_pixel_scale(reference_sensor),
        chk_clahe,
        3.0,
        chk_shadow,
        chk_wallis,
        128.0,
        50.0,
    )


# Build Gradio interface with advanced controls
def build_interface():
    with gr.Blocks(
        title="CHANDRA-ALIGN: Lunar Photogrammetric Workstation",
    ) as interface:
        gr.Markdown(
            "# 🌙 CHANDRA-ALIGN: Lunar Cross-Sensor Photogrammetric Workstation\n"
            "Sub-pixel registration for Chandrayaan-2 OHRC, TMC-2, IIRS, DF-SAR & LRO NAC imagery. "
            "Achieves scientific-grade photogrammetric accuracy with ground-resolution error translation."
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input panel
                gr.Markdown("### 📥 Input Frames")
                ref_input = gr.Image(
                    label="Reference Surface Frame (e.g., LRO NAC / PDS .IMG)",
                    type="filepath",
                    interactive=True,
                )
                sec_input = gr.Image(
                    label="Secondary Surface Frame (e.g., OHRC / TMC-2 / IIRS .qub)",
                    type="filepath",
                    interactive=True,
                )
                with gr.Accordion("Raw PDS / ENVI / Scientific Raster Uploads", open=False):
                    gr.Markdown(
                        "Use these file inputs for binary `.IMG` / `.qub` products or "
                        "rasters Gradio cannot preview. ENVI files require their matching `.hdr` file."
                    )
                    ref_raw_file = gr.File(
                        label="Reference raw raster file",
                        file_types=[".img", ".qub", ".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                        type="filepath",
                    )
                    sec_raw_file = gr.File(
                        label="Secondary raw raster file",
                        file_types=[".img", ".qub", ".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                        type="filepath",
                    )
                
                engine_dropdown = gr.Dropdown(
                    choices=["OHRC", "TMC-2", "IIRS", "DF-SAR", "LROC_NAC", "LROC_WAC", "KAGUYA_TC", "Custom"],
                    value="OHRC",
                    label="Reference Sensor (GSD)",
                    info="Used to interpret unlabelled inputs and convert pixel errors to ground units; embedded sensor metadata takes precedence."
                )

                gr.Markdown("### Preprocessing")
                chk_anms = gr.Checkbox(value=True, label="ANMS / Uniform Keypoint Distribution")
                chk_clahe = gr.Checkbox(value=True, label="CLAHE")
                chk_shadow = gr.Checkbox(
                    value=False,
                    label="Shadow Suppression",
                    info="Opt-in: Otsu masking can discard valid dark-surface features.",
                )
                chk_wallis = gr.Checkbox(value=False, label="Wallis Filter")
                
                btn_submit = gr.Button("Run Registration", variant="primary")
                gr.Markdown(
                    "Processing time varies with image size and hardware. Large or full-resolution pairs can take several minutes; "
                    "the app hides Gradio's unreliable automatic ETA."
                )
            
            with gr.Column(scale=2):
                # Output panel
                gr.Markdown("### 📊 Registration Results")
                
                with gr.Tabs():
                    with gr.TabItem("Warped Result"):
                        warped_result_image = gr.Image(
                            label="Warped Secondary on Reference Grid",
                            type="pil",
                            format="png"
                        )

                    with gr.TabItem("Checkerboard Blend"):
                        checkerboard_image = gr.Image(
                            label="Reference / Warped Secondary Checkerboard",
                            type="pil",
                            format="png"
                        )

                    with gr.TabItem("Vector Overlay"):
                        vector_overlay_image = gr.Image(
                            label="Color-Coded 10× Reprojection Residuals",
                            type="pil",
                            format="png"
                        )

                    with gr.TabItem("Interactive Alpha Blend"):
                        blend_alpha = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                            label="Warp Transparency / Swipe Blend",
                        )
                        blend_image = gr.Image(
                            label="Reference / Warped Secondary Alpha Blend",
                            type="pil", format="png",
                        )
                    
                    with gr.TabItem("Scientific Dossier (4-Panel)"):
                        dossier_image = gr.Image(
                            label="Photogrammetric Verification Dossier",
                            type="filepath"
                        )
                
                report_text = gr.Markdown(
                    label="Photogrammetric Summary Report",
                    sanitize_html=False,
                )
                metrics_json = gr.JSON(label="Judge Metrics Summary JSON")
                telemetry_text = gr.Textbox(
                    label="Photogrammetric Telemetry Report",
                    lines=15,
                    interactive=False,
                )
                blend_inputs_state = gr.State(value=None)
                
                # Download buttons
                gr.Markdown("### 💾 Scientific Export Package")
                with gr.Row():
                    csv_btn = gr.DownloadButton(
                        label="⬇️ Ground Control Points (CSV)",
                        visible=True
                    )
                    json_btn = gr.DownloadButton(
                        label="⬇️ Transform Matrix + Metrics (JSON)",
                        visible=True
                    )
                    geotiff_btn = gr.DownloadButton(
                        label="⬇️ Warped Secondary (GeoTIFF)",
                        visible=True
                    )
                    png_btn = gr.DownloadButton(
                        label="⬇️ Warped Secondary (16-bit PNG)",
                        visible=True
                    )
                    zip_btn = gr.DownloadButton(
                        label="⬇️ Complete Package (ZIP)",
                        visible=True
                    )
        
        # Event handlers
        btn_submit.click(
            fn=_process_alignment_from_ui,
            api_name="predict",
            inputs=[
                ref_input, sec_input, engine_dropdown,
                chk_anms, chk_clahe, chk_shadow, chk_wallis,
                ref_raw_file, sec_raw_file,
            ],
            outputs=[
                warped_result_image, checkerboard_image, vector_overlay_image, blend_image,
                report_text, metrics_json, telemetry_text, blend_inputs_state,
                csv_btn, json_btn, geotiff_btn, png_btn, dossier_image, zip_btn
            ],
            # Gradio's runtime-based ETA substantially underestimates some
            # planetary scenes; the adjacent notice gives a realistic range.
            show_progress="hidden",
        )

        blend_alpha.change(
            fn=update_interactive_blend,
            inputs=[blend_inputs_state, blend_alpha],
            outputs=[blend_image],
        )
        
        # Native Gradio examples load image paths directly into the image widgets.
        gr.Markdown("### Example Pairs")
        examples_dir = Path(__file__).resolve().parent / "docs" / "assets" / "examples"
        example_specs = (
            ("nac_reference_ohrc.png", "ohrc_secondary.png"),
            ("nac_reference_tmc2.png", "tmc2_secondary.png"),
            ("nac_reference_iirs.png", "iirs_band125_secondary.png"),
            ("synthetic_groundtruth_reference.png", "synthetic_groundtruth_secondary.png"),
        )
        example_rows = []
        for reference_filename, secondary_filename in example_specs:
            reference_path = examples_dir / reference_filename
            secondary_path = examples_dir / secondary_filename
            example_rows.append([str(reference_path), str(secondary_path)])
        if example_rows:
            gr.Examples(
                examples=example_rows,
                inputs=[ref_input, sec_input],
                cache_examples=False,
                examples_per_page=4,
                label="Select Lunar Data Pair Presets"
            )
        else:
            gr.Markdown("No example image pairs are available in this deployment.")

        with gr.Accordion("Scientific Foundation & SAC-ISRO Benchmark Alignment", open=False):
            gr.Markdown(
                "**Citation:** R. Makharia, J. G. Singla, Amitabh, N. Dube, and H. Sharma, "
                "“Comparative Evaluation of Traditional and Deep Learning Feature Matching "
                "Algorithms using Chandrayaan-2 Lunar Data,” 2025, arXiv:2509.04775. "
                "[Paper and metadata](https://arxiv.org/abs/2509.04775). The study compares "
                "traditional and learned matchers across lunar sensor pairs and reports that "
                "illumination and modality differences can degrade classical matching, while "
                "learned matching improves robustness in difficult polar/cross-modal cases.\n\n"
                "**CHANDRA-ALIGN implementation:** LightGlue/ALIKED is the default matcher, "
                "with SIFT + RANSAC as the CPU fallback when learned matching is unavailable "
                "or its geometric fit is weak. RIFT2 is non-functional in current validation "
                "and is attempted only when the operator explicitly sets "
                "`CHANDRA_ENABLE_RIFT2=1`. The report records matcher selection, fallback "
                "reason, in-sample and held-out RMSE/MAE, and gate telemetry."
            )
    
    return interface


# Build and export
try:
    interface = build_interface()
except Exception as startup_error:
    print("CHANDRA-ALIGN interface startup failed.", file=sys.stderr, flush=True)
    print(traceback.format_exc(), file=sys.stderr, flush=True)
    with gr.Blocks(title="CHANDRA-ALIGN startup diagnostic") as interface:
        gr.Markdown("# CHANDRA-ALIGN could not initialize")
        gr.Markdown(
            "The service process started, but interface construction failed. "
            "Check the Space runtime logs for the traceback."
        )
        gr.Code(
            value=f"{type(startup_error).__name__}: {startup_error}",
            language="text",
            label="Startup error",
        )


if __name__ == "__main__":
    # HF Spaces currently enables Gradio SSR by default, which inserts a Node
    # proxy in front of the Python app. Serve client-side to keep the Space on
    # its direct Python listener and remove that extra cold-start failure path.
    interface.launch(ssr_mode=False)
