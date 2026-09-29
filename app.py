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
import html
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
    select_quadrant_keypoints, select_quadrant_balanced_matches,
)
from chandra_align.alignment import decompose_partial_affine
from chandra_align.refine import refine_subpixel_ncc
from chandra_align.metrics import (
    compute_deformation_field, grid_deformation_analysis, compute_ground_metrics,
    get_sensor_pixel_scale, metrics_bundle_with_ground, ResidualVector, GroundMetrics,
    compute_quadrant_metrics, format_quadrant_html, build_judge_metrics_summary,
    validate_registration_gate,
)
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
        lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0, 0)
        if hi <= lo:
            image = np.zeros(image.shape, dtype=np.uint8)
        else:
            image = np.clip((finite - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image)


def generate_synthetic_lunar_pair(ref_path: str, sec_path: str) -> None:
    """
    Generate synthetic lunar terrain image pair with known homography shift.
    Saves reference and secondary images to disk.
    """
    h, w = 512, 512
    ref = np.zeros((h, w), dtype=np.float32)
    np.random.seed(42)

    # Generate crater-like features
    for _ in range(40):
        cx, cy = np.random.randint(40, w - 40), np.random.randint(40, h - 40)
        r = np.random.randint(8, 35)
        cv2.circle(ref, (cx, cy), r, np.random.uniform(0.2, 1.0), -1)

    # Generate ridge-like linear features
    for _ in range(20):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
        cv2.line(ref, (x1, y1), (x2, y2), np.random.uniform(0.15, 0.7), 2)

    # Apply Gaussian blur for realistic texture
    ref = cv2.GaussianBlur(ref, (7, 7), 1.5)
    ref = (ref * 255).astype(np.uint8)

    # Apply known homography (simulate orbital shift + rotation)
    angle = np.deg2rad(1.2)
    tx, ty = 14.5, -9.3
    scale = 1.008
    M = np.array([
        [scale * np.cos(angle), -scale * np.sin(angle), tx],
        [scale * np.sin(angle), scale * np.cos(angle), ty]
    ], dtype=np.float32)

    sec = cv2.warpAffine(ref, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # Add slight radiometric variation to secondary
    noise = np.random.normal(0.0, 3.0, sec.shape).astype(np.float32)
    sec = np.clip(sec.astype(np.float32) * 0.95 + noise * 0.05, 0, 255).astype(np.uint8)

    if not cv2.imwrite(ref_path, ref) or not cv2.imwrite(sec_path, sec):
        raise OSError("OpenCV could not write the synthetic pitch demo image pair")


def ensure_sample_files() -> tuple[str, str]:
    """
    Check for cached sample files; generate in writable temporary storage if missing.
    Returns paths to reference and secondary sample images.
    """
    demo_dir = Path(tempfile.gettempdir()) / "chandra_align_pitch_demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    ref_path = demo_dir / "sample_ref.png"
    sec_path = demo_dir / "sample_sec.png"

    pair_is_readable = (
        ref_path.is_file()
        and sec_path.is_file()
        and cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED) is not None
        and cv2.imread(str(sec_path), cv2.IMREAD_UNCHANGED) is not None
    )
    if not pair_is_readable:
        generate_synthetic_lunar_pair(str(ref_path), str(sec_path))
    if (
        not ref_path.is_file()
        or not sec_path.is_file()
        or cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED) is None
        or cv2.imread(str(sec_path), cv2.IMREAD_UNCHANGED) is None
    ):
        raise RuntimeError("Could not create the synthetic pitch demo image pair")

    return str(ref_path), str(sec_path)


def match_pair_hf(
    img1: np.ndarray, img2: np.ndarray, ransac_threshold_px: float = 3.0
) -> tuple[np.ndarray, np.ndarray, str, dict]:
    """Run phase-congruency RIFT2 first and disclose any LightGlue/ALIKED handoff."""
    # SIFT and most deep matcher frontends require finite uint8 image arrays.
    def as_uint8(image):
        image = ensure_uint8(image)
        if image.ndim != 2 or image.size == 0:
            raise ValueError("Feature matching requires non-empty grayscale images")
        if image.dtype == np.uint8:
            return np.ascontiguousarray(image)
        image = np.nan_to_num(image.astype(np.float32, copy=False))
        lo, hi = np.percentile(image, [1, 99])
        return np.zeros(image.shape, np.uint8) if hi <= lo else np.clip((image - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

    sift_input1, sift_input2 = np.asarray(img1), np.asarray(img2)
    img1, img2 = as_uint8(sift_input1), as_uint8(sift_input2)
    diagnostics = {
        "primary_engine": "Phase Congruency + Quad-Tree",
        "fallback_triggered": False,
        "fallback_reason": None,
        "registration_engine": "Phase Congruency + Quad-Tree",
    }
    # RIFT2 computes phase-congruency features and is the deterministic primary.
    try:
        from chandra_align.matcher import RIFT2Matcher
        primary = RIFT2Matcher(npt=2048)
        src_pts, dst_pts = primary.match(img1, img2)
        if len(src_pts) >= 4:
            return src_pts, dst_pts, "RIFT2 (Phase Congruency)", diagnostics
        primary_error = "RIFT2 returned fewer than four correspondences"
        fallback_reason = "Low Inlier Ratio (< 0.15)"
    except Exception as exc:
        primary_error = str(exc)
        fallback_reason = "Execution Exception"

    # If the primary cannot produce a geometric candidate at all, make one
    # bounded LightGlue/ALIKED attempt before trying the classical fallback.
    diagnostics.update(fallback_triggered=True, fallback_reason=fallback_reason)
    try:
        from chandra_align.matching.deep_matchers import DeepMatcherChain, LightGlueALIKEDMatcher
        result = DeepMatcherChain(matchers=[LightGlueALIKEDMatcher(max_keypoints=2048)]).match(
            img1, img2, min_matches=4
        )
        if len(result.pts_src) >= 4:
            diagnostics["registration_engine"] = f"LightGlue/ALIKED ({result.matcher_name})"
            return result.pts_src, result.pts_ref, diagnostics["registration_engine"], diagnostics
        diagnostics["fallback_error"] = result.error_msg or "LightGlue/ALIKED returned fewer than four correspondences"
    except Exception as exc:
        diagnostics["fallback_error"] = str(exc)

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
            key_ref, idx_ref = select_quadrant_keypoints(key_ref, reference_image.shape, 50)
            key_sec, idx_sec = select_quadrant_keypoints(key_sec, secondary_image.shape, 50)
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
            return (candidate_ref.astype(np.float32), candidate_sec.astype(np.float32),
                    diagnostics["registration_engine"], diagnostics)
        diagnostics["sift_error"] = "No SIFT candidate reached eight partial-affine RANSAC inliers"
    except Exception as exc:
        diagnostics["sift_error"] = str(exc)
    diagnostics["primary_error"] = primary_error
    return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), "Failed", diagnostics


def _align_core(
    ref_img: np.ndarray,
    sec_img: np.ndarray,
    pixel_scale_m: float = 0.25,
    enable_clahe: bool = True,
    clahe_clip_limit: float = 3.0,
    enable_shadow_suppression: bool = True,
    enable_wallis: bool = False,
    wallis_target_mean: float = 128.0,
    wallis_target_std: float = 50.0,
    enforce_uniform_distribution: bool = True,
    sensor_pair_mode: str = "Optical <-> Optical",
    secondary_sensor_name: str = "TMC-2",
    reference_sensor_name: str = "OHRC",
    max_image_dimension: int = MAX_IMAGE_DIMENSION,
) -> dict:
    """
    Core alignment logic - runs on GPU when called from process_alignment.
    Returns comprehensive results dictionary.
    """
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
    pts_ref, pts_sec, engine_name, execution_diagnostics = match_pair_hf(
        match_ref, match_sec, ransac_threshold_px
    )
    if ref_match_scale != 1.0:
        pts_ref = pts_ref / ref_match_scale
    if sec_match_scale != 1.0:
        pts_sec = pts_sec / sec_match_scale

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
    # is bounded to one LightGlue/ALIKED pass with at most 2048 keypoints.
    if not execution_diagnostics["fallback_triggered"]:
        inlier_ratio = inlier_cnt / max(len(pts_ref), 1)
        primary_uniformity = spatial_distribution_metrics(
            pts_ref[inliers], ref_original.shape[:2], (8, 8)
        )["uniformity"] if inlier_cnt else 0.0
        fallback_reason = None
        if geometry_exception:
            fallback_reason = "Execution Exception"
        elif inlier_ratio < 0.15:
            fallback_reason = "Low Inlier Ratio (< 0.15)"
        elif primary_uniformity < 0.125:
            fallback_reason = "High Spatial Entropy Deficit"

        if fallback_reason:
            execution_diagnostics.update(
                fallback_triggered=True, fallback_reason=fallback_reason
            )
            try:
                from chandra_align.matching.deep_matchers import DeepMatcherChain, LightGlueALIKEDMatcher
                fallback = DeepMatcherChain(
                    matchers=[LightGlueALIKEDMatcher(max_keypoints=2048)]
                ).match(match_ref, match_sec, min_matches=4)
                fallback_ref, fallback_sec = fallback.pts_src, fallback.pts_ref
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
                if len(fallback_ref) >= 3:
                    fallback_affine, fallback_inliers = _estimate_partial_affine_with_threshold(
                        fallback_sec, fallback_ref, ransac_threshold_px
                    )
                    if fallback_affine is not None:
                        if int(fallback_inliers.sum()) >= MIN_REGISTRATION_INLIERS:
                            pts_ref, pts_sec = fallback_ref, fallback_sec
                            affine_matrix, inliers = fallback_affine, fallback_inliers
                            inlier_cnt = int(inliers.sum())
                            engine_name = f"LightGlue/ALIKED ({fallback.matcher_name})"
                            execution_diagnostics["registration_engine"] = engine_name
                if execution_diagnostics["registration_engine"] == "Phase Congruency + Quad-Tree":
                    execution_diagnostics["fallback_error"] = fallback.error_msg or "Fallback did not yield a valid partial affine with at least eight inliers"
            except Exception as exc:
                execution_diagnostics["fallback_error"] = str(exc)

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

    rmse_px = float(np.sqrt(np.mean(residuals_mag ** 2))) if len(residuals_mag) > 0 else 0.0
    mae_px = float(np.mean(residuals_mag)) if len(residuals_mag) > 0 else 0.0
    affine_telemetry = decompose_partial_affine(affine_matrix)
    status_message, status_code = validate_registration_gate(
        rmse_px, inlier_cnt, MIN_REGISTRATION_INLIERS,
        quadrant_spatial_entropy, quadrant_metrics,
    )
    judge_metrics = build_judge_metrics_summary(
        rmse_px,
        inlier_cnt,
        len(pts_ref),
        quadrant_spatial_entropy,
        quadrant_metrics,
        min_inliers=MIN_REGISTRATION_INLIERS,
    )

    engine_lower = engine_name.lower()
    if "sift" in engine_lower:
        engine_used = "Fallback (SIFT)"
    elif "lightglue" in engine_lower or "aliked" in engine_lower:
        engine_used = "Fallback (LightGlue/ALIKED)"
    else:
        engine_used = "Primary (Phase Congruency)"
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
            "rmse_px": rmse_px,
            "quadrant_spatial_entropy": quadrant_spatial_entropy,
            "engine_used": engine_used,
            "execution_device": execution_device,
            "affine_telemetry": None,
        }

    H = _homogeneous_affine(affine_matrix)
    warped_sec = cv2.warpAffine(sec_original, affine_matrix, (ref_original.shape[1], ref_original.shape[0]))
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
        "inlier_cnt": inlier_cnt,
        "total_matches": len(pts_ref),
        "rmse_px": rmse_px,
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
    return build_judge_metrics_summary(
        0.0, inlier_count, total_correspondences, 0.0, empty_quadrants
    )


def format_telemetry_report(
    status, inliers, total_pts, inlier_ratio, spatial_entropy,
    quad_counts, rmse=None, affine_telemetry=None, engine="N/A", device="N/A",
    status_message=None, active_quadrants=None,
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
    rmse_pass = rmse_value <= 0.5
    entropy_pass = entropy_value >= 0.75
    quadrant_pass = active >= 3
    status_text = status_message or (
        "ACCEPTED (Sub-Pixel Precision)" if tier1 else
        "COARSE ALIGNMENT (Regional Fit Advisory)" if tier2 else
        "REJECTED (Degenerate Single-Quadrant Cluster)"
    )
    check = lambda passed: "✓" if passed else "✗"
    lines = [
        "===================================================",
        "PHOTOGRAMMETRIC TELEMETRY REPORT",
        "===================================================",
        f"Registration Status   : {status_text}",
        f"Hardware Runtime Mode : {device}",
        "",
        "VALIDATION GATE CHECKLIST:",
        f"[{check(rmse_pass)}] Sub-Pixel Precision  : RMSE <= 0.50 px (Measured: {rmse_value:.4f} px)" if math.isfinite(rmse_value) else "[✗] Sub-Pixel Precision  : RMSE <= 0.50 px (Measured: N/A)",
        f"[{check(entropy_pass)}] Spatial Spread Score : Entropy >= 0.75 (Measured: {entropy_value:.4f} / 2.00)",
        f"[{check(quadrant_pass)}] Quadrant Distribution: Active Quads >= 3 (Measured: {active} / 4)",
        "",
        f"QUADRANT BREAKDOWN: Q1:{int(counts[0])} | Q2:{int(counts[1])} | Q3:{int(counts[2])} | Q4:{int(counts[3])}",
        "---------------------------------------------------",
    ]
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


def _rejected_output_tuple(summary, status_message=None, image_shape=(768, 1024), total_pts=0):
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
    report = (
        f"❌ {message}\n\n"
        "REGISTRATION REJECTED: Insufficient Spatial Uniformity\n"
        "Sub-pixel alignment gate prevented degenerate warp execution."
    )
    telemetry = format_telemetry_report(
        metrics.get("status_code", "FAILED"), inliers, total_pts, ratio,
        entropy, counts,
    )
    return (
        banner_pil, banner_pil, banner_pil, banner_pil,
        report, metrics, telemetry, (banner_bgr, banner_bgr),
        None, None, None, None, None,
    )


def _minimal_rejection_output_tuple(status_message, summary=None):
    """Dependency-light emergency result that always matches the 13 outputs."""
    blank = np.zeros((512, 512, 3), dtype=np.uint8)
    preview = Image.fromarray(blank)
    metrics = summary if isinstance(summary, dict) else _failed_judge_metrics_summary()
    report = f"REGISTRATION REJECTED: {status_message}"
    telemetry = (
        "PHOTOGRAMMETRIC TELEMETRY REPORT\n"
        "Registration Status : REJECTED\n"
        "Recovered Transform : N/A — Alignment Rejected\n"
        f"Reason               : {status_message}"
    )
    return (
        preview, preview, preview, preview,
        report, metrics, telemetry, (blank, blank),
        None, None, None, None, None,
    )


def _safe_rejection_output_tuple(summary, status_message, image_shape=(768, 1024), total_pts=0):
    """Prevent banner/telemetry formatting failures from escaping the callback."""
    try:
        outputs = tuple(_rejected_output_tuple(
            summary, status_message=status_message,
            image_shape=image_shape, total_pts=total_pts,
        ))
        if len(outputs) == 13:
            return outputs
        raise ValueError(f"Rejection callback produced {len(outputs)} outputs, expected 13")
    except Exception:
        print("Failed to render the normal rejection banner; using a minimal safe result.",
              file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return _minimal_rejection_output_tuple(status_message, summary)


def process_alignment(
    ref_file,
    sec_file,
    pixel_scale_m: float = 0.25,
    enable_clahe: bool = True,
    clahe_clip_limit: float = 3.0,
    enable_shadow_suppression: bool = True,
    enable_wallis: bool = False,
    wallis_target_mean: float = 128.0,
    wallis_target_std: float = 50.0,
    sensor_name: str = "OHRC",
    secondary_sensor_name: str = "TMC-2",
    sensor_pair_mode: str = "Optical <-> Optical",
    enforce_uniform_distribution: bool = True
):
    """
    Main alignment pipeline - runs on GPU when called from process_wrapper.
    Returns visual previews, telemetry, judge metrics, and scientific export paths.
    """
    # Pre-allocate safe values before image I/O or processing. The declared
    # Gradio return order remains the fixed 13-component contract below.
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
    banner_img = default_blank_img.copy()
    active_badge = "Runtime Mode: Active"
    checklist_status = "INITIALIZING"
    dossier_path = None
    report_text = "Initializing pipeline..."
    reference_image = None

    if ref_file is None or sec_file is None:
        return _safe_rejection_output_tuple(
            _failed_judge_metrics_summary(),
            "Empty input: provide both Reference and Secondary surface frames.",
        )

    try:
        is_iirs_pair = sensor_pair_mode == "Optical <-> Infrared"
        max_image_dimension = IIRS_MAX_IMAGE_DIMENSION if is_iirs_pair else MAX_IMAGE_DIMENSION
        ref_img = _read_grayscale_image(ref_file, max_image_dimension)
        sec_img = _read_grayscale_image(sec_file, max_image_dimension)
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
            max_image_dimension=max_image_dimension
        )

        if result.get("rejected"):
            return _rejected_output_tuple(
                result["judge_metrics"],
                status_message=result["status_message"],
                image_shape=result["ref_original"].shape[:2],
                total_pts=result["total_matches"],
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
        mae_px = result["mae_px"]
        H = result["H"]
        deformation_vectors = result["deformation_vectors"]
        grid_analysis = result["grid_analysis"]
        ground_metrics = result["ground_metrics"]
        uniformity = result["uniformity"]
        spatial_entropy = result["spatial_entropy"]
        quadrant_entropy = result["quadrant_spatial_entropy"]
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
        
        report = (
            f"{'✅' if status_code == 'SUCCESS_SUBPIXEL' else '🟠' if status_code == 'COARSE_ADVISORY' else '❌'} {status_message}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Matcher Engine: {engine_name}\n"
            f"Registration Transform: 4-DOF Partial Affine (2x3), RANSAC threshold {result['ransac_threshold_px']:.1f} px\n"
            f"Registration Matrix Engine: {result['execution_diagnostics'].get('registration_engine', engine_name)}\n"
            f"Primary Engine: {result['execution_diagnostics'].get('primary_engine', 'Phase Congruency + Quad-Tree')}\n"
            f"Fallback Triggered: {result['execution_diagnostics'].get('fallback_triggered', False)}\n"
            f"Fallback Reason: {result['execution_diagnostics'].get('fallback_reason') or 'None'}\n"
            f"Sensor Pair Mode: {result['sensor_pair_mode']}\n"
            f"Verified Inliers: {inlier_cnt} / {total_matches} ({inlier_pct:.1f}%)\n"
            f"Spatial Uniformity U: {uniformity:.4f} (entropy {spatial_entropy:.4f} nats)\n"
            f"Refined Matches: {refinement_stats.get('refined_pairs', 0)} ({refinement_stats.get('status', 'not run')})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"PIXEL METRICS:\n"
            f"  RMSE: {rmse_px:.4f} px\n"
            f"  MAE:  {mae_px:.4f} px\n"
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
            affine_telemetry=affine_telemetry,
            engine=result["engine_used"], device=result["execution_device"],
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
            mae_px=mae_px,
            engine_name=engine_name,
            inlier_count=inlier_cnt,
            total_matches=total_matches
        )
        fig_to_file(fig, str(viz_path), dpi=300)
        export_paths["dossier_png"] = str(viz_path)

        # Create zip package
        zip_path = output_dir / f"{base_name}_package.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for key, path in export_paths.items():
                if path and Path(path).exists():
                    zf.write(path, Path(path).name)
        
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
            str(viz_path)
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
    values = list(outputs) if isinstance(outputs, (tuple, list)) else [None] * 13
    if len(values) != 13:
        values = (values + [None] * 13)[:13]
    report = values[4] if isinstance(values[4], str) else ""
    values[4] = _format_metric_report(f"{report.rstrip()}\n\n{status}".strip())
    telemetry = values[6] if isinstance(values[6], str) else ""
    device = "GPU (ZeroGPU)" if "ZeroGPU" in status else "CPU"
    telemetry = re.sub(
        r"(?m)^Hardware Runtime Mode\s*:\s*.*$",
        f"Hardware Runtime Mode : {'GPU (ZeroGPU Active)' if 'ZeroGPU' in status else 'CPU Fallback'}",
        telemetry,
    )
    values[6] = telemetry
    return tuple(values)


def _format_metric_report(report: str) -> str:
    """Render the existing plain-text report as UI-only metric cards."""
    quadrant_html = ""
    quadrant_match = re.search(
        r"<!--QUADRANT_METRICS_START-->(.*?)<!--QUADRANT_METRICS_END-->",
        report,
        flags=re.DOTALL,
    )
    plain_report = report
    if quadrant_match:
        quadrant_html = quadrant_match.group(1).strip()
        plain_report = (report[:quadrant_match.start()] + report[quadrant_match.end():]).strip()
    escaped_report = html.escape(plain_report)
    accepted = report.startswith("✅ ACCEPTED (Sub-Pixel Precision)")
    coarse = report.startswith("🟠 COARSE ALIGNMENT")
    badge_class = "pass-badge" if accepted else "advisory-badge" if coarse else "fail-badge"
    badge_text = "SUB-PIXEL ACCEPTED" if accepted else "COARSE ADVISORY" if coarse else "REGISTRATION REJECTED"

    def first_match(pattern: str, default: str = "—") -> str:
        match = re.search(pattern, report, flags=re.MULTILINE)
        return html.escape(match.group(1).strip()) if match else default

    consensus = first_match(r"^Verified Inliers:\s*(.+)$")
    pixel_rmse = first_match(r"^\s*RMSE:\s*([\d.eE+-]+\s*px)")
    ground_rmse = first_match(r"^\s*RMSE:\s*([\d.eE+-]+\s*m)\s*$")
    sensor_mode = first_match(r"^Sensor Pair Mode:\s*(.+)$")
    threshold = first_match(r"^Registration Transform:.*RANSAC threshold\s*([\d.]+\s*px)")

    return (
        '<div class="metric-card">'
        f'<span class="{badge_class}">{badge_text}</span> '
        f'<span>{sensor_mode}</span>'
        '</div>'
        '<div class="metric-card"><strong>Inlier Consensus Ratio</strong><br>'
        f'<span>{consensus}</span></div>'
        '<div class="metric-card"><strong>Fit Residual / RMSE</strong><br>'
        f'<span>{pixel_rmse} · {ground_rmse}</span></div>'
        '<div class="metric-card"><strong>RANSAC Threshold</strong><br>'
        f'<span>{threshold}</span></div>'
        + ('<div class="metric-card advisory-banner"><strong>NOTICE:</strong> Regional fit established (RMSE &lt;= 2.5 px). Sub-pixel refinement recommended for DEM production.</div>' if coarse else '')
        + f'{quadrant_html}'
        '<details class="metric-card"><summary>Full processing report</summary>'
        f'<pre>{escaped_report}</pre></details>'
    )


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


@spaces.GPU(duration=25)
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
        if len(normalized) != 13:
            raise ValueError(f"Alignment callback returned {len(normalized)} outputs; expected 13")
        return normalized
    except Exception as callback_error:
        # Last-resort boundary: Gradio callbacks must not leak an exception or
        # return a shape/type mismatch that can surface as an HTTP 500.
        print("Unhandled alignment callback error; returning a safe 13-output rejection.",
              file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        try:
            fallback = _minimal_rejection_output_tuple(
                f"Registration callback failed: {callback_error}", summary={}
            )
            return tuple(fallback) if len(fallback) == 13 else (None,) * 13
        except Exception:
            print("Emergency callback fallback construction also failed.",
                  file=sys.stderr, flush=True)
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            return (None,) * 13


def update_interactive_blend(images, alpha):
    """Rebuild the blend preview from cached registration rasters and slider alpha."""
    if not isinstance(images, (tuple, list)) or len(images) != 2:
        return None
    try:
        return Image.fromarray(create_interactive_blend(images[0], images[1], alpha=alpha))
    except (TypeError, ValueError):
        return None


# Build Gradio interface with advanced controls
def build_interface():
    """Build the Gradio interface with scientific controls."""
    
    custom_css = """
    body, .gradio-container { background-color: #0f172a !important; color: #f8fafc !important; }
    .metric-card { background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px; margin-top: 6px; }
    .pass-badge { background-color: #065f46; color: #34d399; padding: 4px 8px; border-radius: 4px; font-weight: 600; }
    .advisory-badge { background-color: #78350f; color: #fbbf24; padding: 4px 8px; border-radius: 4px; font-weight: 600; }
    .advisory-banner { border-color: #f59e0b; color: #fcd34d; }
    .fail-badge { background-color: #881337; color: #f87171; padding: 4px 8px; border-radius: 4px; font-weight: 600; }
    .accent-button button { background: linear-gradient(90deg, #4f46e5, #0891b2) !important; font-weight: 700 !important; }
    .runtime-mode-badge { display: inline-block; padding: 6px 12px; border-radius: 999px; background: #1e293b; border: 1px solid #334155; color: #e2e8f0; }
    .metric-card pre { white-space: pre-wrap; color: #cbd5e1; }
    """

    with gr.Blocks(
        title="CHANDRA-ALIGN: Lunar Photogrammetric Workstation",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="cyan", neutral_hue="slate"),
        css=custom_css,
    ) as interface:
        gr.Markdown(f"### {RUNTIME_MODE_BADGE}", elem_classes=["runtime-mode-badge"])
        gr.Markdown(
            "# 🌙 CHANDRA-ALIGN: Lunar Cross-Sensor Photogrammetric Workstation\n"
            "Sub-pixel registration for Chandrayaan-2 OHRC, TMC-2, IIRS, DF-SAR & LRO NAC imagery. "
            "Achieves scientific-grade photogrammetric accuracy with ground-resolution error translation."
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input panel
                gr.Markdown("### 📥 Input Frames")
                gr.Markdown(
                    "Accepts multi-agency satellite imagery and sensor formats, including "
                    "ISRO Chandrayaan payloads, NASA LRO products, and map imagery."
                )
                ref_file = gr.File(
                    label="Reference Image (ISRO Chandrayaan / NASA LRO / Base Map)",
                    file_types=[".png", ".tif", ".tiff", ".jpg", ".jpeg"]
                )
                sec_file = gr.File(
                    label="Secondary Image (Onboard Sensor / Target Frame)",
                    file_types=[".png", ".tif", ".tiff", ".jpg", ".jpeg"]
                )
                
                # Advanced Controls
                with gr.Accordion("⚙️ Advanced Algorithm Parameters", open=False):
                    with gr.Accordion("Sensor Pair & Resolution", open=False):
                        sensor_dropdown = gr.Dropdown(
                            choices=["OHRC", "TMC-2", "IIRS", "DF-SAR", "LROC_NAC", "LROC_WAC", "KAGUYA_TC", "Custom"],
                            value="OHRC",
                            label="Reference Sensor"
                        )
                        secondary_sensor_dropdown = gr.Dropdown(
                            choices=["OHRC", "TMC-2", "IIRS", "DF-SAR", "LROC_NAC", "LROC_WAC", "KAGUYA_TC", "Custom"],
                            value="TMC-2",
                            label="Secondary Sensor"
                        )
                        sensor_pair_mode = gr.Dropdown(
                            choices=["Optical <-> Optical", "Optical <-> Infrared"],
                            value="Optical <-> Optical",
                            label="Sensor Pair Mode"
                        )
                        pixel_scale = gr.Number(
                            value=0.25,
                            label="Reference Pixel Scale (m/px)",
                            minimum=0.01,
                            maximum=100.0,
                            step=0.01,
                            info="Ground resolution in meters per pixel"
                        )

                    with gr.Accordion("Illumination-Invariant Preprocessing", open=False):
                        enforce_uniformity = gr.Checkbox(
                            value=True,
                            label="Enforce Uniform Keypoint Distribution (ANMS / 8×8 buckets)"
                        )
                        enable_clahe = gr.Checkbox(value=True, label="CLAHE Enhancement")
                        clahe_clip = gr.Slider(
                            minimum=1.0, maximum=10.0, value=3.0, step=0.5,
                            label="CLAHE Clip Limit"
                        )
                        enable_shadow = gr.Checkbox(value=True, label="Shadow Suppression")
                        enable_wallis = gr.Checkbox(value=False, label="Wallis Filter (Experimental)")
                        with gr.Row(visible=False) as wallis_params:
                            wallis_mean = gr.Number(value=128.0, label="Target Mean")
                            wallis_std = gr.Number(value=50.0, label="Target Std")

                        enable_wallis.change(
                            lambda x: gr.update(visible=x),
                            inputs=[enable_wallis],
                            outputs=[wallis_params]
                        )
                
                process_btn = gr.Button("🚀 Process Alignment", variant="primary", size="lg")
                btn_judge_demo = gr.Button(
                    "⚡ Run Pitch Demo", variant="primary", elem_classes=["accent-button"]
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
                    elem_classes=["metric-report"],
                )
                metrics_json = gr.JSON(label="Judge Metrics Summary JSON")
                telemetry_text = gr.Textbox(
                    label="Photogrammetric Telemetry Report",
                    lines=15,
                    interactive=False,
                    elem_classes=["metric-card"],
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
        process_btn.click(
            fn=process_wrapper,
            api_name="predict",
            inputs=[
                ref_file, sec_file, sensor_dropdown, secondary_sensor_dropdown,
                sensor_pair_mode, enforce_uniformity, pixel_scale,
                enable_clahe, clahe_clip, enable_shadow, enable_wallis,
                wallis_mean, wallis_std
            ],
            outputs=[
                warped_result_image, checkerboard_image, vector_overlay_image, blend_image,
                report_text, metrics_json, telemetry_text, blend_inputs_state,
                csv_btn, json_btn, geotiff_btn, png_btn, dossier_image
            ]
        )

        def apply_judge_demo_preset():
            """Load the synthetic pair and force the calibrated demo defaults."""
            sample_ref, sample_sec = ensure_sample_files()
            gr.Info("Applied preset defaults")
            return (
                sample_ref, sample_sec,
                "OHRC", "OHRC", "Optical <-> Optical", False, 0.25,
                True, 3.0, False, False, 128.0, 50.0,
            )

        btn_judge_demo.click(
            fn=apply_judge_demo_preset,
            inputs=None,
            outputs=[
                ref_file, sec_file, sensor_dropdown, secondary_sensor_dropdown,
                sensor_pair_mode, enforce_uniformity, pixel_scale,
                enable_clahe, clahe_clip, enable_shadow, enable_wallis,
                wallis_mean, wallis_std,
            ],
            show_progress="hidden",
        ).then(
            fn=process_wrapper,
            inputs=[
                ref_file, sec_file, sensor_dropdown, secondary_sensor_dropdown,
                sensor_pair_mode, enforce_uniformity, pixel_scale,
                enable_clahe, clahe_clip, enable_shadow, enable_wallis,
                wallis_mean, wallis_std,
            ],
            outputs=[
                warped_result_image, checkerboard_image, vector_overlay_image, blend_image,
                report_text, metrics_json, telemetry_text, blend_inputs_state,
                csv_btn, json_btn, geotiff_btn, png_btn, dossier_image,
            ],
            show_progress="hidden",
        )

        blend_alpha.change(
            fn=update_interactive_blend,
            inputs=[blend_inputs_state, blend_alpha],
            outputs=[blend_image],
        )
        
        # Sensor change updates pixel scale
        def update_pixel_scale(sensor):
            return get_sensor_pixel_scale(sensor)
        
        sensor_dropdown.change(
            fn=update_pixel_scale,
            inputs=[sensor_dropdown],
            outputs=[pixel_scale]
        )
        
        # Examples
        gr.Markdown("### 📝 Example Pairs")
        examples_dir = Path(__file__).resolve().parent / "docs" / "assets" / "examples"
        example_specs = (
            ("nac_reference_ohrc.png", "ohrc_secondary.png", "LROC_NAC", "OHRC",
             "Optical <-> Optical", "OHRC vs NASA LROC NAC · 2× GSD gap", 0.5, 3.0),
            ("nac_reference_tmc2.png", "tmc2_secondary.png", "LROC_NAC", "TMC-2",
             "Optical <-> Optical", "TMC-2 vs NASA LROC NAC · 10× GSD gap", 0.5, 3.0),
            ("nac_reference_iirs.png", "iirs_band125_secondary.png", "LROC_NAC", "IIRS",
             "Optical <-> Infrared", "IIRS Band 125 (2.802 µm) vs NASA LROC NAC · 160× GSD gap", 0.5, 4.5),
            ("synthetic_groundtruth_reference.png", "synthetic_groundtruth_secondary.png",
             "OHRC", "OHRC", "Optical <-> Optical",
             "Synthetic Ground Truth · 1:1 GSD", 0.25, 3.0),
        )
        example_rows = []
        example_labels = []
        for (reference_filename, secondary_filename, reference_sensor, secondary_sensor,
             pair_mode, label, pixel_scale_m, clip_limit) in example_specs:
            reference_path = examples_dir / reference_filename
            secondary_path = examples_dir / secondary_filename
            if not reference_path.is_file() or not secondary_path.is_file():
                print(
                    f"Skipping incomplete example {label!r}: "
                    f"missing {reference_path if not reference_path.is_file() else secondary_path}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            example_rows.append([
                str(reference_path), str(secondary_path), reference_sensor, secondary_sensor,
                pair_mode, False, pixel_scale_m, True, clip_limit, False, False, 128.0, 50.0
            ])
            example_labels.append(label)
        if example_rows:
            gr.Examples(
                examples=example_rows,
                inputs=[
                    ref_file, sec_file, sensor_dropdown, secondary_sensor_dropdown,
                    sensor_pair_mode, enforce_uniformity, pixel_scale,
                    enable_clahe, clahe_clip, enable_shadow, enable_wallis,
                    wallis_mean, wallis_std
                ],
                outputs=[
                    warped_result_image, checkerboard_image, vector_overlay_image, blend_image,
                    report_text, metrics_json, telemetry_text, blend_inputs_state,
                    csv_btn, json_btn, geotiff_btn, png_btn, dossier_image
                ],
                fn=process_wrapper,
                cache_examples=False,
                examples_per_page=3,
                run_on_click=True,
                example_labels=example_labels,
                label="One-click lunar alignment presets"
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
                "**CHANDRA-ALIGN implementation:** RIFT2 phase-congruency features are the "
                "deterministic, bit-exact primary matcher, with quad-tree distribution checks and geometric verification. "
                "A single bounded LightGlue/ALIKED escalation is attempted when the primary "
                "solution has a low inlier ratio, poor spatial coverage, or cannot execute. "
                "The report records the primary engine, whether escalation was attempted, its "
                "trigger, and which engine supplied the accepted registration matrix."
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
