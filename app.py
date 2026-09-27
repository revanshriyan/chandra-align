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
import traceback
from pathlib import Path
from PIL import Image

# Import new modules
from chandra_align.preprocessing import (
    apply_clahe, detect_shadows, apply_wallis_filter, preprocess_multimodal_pair,
    preprocess_iirs_raster, resize_to_common_ground_sample
)
from chandra_align.features import select_distributed_matches, spatial_distribution_metrics
from chandra_align.refine import refine_subpixel_ncc
from chandra_align.metrics import (
    compute_deformation_field, grid_deformation_analysis, compute_ground_metrics,
    get_sensor_pixel_scale, metrics_bundle_with_ground, ResidualVector, GroundMetrics
)
from chandra_align.visualization import (
    create_combined_visualization, fig_to_file
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


def _read_grayscale_image(image_file, max_dimension: int = MAX_IMAGE_DIMENSION) -> np.ndarray:
    """Decode an image input, normalize its channels/depth, and cap its dimensions."""
    if isinstance(image_file, np.ndarray):
        image = np.asarray(image_file)
    elif isinstance(image_file, Image.Image):
        image = np.asarray(image_file)
    else:
        path = getattr(image_file, "name", image_file)
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
    sec = cv2.addWeighted(sec, 0.95, np.random.normal(0, 3, sec.shape).astype(np.uint8), 0.05, 0)

    cv2.imwrite(ref_path, ref)
    cv2.imwrite(sec_path, sec)


def ensure_sample_files() -> tuple[str, str]:
    """
    Check for sample files; generate if missing.
    Returns paths to reference and secondary sample images.
    """
    ref_path = "sample_ref.png"
    sec_path = "sample_sec.png"

    if not (os.path.exists(ref_path) and os.path.exists(sec_path)):
        generate_synthetic_lunar_pair(ref_path, sec_path)

    return ref_path, sec_path


def match_pair_hf(
    img1: np.ndarray, img2: np.ndarray, ransac_threshold_px: float = 3.0
) -> tuple[np.ndarray, np.ndarray, str, dict]:
    """Run phase-congruency RIFT2 first and disclose any LightGlue/ALIKED handoff."""
    # SIFT and most deep matcher frontends require finite uint8 image arrays.
    def as_uint8(image):
        image = np.asarray(image)
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
            image = np.asarray(image)
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
        affine_matrix, inliers = _estimate_partial_affine(pts_sec, pts_ref, ransac_threshold_px)
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
                if shadow_mask is not None and len(fallback_ref):
                    keep = outside_shadow(fallback_ref, shadow_mask) & outside_shadow(fallback_sec, shadow_mask_sec)
                    fallback_ref, fallback_sec = fallback_ref[keep], fallback_sec[keep]
                if enforce_uniform_distribution and len(fallback_ref):
                    fallback_ref, fallback_sec, _ = select_distributed_matches(
                        fallback_ref, fallback_sec, ref_original.shape[:2],
                        grid_shape=(8, 8), max_per_bucket=8
                    )
                if len(fallback_ref) >= 3:
                    fallback_affine, fallback_inliers = _estimate_partial_affine(
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

    if affine_matrix is None or inlier_cnt < MIN_REGISTRATION_INLIERS:
        raise ValueError(
            f"Insufficient Inliers: partial affine registration requires at least "
            f"{MIN_REGISTRATION_INLIERS}; found {inlier_cnt}."
        )

    # Refine only geometrically verified matches, then re-estimate the model.
    refinement_stats = {"refined_pairs": 0, "status": "insufficient_verified_matches"}
    inlier_ref, inlier_sec = pts_ref[inliers], pts_sec[inliers]
    try:
        refined_ref, refined_sec, refinement_stats = refine_subpixel_ncc(
            ref_processed, sec_processed, inlier_ref, inlier_sec,
            ncc_window=11, search_range_px=1
        )
        refinement_stats["status"] = "subpixel_pairs_refined" if len(refined_ref) else "no_subpixel_pairs"
        if len(refined_ref) >= 3:
            refined_affine, refined_inliers = _estimate_partial_affine(
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

    H = _homogeneous_affine(affine_matrix)
    warped_sec = cv2.warpAffine(sec_original, affine_matrix, (ref_original.shape[1], ref_original.shape[0]))
    diff_map = cv2.absdiff(ref_original, warped_sec)

    inlier_cnt = len(inlier_ref)
    pts_sec_h = cv2.transform(inlier_sec.reshape(-1, 1, 2), affine_matrix).reshape(-1, 2)
    residuals_vec = inlier_ref - pts_sec_h
    residuals_mag = np.linalg.norm(residuals_vec, axis=1)

    rmse_px = float(np.sqrt(np.mean(residuals_mag ** 2))) if len(residuals_mag) > 0 else 0.0
    mae_px = float(np.mean(residuals_mag)) if len(residuals_mag) > 0 else 0.0

    side_by_side = np.hstack((ref_original, warped_sec, diff_map))
    side_by_side_rgb = cv2.cvtColor(side_by_side, cv2.COLOR_GRAY2RGB)

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
        "side_by_side_rgb": side_by_side_rgb,
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
    Returns (PIL Image, report_text, csv_path, json_path, geotiff_path, png_path, viz_path)
    """
    if ref_file is None or sec_file is None:
        return None, "Error: Please provide both Reference and Secondary surface frames.", None, None, None, None, None

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

        side_by_side_rgb = result["side_by_side_rgb"]
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
        refinement_stats = result["refinement_stats"]
        pixel_scale_m = result["pixel_scale_m"]

        # Build comprehensive report
        inlier_pct = inlier_cnt / max(total_matches, 1) * 100
        rmse_m = rmse_px * pixel_scale_m
        mae_m = mae_px * pixel_scale_m
        
        report = (
            f"✅ REGISTRATION COMPLETE\n"
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
        with transform_json.open("w", encoding="utf-8") as stream:
            json.dump(transform_payload, stream, indent=2)
        
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
        
        # Convert main output to PIL
        output_pil = Image.fromarray(np.asarray(side_by_side_rgb, dtype=np.uint8))
        
        return (
            output_pil,
            report,
            export_paths.get("gcp_csv"),
            export_paths.get("transform_json"),
            export_paths.get("warped_geotiff"),
            export_paths.get("warped_png"),
            str(viz_path)
        )

    except ValueError as e:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return None, f"Registration Failed: {str(e)}", None, None, None, None, None
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return None, f"Registration Failed: {str(e)}", None, None, None, None, None


GPU_EXECUTION_STATUS = "⚡ Execution Mode: ZeroGPU (A10G Accelerated)"
CPU_FALLBACK_STATUS = "💻 Execution Mode: CPU (Fallback Active - Quota/Worker Limit Handled Gracefully)"


def _append_execution_status(outputs, status: str):
    """Append execution mode to the UI report without changing output arity."""
    values = list(outputs) if isinstance(outputs, (tuple, list)) else [None] * 7
    if len(values) != 7:
        values = (values + [None] * 7)[:7]
    report = values[1] if isinstance(values[1], str) else ""
    values[1] = f"{report.rstrip()}\n\n{status}".strip()
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
        outputs = run_alignment_on_gpu(
            ref, sec, sensor, secondary_sensor, pair_mode, enforce_uniform,
            px_scale, clahe, clip, shadow, wallis, wallis_m, wallis_s
        )
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
            outputs = (None, f"Registration Failed: {cpu_error}", None, None, None, None, None)
        return _append_execution_status(outputs, CPU_FALLBACK_STATUS)
    status = GPU_EXECUTION_STATUS if _zerogpu_runtime_enabled() else "💻 Execution Mode: CPU (Local Runtime)"
    return _append_execution_status(outputs, status)


# Build Gradio interface with advanced controls
def build_interface():
    """Build the Gradio interface with scientific controls."""
    
    with gr.Blocks(title="CHANDRA-ALIGN: Lunar Photogrammetric Workstation") as interface:
        gr.Markdown(
            "# 🌙 CHANDRA-ALIGN: Lunar Cross-Sensor Photogrammetric Workstation\n"
            "Sub-pixel registration for Chandrayaan-2 OHRC, TMC-2, IIRS, DF-SAR & LRO NAC imagery. "
            "Achieves scientific-grade photogrammetric accuracy with ground-resolution error translation."
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input panel
                gr.Markdown("### 📥 Input Frames")
                ref_file = gr.File(
                    label="Reference Frame (NASA LROC NAC)",
                    file_types=[".png", ".tif", ".tiff", ".jpg", ".jpeg"]
                )
                sec_file = gr.File(
                    label="Secondary Frame (ISRO Payload)",
                    file_types=[".png", ".tif", ".tiff", ".jpg", ".jpeg"]
                )
                
                # Advanced Controls
                gr.Markdown("### ⚙️ Advanced Photogrammetric Controls")
                
                with gr.Accordion("Sensor Pair & Resolution", open=True):
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
                
                with gr.Accordion("Illumination-Invariant Preprocessing", open=True):
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
            
            with gr.Column(scale=2):
                # Output panel
                gr.Markdown("### 📊 Registration Results")
                
                with gr.Tabs():
                    with gr.TabItem("Side-by-Side View"):
                        output_image = gr.Image(
                            label="Registration View [Reference | Aligned Secondary | Radiometric Delta]",
                            type="pil",
                            format="png"
                        )
                    
                    with gr.TabItem("Scientific Dossier (4-Panel)"):
                        dossier_image = gr.Image(
                            label="Photogrammetric Verification Dossier",
                            type="filepath"
                        )
                
                report_text = gr.Textbox(
                    label="Photogrammetric Summary Report",
                    lines=20,
                    max_lines=30
                )
                
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
                output_image, report_text,
                csv_btn, json_btn, geotiff_btn, png_btn, dossier_image
            ]
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
            ("nac_reference_ohrc.png", "ohrc_secondary.png", "OHRC",
             "Optical <-> Optical", "OHRC vs NASA LROC NAC · 2× GSD gap"),
            ("nac_reference_tmc2.png", "tmc2_secondary.png", "TMC-2",
             "Optical <-> Optical", "TMC-2 vs NASA LROC NAC · 10× GSD gap"),
            ("nac_reference_iirs.png", "iirs_band125_secondary.png", "IIRS",
             "Optical <-> Infrared", "IIRS Band 125 (2.802 µm) vs NASA LROC NAC · 160× GSD gap"),
        )
        example_rows = []
        example_labels = []
        for reference_filename, secondary_filename, secondary_sensor, pair_mode, label in example_specs:
            reference_path = examples_dir / reference_filename
            secondary_path = examples_dir / secondary_filename
            for sample_path in (reference_path, secondary_path):
                if not sample_path.is_file():
                    raise FileNotFoundError(f"Tracked Gradio example asset is missing: {sample_path}")
            example_rows.append([
                str(reference_path), str(secondary_path), "LROC_NAC", secondary_sensor,
                pair_mode, False, 0.5, True, 3.0, False, False, 128.0, 50.0
            ])
            example_labels.append(label)
        gr.Examples(
            examples=example_rows,
            inputs=[
                ref_file, sec_file, sensor_dropdown, secondary_sensor_dropdown,
                sensor_pair_mode, enforce_uniformity, pixel_scale,
                enable_clahe, clahe_clip, enable_shadow, enable_wallis,
                wallis_mean, wallis_std
            ],
            outputs=[
                output_image, report_text,
                csv_btn, json_btn, geotiff_btn, png_btn, dossier_image
            ],
            fn=process_wrapper,
            cache_examples=False,
            examples_per_page=3,
            run_on_click=True,
            example_labels=example_labels,
            label="One-click lunar alignment presets"
        )

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
interface = build_interface()


if __name__ == "__main__":
    interface.launch()
