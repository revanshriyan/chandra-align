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


MAX_IMAGE_DIMENSION = 4096


def _read_grayscale_image(image_file) -> np.ndarray:
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
    scale = min(1.0, MAX_IMAGE_DIMENSION / max(height, width))
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


def match_pair_hf(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Match features between two images with fallback chain.
    Returns (src_pts, dst_pts, matcher_name).
    """
    # SIFT and most deep matcher frontends require finite uint8 image arrays.
    def as_uint8(image):
        image = np.asarray(image)
        if image.ndim != 2 or image.size == 0:
            raise ValueError("Feature matching requires non-empty grayscale images")
        image = np.nan_to_num(image.astype(np.float32, copy=False))
        if image.dtype == np.uint8:
            return np.ascontiguousarray(image)
        lo, hi = np.percentile(image, [1, 99])
        return np.zeros(image.shape, np.uint8) if hi <= lo else np.clip((image - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

    img1, img2 = as_uint8(img1), as_uint8(img2)
    # Try DeepMatcherChain first
    try:
        from chandra_align.matching.deep_matchers import ClassicalSIFTMatcher, DeepMatcherChain
        matcher = DeepMatcherChain(matchers=[ClassicalSIFTMatcher()])
        res = matcher.match(img1, img2)
        if len(res.pts_src) >= 4:
            return res.pts_src, res.pts_ref, f"DeepMatcherChain[{res.matcher_name}]"
    except Exception:
        pass

    # Fallback to classical SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return np.empty((0, 2)), np.empty((0, 2)), "Failed"

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 4:
        return np.empty((0, 2)), np.empty((0, 2)), "ClassicalSIFT"

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

    return src_pts, dst_pts, "ClassicalSIFT"


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
    reference_sensor_name: str = "OHRC"
) -> dict:
    """
    Core alignment logic - runs on GPU when called from process_alignment.
    Returns comprehensive results dictionary.
    """
    # Store original images for visualization
    ref_original = _read_grayscale_image(ref_img)
    sec_original = _read_grayscale_image(sec_img)
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

    if sensor_pair_mode == "Optical <-> Infrared":
        if str(reference_sensor_name).upper() == "IIRS":
            ref_processed = preprocess_iirs_raster(ref_processed).processed_2d_raster
        if str(secondary_sensor_name).upper() == "IIRS":
            sec_processed = preprocess_iirs_raster(sec_processed).processed_2d_raster

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
        match_ref, ref_match_scale = resize_to_common_ground_sample(ref_processed, ref_gsd_m, target_gsd_m)
        match_sec, sec_match_scale = resize_to_common_ground_sample(sec_processed, sec_gsd_m, target_gsd_m)
    except ValueError:
        match_ref, match_sec = ref_processed, sec_processed
        ref_match_scale = sec_match_scale = 1.0
    pts_ref, pts_sec, engine_name = match_pair_hf(match_ref, match_sec)
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
    
    if len(pts_ref) < 4:
        raise ValueError("Insufficient keypoint correspondences detected.")

    try:
        H, mask = cv2.findHomography(pts_sec, pts_ref, cv2.RANSAC, 3.0)
    except cv2.error as exc:
        raise ValueError(f"Homography estimation failed for the detected correspondences: {exc}") from exc
    if H is None:
        raise ValueError("Homography calculation failed; matches may be degenerate or collinear.")

    inliers = mask.reshape(-1).astype(bool) if mask is not None else np.zeros(len(pts_ref), dtype=bool)
    inlier_cnt = int(inliers.sum())
    if inlier_cnt < 4:
        raise ValueError("Registration failed: fewer than four geometrically consistent inliers were found.")

    # Refine only geometrically verified matches, then re-estimate the model.
    refinement_stats = {"refined_pairs": 0, "status": "insufficient_verified_matches"}
    inlier_ref, inlier_sec = pts_ref[inliers], pts_sec[inliers]
    try:
        refined_ref, refined_sec, refinement_stats = refine_subpixel_ncc(
            ref_processed, sec_processed, inlier_ref, inlier_sec,
            ncc_window=11, search_range_px=1
        )
        refinement_stats["status"] = "subpixel_pairs_refined" if len(refined_ref) else "no_subpixel_pairs"
        if len(refined_ref) >= 4:
            refined_H, refined_mask = cv2.findHomography(refined_sec, refined_ref, cv2.RANSAC, 3.0)
            if refined_H is not None and refined_mask is not None:
                refined_inliers = refined_mask.reshape(-1).astype(bool)
                if int(refined_inliers.sum()) >= 4:
                    H = refined_H
                    inlier_ref = refined_ref[refined_inliers]
                    inlier_sec = refined_sec[refined_inliers]
                    refinement_stats["status"] = "subpixel_model_reestimated"
    except (cv2.error, ValueError, FloatingPointError) as exc:
        refinement_stats = {"refined_pairs": 0, "status": f"skipped: {exc}"}

    warped_sec = cv2.warpPerspective(sec_original, H, (ref_original.shape[1], ref_original.shape[0]))
    diff_map = cv2.absdiff(ref_original, warped_sec)

    inlier_cnt = len(inlier_ref)
    pts_sec_h = cv2.perspectiveTransform(inlier_sec.reshape(-1, 1, 2), H).reshape(-1, 2)
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
        "inlier_cnt": inlier_cnt,
        "total_matches": len(pts_ref),
        "rmse_px": rmse_px,
        "mae_px": mae_px,
        "H": H,
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
        "pixel_scale_m": pixel_scale_m
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
        ref_img = _read_grayscale_image(ref_file)
        sec_img = _read_grayscale_image(sec_file)
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
            reference_sensor_name=sensor_name
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
                    label="Reference Frame (OHRC / Baseline)",
                    file_types=[".png", ".tif", ".tiff", ".jpg", ".jpeg"]
                )
                sec_file = gr.File(
                    label="Secondary Frame (LRO NAC / TMC-2 / Target)",
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
                            type="pil"
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
        sample_ref, sample_sec = ensure_sample_files()
        gr.Examples(
            examples=[[
                sample_ref, sample_sec, "OHRC", "TMC-2", "Optical <-> Optical",
                True, 0.25, True, 3.0, True, False, 128.0, 50.0
            ]],
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
            run_on_click=True,
            example_labels=["Run synthetic OHRC / TMC-2 demo"],
            label="One-click synthetic lunar alignment"
        )
    
    return interface


# Build and export
interface = build_interface()


if __name__ == "__main__":
    interface.launch()
