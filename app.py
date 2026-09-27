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
from pathlib import Path
from PIL import Image

# Import new modules
from chandra_align.preprocessing import (
    apply_clahe, detect_shadows, apply_wallis_filter, preprocess_pipeline, suppress_keypoints_in_shadows
)
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
    for m, n in matches:
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
    wallis_target_std: float = 50.0
) -> dict:
    """
    Core alignment logic - runs on GPU when called from process_alignment.
    Returns comprehensive results dictionary.
    """
    # Store original images for visualization
    ref_original = ref_img.copy()
    sec_original = sec_img.copy()
    
    # Apply preprocessing
    ref_processed = ref_img.copy()
    sec_processed = sec_img.copy()
    
    if enable_clahe:
        ref_processed = apply_clahe(ref_processed, clip_limit=clahe_clip_limit, normalize_range=(0.0, 255.0))
        sec_processed = apply_clahe(sec_processed, clip_limit=clahe_clip_limit, normalize_range=(0.0, 255.0))
    
    shadow_mask = None
    if enable_shadow_suppression:
        shadow_mask = detect_shadows(ref_processed, method="otsu")
        # We'll apply shadow suppression during keypoint filtering
    
    if enable_wallis:
        ref_processed = apply_wallis_filter(ref_processed, target_mean=wallis_target_mean, target_std=wallis_target_std)
        sec_processed = apply_wallis_filter(sec_processed, target_mean=wallis_target_mean, target_std=wallis_target_std)
    
    # Feature matching on processed images
    pts_ref, pts_sec, engine_name = match_pair_hf(ref_processed, sec_processed)
    
    # Apply shadow suppression if enabled
    if shadow_mask is not None:
        # Convert points to keypoints for filtering
        kp_ref = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1.0) for pt in pts_ref]
        kp_sec = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1.0) for pt in pts_sec]
        
        # Filter keypoints in shadow regions
        kp_ref_filtered = suppress_keypoints_in_shadows(kp_ref, shadow_mask)
        kp_sec_filtered = suppress_keypoints_in_shadows(kp_sec, shadow_mask)
        
        # Rebuild point arrays
        pts_ref = np.float32([kp.pt for kp in kp_ref_filtered])
        pts_sec = np.float32([kp.pt for kp in kp_sec_filtered])
    
    if len(pts_ref) < 4:
        raise ValueError("Insufficient keypoint correspondences detected.")

    H, mask = cv2.findHomography(pts_sec, pts_ref, cv2.RANSAC, 3.0)
    if H is None:
        raise ValueError("Homography calculation failed.")

    inliers = mask.squeeze().astype(bool) if mask is not None else np.zeros(len(pts_ref), dtype=bool)
    inlier_cnt = int(inliers.sum())

    warped_sec = cv2.warpPerspective(sec_original, H, (ref_original.shape[1], ref_original.shape[0]))
    diff_map = cv2.absdiff(ref_original, warped_sec)

    pts_sec_h = cv2.perspectiveTransform(pts_sec[inliers].reshape(-1, 1, 2), H).squeeze()
    residuals_vec = pts_ref[inliers] - pts_sec_h
    residuals_mag = np.linalg.norm(residuals_vec, axis=1) if inlier_cnt > 0 else np.array([0])

    rmse_px = float(np.sqrt(np.mean(residuals_mag ** 2))) if len(residuals_mag) > 0 else 0.0
    mae_px = float(np.mean(residuals_mag)) if len(residuals_mag) > 0 else 0.0

    side_by_side = np.hstack((ref_original, warped_sec, diff_map))
    side_by_side_rgb = cv2.cvtColor(side_by_side, cv2.COLOR_GRAY2RGB)

    # Compute deformation field
    deformation_vectors = compute_deformation_field(
        pts_ref[inliers], pts_sec[inliers], H, pixel_scale_m
    )
    
    # Grid deformation analysis
    grid_analysis = grid_deformation_analysis(deformation_vectors, (8, 8), ref_original.shape[:2], pixel_scale_m)
    
    # Ground metrics
    ground_metrics = compute_ground_metrics(residuals_mag, pixel_scale_m)
    
    # Spatial entropy (uniformity)
    from chandra_align.evaluators.quadtree import evaluate_quadtree_uniformity
    uniformity = 0.0
    if inlier_cnt > 0:
        quadtree_result = evaluate_quadtree_uniformity(pts_ref[inliers], ref_original.shape[:2], depth=4)
        uniformity = quadtree_result.uniformity_score

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
        "pts_ref_inliers": pts_ref[inliers],
        "pts_sec_inliers": pts_sec[inliers],
        "deformation_vectors": deformation_vectors,
        "grid_analysis": grid_analysis,
        "ground_metrics": ground_metrics,
        "uniformity": uniformity,
        "shadow_mask": shadow_mask,
        "pixel_scale_m": pixel_scale_m
    }


@spaces.GPU(duration=120)
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
    sensor_name: str = "OHRC"
):
    """
    Main alignment pipeline - runs on GPU when called from process_wrapper.
    Returns (PIL Image, report_text, csv_path, json_path, geotiff_path, png_path, viz_path)
    """
    if ref_file is None or sec_file is None:
        return None, "Error: Please provide both Reference and Secondary surface frames.", None, None, None, None, None

    ref_img = cv2.imread(ref_file.name if hasattr(ref_file, 'name') else ref_file, cv2.IMREAD_GRAYSCALE)
    sec_img = cv2.imread(sec_file.name if hasattr(sec_file, 'name') else sec_file, cv2.IMREAD_GRAYSCALE)

    if ref_img is None or sec_img is None:
        return None, "Error: Unable to decode input image files.", None, None, None, None, None

    try:
        result = _align_core(
            ref_img, sec_img,
            pixel_scale_m=pixel_scale_m,
            enable_clahe=enable_clahe,
            clahe_clip_limit=clahe_clip_limit,
            enable_shadow_suppression=enable_shadow_suppression,
            enable_wallis=enable_wallis,
            wallis_target_mean=wallis_target_mean,
            wallis_target_std=wallis_target_std
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
        pixel_scale_m = result["pixel_scale_m"]

        # Build comprehensive report
        inlier_pct = inlier_cnt / max(total_matches, 1) * 100
        rmse_m = rmse_px * pixel_scale_m
        mae_m = mae_px * pixel_scale_m
        
        report = (
            f"✅ REGISTRATION COMPLETE\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Matcher Engine: {engine_name}\n"
            f"Verified Inliers: {inlier_cnt} / {total_matches} ({inlier_pct:.1f}%)\n"
            f"Spatial Uniformity: {uniformity:.4f}\n"
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
                "std_px": float(np.std(residuals_mag)) if len(deformation_vectors) > 0 else 0.0,
                "rmse_m": rmse_m,
                "mae_m": mae_m,
                "std_m": ground_metrics.std_m
            },
            str(output_dir),
            base_name,
            pixel_scale_m=pixel_scale_m,
            sensor_name=sensor_name,
            spatial_entropy=uniformity,
            inlier_count=inlier_cnt,
            total_matches=total_matches,
            matcher_name=engine_name,
            trust_flag="TRUSTED" if inlier_cnt > 50 and rmse_px < 1.0 else "UNTRUSTED"
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
        output_pil = Image.fromarray(side_by_side_rgb.astype(np.uint8))
        
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
        return None, f"Registration Failed: {str(e)}", None, None, None, None, None
    except Exception as e:
        return None, f"Registration Failed: {str(e)}", None, None, None, None, None


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
                
                with gr.Accordion("Sensor & Resolution", open=True):
                    sensor_dropdown = gr.Dropdown(
                        choices=["OHRC", "TMC-2", "IIRS", "DF-SAR", "LROC_NAC", "LROC_WAC", "KAGUYA_TC", "Custom"],
                        value="OHRC",
                        label="Sensor Type"
                    )
                    pixel_scale = gr.Number(
                        value=0.25,
                        label="Pixel Scale (m/px)",
                        minimum=0.1,
                        maximum=5.0,
                        step=0.01,
                        info="Ground resolution in meters per pixel"
                    )
                
                with gr.Accordion("Illumination-Invariant Preprocessing", open=True):
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
        @spaces.GPU(duration=120)
        def process_wrapper(ref, sec, sensor, px_scale, clahe, clip, shadow, wallis, wallis_m, wallis_s):
            return process_alignment(
                ref, sec,
                pixel_scale_m=px_scale,
                enable_clahe=clahe,
                clahe_clip_limit=clip,
                enable_shadow_suppression=shadow,
                enable_wallis=wallis,
                wallis_target_mean=wallis_m,
                wallis_target_std=wallis_s,
                sensor_name=sensor
            )
        
        process_btn.click(
            fn=process_wrapper,
            inputs=[
                ref_file, sec_file, sensor_dropdown, pixel_scale,
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
            examples=[[sample_ref, sample_sec]],
            inputs=[ref_file, sec_file],
            label="Click to load synthetic lunar pair"
        )
    
    return interface


# Build and export
interface = build_interface()


if __name__ == "__main__":
    interface.launch()