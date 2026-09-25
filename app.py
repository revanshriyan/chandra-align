"""
CHANDRA-ALIGN: Lunar Cross-Sensor Registration Engine
Gradio web application for Hugging Face Spaces deployment.
"""

import os
import sys
import cv2
import numpy as np
import gradio as gr


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


def process_alignment(ref_file, sec_file):
    """
    Main alignment pipeline for Gradio interface.
    Returns (visualization_image, report_text).
    """
    if ref_file is None or sec_file is None:
        return None, "Error: Please upload both Reference Frame and Secondary Frame."

    ref_img = cv2.imread(ref_file.name, cv2.IMREAD_GRAYSCALE)
    sec_img = cv2.imread(sec_file.name, cv2.IMREAD_GRAYSCALE)

    if ref_img is None or sec_img is None:
        return None, "Error: Failed to decode input images. Ensure valid image format."

    # Feature matching with fallback
    src_pts, dst_pts, matcher_name = match_pair_hf(ref_img, sec_img)

    if len(src_pts) < 4:
        return None, "Registration Failed: Less than 4 feature correspondences detected."

    # Robust homography estimation via RANSAC
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0)

    if H is None:
        return None, "Registration Failed: Homography estimation failed."

    # Inlier statistics
    inliers = mask.squeeze().astype(bool) if mask is not None else np.zeros(len(src_pts), dtype=bool)
    inlier_count = int(inliers.sum())
    total_matches = len(src_pts)
    inlier_ratio = inlier_count / total_matches if total_matches > 0 else 0.0

    # Warp secondary onto reference frame
    warped_sec = cv2.warpPerspective(sec_img, H, (ref_img.shape[1], ref_img.shape[0]))

    # Radiometric difference map
    diff_map = cv2.absdiff(ref_img, warped_sec)
    diff_heatmap = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)

    # Residual metrics on inliers
    if inlier_count > 0:
        src_inliers = src_pts[inliers]  # Shape: (N, 2)
        dst_inliers = dst_pts[inliers].reshape(-1, 1, 2)

        # Project source inliers through homography
        projected = cv2.perspectiveTransform(dst_inliers, H).squeeze()  # Shape: (N, 2)
        residuals = np.linalg.norm(src_inliers - projected, axis=1)

        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mae = float(np.mean(residuals))
    else:
        rmse = 0.0
        mae = 0.0

    # Compose side-by-side visualization: Reference | Aligned Secondary | Delta Heatmap
    ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB)
    warped_rgb = cv2.cvtColor(warped_sec, cv2.COLOR_GRAY2RGB)
    side_by_side = np.hstack((ref_rgb, warped_rgb, diff_heatmap))

    # Executive report
    report = (
        f"CHANDRA-ALIGN Registration Report\n"
        f"{'=' * 45}\n"
        f"Matcher Engine: {matcher_name}\n"
        f"Inlier Count: {inlier_count} / {total_matches} ({inlier_ratio * 100:.1f}%)\n"
        f"Registration Precision (RMSE): {rmse:.4f} px\n"
        f"Mean Absolute Error (MAE): {mae:.4f} px\n"
    )

    return side_by_side, report


def build_interface() -> gr.Interface:
    """Construct and return the Gradio Interface."""
    # Ensure sample files exist on startup
    sample_ref, sample_sec = ensure_sample_files()

    title = "CHANDRA-ALIGN: Lunar Cross-Sensor Registration Engine"
    description = (
        "Sub-pixel photogrammetric registration for Chandrayaan-2 OHRC and LRO NAC "
        "lunar surface imagery. Upload reference and secondary frames to compute "
        "cross-sensor alignment with robust outlier rejection and residual analysis."
    )

    interface = gr.Interface(
        fn=process_alignment,
        inputs=[
            gr.File(label="Reference Frame (OHRC / Baseline)", file_types=[".png", ".tif", ".tiff", ".jpg", ".jpeg"]),
            gr.File(label="Secondary Frame (LRO NAC / Target)", file_types=[".png", ".tif", ".tiff", ".jpg", ".jpeg"])
        ],
        outputs=[
            gr.Image(label="Registration View [Reference | Aligned Secondary | Radiometric Delta]"),
            gr.Textbox(label="Photogrammetric Summary Report", lines=10)
        ],
        title=title,
        description=description,
        examples=[[sample_ref, sample_sec]],
        cache_examples=False
    )

    return interface


if __name__ == "__main__":
    app = build_interface()
    app.launch()