"""
CHANDRA-ALIGN: Lunar Cross-Sensor Registration Engine
Gradio web application for Hugging Face Spaces deployment.
"""

import os
import sys
import cv2
import numpy as np
import gradio as gr
import spaces
import tempfile
import shutil


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


# Persistent output directory for Gradio to serve files
OUTPUT_DIR = "/tmp/chandra_align_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@spaces.GPU(duration=60)
def process_alignment(ref_file, sec_file):
    """
    Main alignment pipeline - decorated with @spaces.GPU for ZeroGPU execution.
    Returns (file_path, report_text) - file path accessible from main process.
    """
    if ref_file is None or sec_file is None:
        return None, "Error: Please provide both Reference and Secondary surface frames."

    ref_img = cv2.imread(ref_file.name if hasattr(ref_file, 'name') else ref_file, cv2.IMREAD_GRAYSCALE)
    sec_img = cv2.imread(sec_file.name if hasattr(sec_file, 'name') else sec_file, cv2.IMREAD_GRAYSCALE)

    if ref_img is None or sec_img is None:
        return None, "Error: Unable to decode input image files."

    pts_ref, pts_sec, engine_name = match_pair_hf(ref_img, sec_img)

    if len(pts_ref) < 4:
        return None, "Registration Failed: Insufficient keypoint correspondences detected."

    H, mask = cv2.findHomography(pts_sec, pts_ref, cv2.RANSAC, 3.0)
    if H is None:
        return None, "Registration Failed: Homography calculation failed."

    inliers = mask.squeeze().astype(bool) if mask is not None else np.zeros(len(pts_ref), dtype=bool)
    inlier_cnt = int(inliers.sum())

    warped_sec = cv2.warpPerspective(sec_img, H, (ref_img.shape[1], ref_img.shape[0]))
    diff_map = cv2.absdiff(ref_img, warped_sec)

    pts_sec_h = cv2.perspectiveTransform(pts_sec[inliers].reshape(-1, 1, 2), H).squeeze()
    residuals = np.linalg.norm(pts_ref[inliers] - pts_sec_h, axis=1) if inlier_cnt > 0 else np.array([0])

    rmse = float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) > 0 else 0.0
    mae = float(np.mean(residuals)) if len(residuals) > 0 else 0.0

    side_by_side = np.hstack((ref_img, warped_sec, diff_map))
    side_by_side_rgb = cv2.cvtColor(side_by_side, cv2.COLOR_GRAY2RGB)

    # Save to persistent output directory with unique name
    import uuid
    filename = f"result_{uuid.uuid4().hex[:8]}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, cv2.cvtColor(side_by_side_rgb, cv2.COLOR_RGB2BGR))

    report = (
        f"✅ REGISTRATION COMPLETE\n"
        f"Matcher Engine: {engine_name}\n"
        f"Verified Inliers: {inlier_cnt} / {len(pts_ref)} ({inlier_cnt/max(len(pts_ref),1)*100:.1f}%)\n"
        f"RMSE Accuracy: {rmse:.4f} px\n"
        f"MAE Error: {mae:.4f} px"
    )

    return output_path, report


# Build interface with lazy sample loading
def get_examples():
    """Lazy loading of sample files for Gradio examples."""
    return ensure_sample_files()


interface = gr.Interface(
    fn=process_alignment,
    inputs=[
        gr.File(label="Reference Frame (OHRC / Baseline)", file_types=[".png", ".tif", ".tiff", ".jpg", ".jpeg"]),
        gr.File(label="Secondary Frame (LRO NAC / Target)", file_types=[".png", ".tif", ".tiff", ".jpg", ".jpeg"])
    ],
    outputs=[
        gr.Image(label="Registration View [Reference | Aligned Secondary | Radiometric Delta]", type="numpy"),
        gr.Textbox(label="Photogrammetric Summary Report", lines=10)
    ],
    title="CHANDRA-ALIGN: Lunar Cross-Sensor Registration Engine",
    description=(
        "Sub-pixel photogrammetric registration for Chandrayaan-2 OHRC and LRO NAC "
        "lunar surface imagery. Upload reference and secondary frames to compute "
        "cross-sensor alignment with robust outlier rejection and residual analysis."
    ),
    examples=[],  # Empty initially, samples generated on first request
    cache_examples=False
)


if __name__ == "__main__":
    # Generate samples on startup
    sample_ref, sample_sec = ensure_sample_files()
    # Update examples with generated samples
    interface.examples = [[sample_ref, sample_sec]]
    interface.launch()