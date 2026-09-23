import os
import json
import tempfile
import numpy as np
import cv2
import gradio as gr

from chandra_align.matching.deep_matchers import ClassicalSIFTMatcher, DeepMatcherChain
from chandra_align.matching.sar_optical import SAROpticalGradientMatcher

def align_images_ui(ref_img_path: str, sec_img_path: str, mode: str):
    """Gradio handler for browser-based 2D alignment execution."""
    if not ref_img_path or not sec_img_path:
        return None, "Error: Both Reference and Secondary images are required.", None, None

    # Load images
    ref = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(sec_img_path, cv2.IMREAD_GRAYSCALE)

    if ref is None or sec is None:
        return None, "Error: Unable to load input image files.", None, None

    # Select Matcher
    if mode == "DF-SAR Cross-Modal":
        matcher = SAROpticalGradientMatcher()
        res = matcher.match(sec, ref)
    else:
        # CPU-safe SIFT chain
        chain = DeepMatcherChain(matchers=[ClassicalSIFTMatcher()])
        res = chain.match(sec, ref)

    if res.status == "FAILED" or len(res.pts_src) < 4:
        status_msg = f"❌ Alignment Failed: {res.error_msg or 'Insufficient matching keypoints'}"
        return None, status_msg, None, None

    # Estimate Homography matrix
    H, inlier_mask = cv2.findHomography(res.pts_src, res.pts_ref, cv2.RANSAC, 3.0)
    inlier_count = int(np.sum(inlier_mask)) if inlier_mask is not None else 0

    # Calculate dummy RMSE for visualization
    if H is not None and inlier_mask is not None:
        inlier_src = res.pts_src[inlier_mask.ravel() == 1]
        inlier_ref = res.pts_ref[inlier_mask.ravel() == 1]
        
        # Transform source points using H
        pts_src_hom = np.hstack([inlier_src, np.ones((len(inlier_src), 1))])
        pts_proj = (H @ pts_src_hom.T).T
        pts_proj = pts_proj[:, :2] / pts_proj[:, 2:]
        
        rmse = float(np.sqrt(np.mean(np.sum((pts_proj - inlier_ref) ** 2, axis=1))))
        aligned_sec = cv2.warpPerspective(sec, H, (ref.shape[1], ref.shape[0]))
    else:
        rmse = 999.0
        aligned_sec = sec

    # Create difference overlay map
    diff_map = cv2.absdiff(ref, aligned_sec)
    diff_colored = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)

    metrics = {
        "status": "TRUSTED" if rmse < 1.5 and inlier_count >= 15 else "UNTRUSTED",
        "inlier_count": inlier_count,
        "rmse_pixels": round(rmse, 3),
        "matcher_used": res.matcher_name,
        "execution_time_sec": round(res.execution_time_sec, 3)
    }

    # Save metrics JSON to temp file
    temp_dir = tempfile.mkdtemp()
    metrics_path = os.path.join(temp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    status_str = f"✅ Status: {metrics['status']} | Inliers: {inlier_count} | RMSE: {rmse:.3f}px | Time: {res.execution_time_sec:.2f}s"

    return diff_colored, status_str, metrics_path, metrics_path

def build_app():
    with gr.Blocks(title="chandra_align — Lunar Image Alignment System") as demo:
        gr.Markdown("# 🌖 chandra_align: Sub-Pixel Lunar Surface Registration Prototype")
        gr.Markdown("Upload Chandrayaan-2 / LROC image pairs to perform automated 2D alignment and residual verification.")

        with gr.Row():
            with gr.Column():
                ref_input = gr.Image(type="filepath", label="Reference Image (Optical / OHRC)")
                sec_input = gr.Image(type="filepath", label="Secondary Image (SAR / SWIR / TMC)")
                mode_select = gr.Dropdown(
                    choices=["Auto-Detect", "Optical (OHRC/TMC)", "DF-SAR Cross-Modal"],
                    value="Auto-Detect",
                    label="Alignment Pipeline Route"
                )
                align_btn = gr.Button("Run Alignment", variant="primary")

            with gr.Column():
                output_overlay = gr.Image(label="Alignment Residual Map (Diff Map)")
                status_box = gr.Textbox(label="Execution Summary", interactive=False)
                metrics_file = gr.File(label="Download metrics.json")

        align_btn.click(
            fn=align_images_ui,
            inputs=[ref_input, sec_input, mode_select],
            outputs=[output_overlay, status_box, metrics_file]
        )

    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)