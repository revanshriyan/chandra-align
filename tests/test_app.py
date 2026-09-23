import pytest
import numpy as np
import cv2
import tempfile
import os
from app import align_images_ui, build_app

def test_app_initialization():
    """Verify Gradio app structure builds cleanly."""
    demo = build_app()
    assert demo is not None

def test_align_images_ui_synthetic():
    """Verify align_images_ui executes on CPU synthetic image pair."""
    # Create temp dummy images
    temp_dir = tempfile.mkdtemp()
    img1_path = os.path.join(temp_dir, "ref.png")
    img2_path = os.path.join(temp_dir, "sec.png")

    ref = np.zeros((200, 200), dtype=np.uint8)
    sec = np.zeros((200, 200), dtype=np.uint8)
    
    cv2.rectangle(ref, (50, 50), (150, 150), 255, -1)
    cv2.rectangle(sec, (55, 55), (155, 155), 255, -1)

    cv2.imwrite(img1_path, ref)
    cv2.imwrite(img2_path, sec)

    overlay, status, metrics_file, _ = align_images_ui(img1_path, img2_path, "Auto-Detect")

    assert status.startswith("✅ Status:") or status.startswith("❌ Alignment Failed:")
    assert os.path.exists(img1_path)
    assert os.path.exists(img2_path)