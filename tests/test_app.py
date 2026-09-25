import pytest
import numpy as np
import cv2
import tempfile
import os
from app import process_alignment, ensure_sample_files, interface


def test_app_initialization():
    """Verify Gradio app structure builds cleanly."""
    assert interface is not None


def test_ensure_sample_files():
    """Verify sample files are generated on startup."""
    sample_ref, sample_sec = ensure_sample_files()
    assert os.path.exists(sample_ref)
    assert os.path.exists(sample_sec)
    # Should be valid images
    ref = cv2.imread(sample_ref, cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(sample_sec, cv2.IMREAD_GRAYSCALE)
    assert ref is not None and sec is not None
    assert ref.shape == (512, 512)
    assert sec.shape == (512, 512)


def test_process_alignment_synthetic():
    """Verify process_alignment executes on CPU synthetic image pair."""
    temp_dir = tempfile.mkdtemp()
    img1_path = os.path.join(temp_dir, "ref.png")
    img2_path = os.path.join(temp_dir, "sec.png")

    # Create synthetic images with sufficient distinctive features for SIFT
    ref = np.zeros((300, 300), dtype=np.uint8)
    sec = np.zeros((300, 300), dtype=np.uint8)
    for i in range(5):
        for j in range(5):
            cv2.circle(ref, (50 + i * 50, 50 + j * 50), 15, 255, -1)
            cv2.circle(sec, (55 + i * 50, 55 + j * 50), 15, 255, -1)

    cv2.imwrite(img1_path, ref)
    cv2.imwrite(img2_path, sec)

    class MockFile:
        def __init__(self, path):
            self.name = path

    result_img, status = process_alignment(MockFile(img1_path), MockFile(img2_path))

    assert result_img is not None
    assert "REGISTRATION COMPLETE" in status
    assert "RMSE Accuracy" in status
    assert "MAE Error" in status
    assert isinstance(result_img, dict)
    assert "path" in result_img
    assert os.path.exists(result_img["path"])

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_process_alignment_errors():
    """Verify error handling for missing files and insufficient features."""
    temp_dir = tempfile.mkdtemp()
    blank_path = os.path.join(temp_dir, "blank.png")
    cv2.imwrite(blank_path, np.zeros((100, 100), dtype=np.uint8))

    class MockFile:
        def __init__(self, path):
            self.name = path

    # Missing file
    result_img, status = process_alignment(None, MockFile(blank_path))
    assert result_img is None
    assert "Error" in status

    # Insufficient features
    result_img, status = process_alignment(MockFile(blank_path), MockFile(blank_path))
    assert result_img is None
    assert "Registration Failed" in status

    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)