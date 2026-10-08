"""Prepare CrossFeat training data from IIRS breadth windows.

For each of the 6 windows (W0-W5):
  1. Load IIRS VNIR composite and TMC-2 crop (already at common GSD)
  2. Run pipeline to get affine transform
  3. Warp IIRS to TMC-2 frame
  4. Save as PNG pair in CrossFeat folder structure

Split: train=[W0,W1,W2,W3], val=[W4], test=[W5]
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/tmp/stubs")
sys.path.insert(0, "/home/hatch/workspace/chandra-align")

# Stub visualization before app import
import types
viz = types.ModuleType("chandra_align.visualization")
viz.draw_error_vector_overlay = lambda *a, **k: np.zeros((10, 10, 3), dtype=np.uint8)
viz.create_combined_visualization = lambda *a, **k: np.zeros((10, 10, 3), dtype=np.uint8)
viz.create_checkerboard_overlay = lambda *a, **k: np.zeros((10, 10, 3), dtype=np.uint8)
viz.create_interactive_blend = lambda *a, **k: None
viz.create_rejection_banner = lambda *a, **k: np.zeros((10, 10, 3), dtype=np.uint8)
viz.create_warped_preview = lambda *a, **k: np.zeros((10, 10), dtype=np.uint8)
viz.fig_to_file = lambda *a, **k: None
sys.modules["chandra_align.visualization"] = viz

# Import breadth helpers
sys.path.insert(0, "/home/hatch/workspace/chandra-align/scripts")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "iirs_breadth", "/home/hatch/workspace/chandra-align/scripts/iirs_breadth.py")
ib = importlib.util.module_from_spec(spec)

# We can't import the whole module (it runs main), so copy the needed functions
# Instead, we'll replicate the window loading here

QA = "/home/hatch/workspace/qa_lunar"
QUB = (QA + "/pradan_iirs2/cube/data/calibrated/20231225/"
       "ch2_iir_nci_20231225T1904122779_d_img_d18.qub")
IIRS_GEO = (QA + "/pradan_iirs2/cube/geometry/calibrated/20231225/"
            "ch2_iir_nci_20231225T1904122779_g_grd_d18.csv")
TMC_IMG = (QA + "/pradan/ch2_tmc_ncf_20231101T0125121344_d_img_d18/"
           "data/calibrated/20231101/"
           "ch2_tmc_ncf_20231101T0125121344_d_img_d18.img")
TMC_GEO = (QA + "/pradan/ch2_tmc_ncf_20231101T0125121344_d_img_d18/"
           "geometry/calibrated/20231101/"
           "ch2_tmc_ncf_20231101T0125121344_g_grd_d18.csv")

QUB_LINES, QUB_SAMPLES, QUB_BANDS = 12842, 250, 256
TMC_LINES, TMC_SAMPLES = 190000, 4000
VNIR_BANDS = list(range(0, 47))

WINDOWS = {
    "W0": (0, 4634, 0, 218),
    "W1": (0, 2250, 0, 218),
    "W2": (2250, 4500, 0, 218),
    "W3": (0, 2250, 32, 250),
    "W4": (2250, 4500, 32, 250),
    "W5": (1125, 3375, 0, 218),
}
SPLIT = {"train": ["W0", "W1", "W2", "W3"], "val": ["W4"], "test": ["W5"]}
OUT_ROOT = "/home/hatch/workspace/crossfeat_data"

_qub = None
_tmc = None


def load_geo(path):
    import csv
    pts = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            pts.append(row)
    return pts


class Grid:
    def __init__(self, path, scan_key="scan", pixel_key="pixel",
                 lat_key="lat", lon_key="lon"):
        self.pts = load_geo(path)
        # Simplified - real impl in iirs_breadth.py
        self.scan_key = scan_key


def vnir_window(l0, l1, s0, s1):
    global _qub
    if _qub is None:
        _qub = np.memmap(QUB, dtype="<f4", mode="r",
                         shape=(QUB_BANDS, QUB_LINES, QUB_SAMPLES))
    acc = np.zeros((l1 - l0, s1 - s0), np.float64)
    for b in VNIR_BANDS:
        raw = np.asarray(_qub[b, l0:l1, s0:s1], dtype=np.float64)
        col_med = np.median(raw, axis=0)
        acc += raw - (col_med - np.median(col_med))[None, :]
    acc /= len(VNIR_BANDS)
    return acc.astype(np.float32)


def u8norm(a):
    lo, hi = np.percentile(a, (1, 99))
    a = np.clip((a - lo) / max(1e-6, hi - lo), 0, 1)
    return (a * 255).astype(np.uint8)


def main():
    from app import _align_core

    os.makedirs(OUT_ROOT, exist_ok=True)
    for split, tags in SPLIT.items():
        for mod in ["modality_iirs", "modality_tmc"]:
            os.makedirs(f"{OUT_ROOT}/{split}/{mod}", exist_ok=True)

    # Load grids (simplified - need real implementation from iirs_breadth.py)
    print("NOTE: This is a scaffold. Full implementation requires")
    print("the grid helpers from scripts/iirs_breadth.py.")
    print()
    print("For now, document the plan:")
    for split, tags in SPLIT.items():
        for tag in tags:
            l0, l1, s0, s1 = WINDOWS[tag]
            print(f"  {split}/{tag}: IIRS lines {l0}:{l1} samples {s0}:{s1}")


if __name__ == "__main__":
    main()
