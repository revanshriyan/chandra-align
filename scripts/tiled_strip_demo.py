"""Compare eager 8192-square loading with native-resolution tiled matching."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYTHON = sys.executable
REFERENCE_PRODUCT = "ch2_ohr_nrp_20240425T1209509264_d_img_d18"
SOURCE_PRODUCT = "ch2_ohr_nrp_20240425T1406019344_d_img_d18"
ROI = (2000, 10000, 8192, 8192)
OUTPUT = ROOT / "results" / "issue04_tiled_memory.json"


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _working_set_bytes(pid: int) -> int:
    """Get a Windows worker's resident bytes without adding a runtime dependency."""
    if os.name != "nt":
        return 0
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.wintypes.DWORD]
    kernel32.OpenProcess.restype = ctypes.wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), ctypes.wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL
    handle = kernel32.OpenProcess(0x0400 | 0x0010, False, int(pid))
    if not handle:
        return 0
    try:
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return 0
        return int(counters.PeakWorkingSetSize)
    finally:
        kernel32.CloseHandle(handle)


def _lightglue_matcher():
    from chandra_align.matching.deep_matchers import LightGlueALIKEDMatcher
    matcher = LightGlueALIKEDMatcher(max_keypoints=2048)

    def run(reference, source, threshold=3.0):
        result = matcher.match(reference, source)
        if result.status == "FAILED":
            raise RuntimeError(result.error_msg or "LightGlue/ALIKED failed")
        return result.pts_src, result.pts_ref, "LightGlue/ALIKED", {
            "primary_engine": "LightGlue/ALIKED", "registration_engine": "LightGlue/ALIKED",
            "fallback_triggered": False, "fallback_reason": None,
        }
    return run


def worker(mode: str, input_root: Path):
    import app
    import cv2
    import rasterio
    import torch
    from rasterio.windows import Window
    from chandra_align.matching.tiled import tiled_registration
    from scripts.make_benchmark_crops import MAPPINGS

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this measured LightGlue demonstration")
    torch.cuda.init()
    torch.cuda.synchronize()
    ref_path = input_root / f"{REFERENCE_PRODUCT}.img"
    src_path = input_root / f"{SOURCE_PRODUCT}.img"
    rx, ry, width, height = ROI
    matcher = _lightglue_matcher()
    original_matcher = app.match_pair_hf
    app.match_pair_hf = matcher
    started = time.perf_counter()
    try:
        if mode == "eager":
            with rasterio.open(ref_path.with_suffix(".xml")) as dataset:
                ref_raw = dataset.read(1, window=Window(rx, ry, width, height), masked=True)
            sx0, sy0 = _map_origin(MAPPINGS["OHRC"], rx, ry, width, height)
            with rasterio.open(src_path.with_suffix(".xml")) as dataset:
                src_raw = dataset.read(1, window=Window(sx0, sy0, width, height), masked=True)
            ref = app.load_lunar_raster(np.asarray(ref_raw.astype(np.float32).filled(np.nan)), max_dimension=4096)
            src = app.load_lunar_raster(np.asarray(src_raw.astype(np.float32).filled(np.nan)), max_dimension=4096)
            result = app._align_core(
                ref, src, pixel_scale_m=0.26, enable_clahe=True,
                enable_shadow_suppression=False, enable_wallis=False,
                enforce_uniform_distribution=True, sensor_pair_mode="Optical <-> Optical",
                secondary_sensor_name="OHRC", reference_sensor_name="OHRC",
                max_image_dimension=4096, matching_mode="single_scale",
            )
            row = {
                "verdict": result.get("status_code"),
                "rmse_heldout": result.get("rmse_heldout_px"),
                "rmse_insample": result.get("rmse_in_sample_px"),
                "inliers": result.get("inlier_cnt"),
                "load_path": "one 8192x8192 pair read then resized to 4096 max dimension",
            }
        else:
            row = tiled_registration(
                ref_path, src_path, ROI, MAPPINGS["OHRC"], matcher,
                tile_size=2048, overlap=256, sensor_name="OHRC",
            )
        row["runtime_s"] = time.perf_counter() - started
        row["mode"] = mode
        row["peak_rss_bytes"] = _working_set_bytes(os.getpid())
        return row
    finally:
        app.match_pair_hf = original_matcher


def _map_origin(matrix, x, y, width, height):
    center = np.array([[x + width / 2, y + height / 2]], dtype=np.float64)
    mapped = center @ np.asarray(matrix)[:, :2].T + np.asarray(matrix)[:, 2]
    return int(round(mapped[0, 0] - width / 2)), int(round(mapped[0, 1] - height / 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=ROOT / "data" / "qa" / "fullres")
    parser.add_argument("--worker-mode", choices=("eager", "tiled"))
    args = parser.parse_args()
    if args.worker_mode:
        print(json.dumps(worker(args.worker_mode, args.input_root)))
        return

    measurements = {}
    for mode in ("eager", "tiled"):
        command = [PYTHON, str(Path(__file__).resolve()), "--input-root", str(args.input_root), "--worker-mode", mode]
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while child.poll() is None:
            time.sleep(0.05)
        stdout, stderr = child.communicate()
        if child.returncode:
            raise RuntimeError(f"{mode} worker failed ({child.returncode}): {stderr[-4000:]} {stdout[-1000:]}")
        row = json.loads(stdout.strip().splitlines()[-1])
        measurements[mode] = row
        print(f"{mode}: {row}; peak RSS={row['peak_rss_bytes'] / (1024 ** 2):.1f} MiB", flush=True)

    eager = measurements["eager"]["peak_rss_bytes"]
    tiled = measurements["tiled"]["peak_rss_bytes"]
    summary = {
        "input": [REFERENCE_PRODUCT, SOURCE_PRODUCT], "roi": ROI,
        "tile_size": 2048, "overlap": 256, "measurements": measurements,
        "peak_rss_reduction_fraction": (eager - tiled) / eager if eager else None,
        "notes": "Peak working set recorded inside each worker with Windows process counters; this is process RSS, not GPU VRAM. Ground-truth RMSE not available.",
    }
    OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
