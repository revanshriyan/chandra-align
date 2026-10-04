"""Run isolated matcher measurements on the three canonical image pairs.

The harness calls the existing app alignment/metric path while routing exactly
one matcher per run. It never edits application or matcher source files.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CSV_PATH = RESULTS / "table_issue01_gpu_validation.csv"
LITE = ROOT / "data" / "qa" / "lite" / "chandra-align-images"
FULL = ROOT / "data" / "qa" / "fullres"
sys.path.insert(0, str(ROOT))

FIELDS = [
    "pair", "matcher", "correspondences", "rmse_px", "inliers", "entropy",
    "quadrants", "gate", "runtime_s",
]


def read_pds(path: Path, window: Window | None = None) -> np.ndarray:
    """Read a PDS4 raster through its detached label; return app-normalized uint8."""
    label = path.with_suffix(".xml")
    with rasterio.open(label if label.exists() else path) as src:
        if window is None:
            raise ValueError("Full PDS inputs must use app.load_lunar_raster's bounded read")
        raw = src.read(1, window=window, masked=True)
        raw = np.asarray(raw.astype(np.float32).filled(np.nan))
    import app

    return app.load_lunar_raster(raw, max_dimension=4096)


def load_pairs() -> list[tuple[str, np.ndarray, np.ndarray, str, float]]:
    import app

    synthetic_ref = cv2.imread(str(LITE / "lunar_reference.png"), cv2.IMREAD_GRAYSCALE)
    synthetic_sec = cv2.imread(str(LITE / "lunar_target_gentle.png"), cv2.IMREAD_GRAYSCALE)
    if synthetic_ref is None or synthetic_sec is None:
        raise FileNotFoundError("Canonical gentle synthetic pair is missing")

    ohrc_ref_path = FULL / "ch2_ohr_nrp_20240425T1209509264_d_img_d18.img"
    ohrc_sec_path = FULL / "ch2_ohr_nrp_20240425T1406019344_d_img_d18.img"
    ohrc_ref = app.load_lunar_raster(str(ohrc_ref_path), max_dimension=4096)
    ohrc_sec = app.load_lunar_raster(str(ohrc_sec_path), max_dimension=4096)
    if ohrc_ref is None or ohrc_sec is None:
        raise RuntimeError("Could not load labeled OHRC rasters with the app loader")

    # Audit extents are 1-based inclusive. Convert to Rasterio's zero-based
    # offsets: fore lines 93329-95376 / samples 362-2409; nadir lines
    # 101612-103659 / samples 417-2464.
    tmc_fore = read_pds(
        FULL / "ch2_tmc_ncf_20231101T0125121344_d_img_d18.img",
        Window(col_off=361, row_off=93328, width=2048, height=2048),
    )
    tmc_nadir = read_pds(
        FULL / "ch2_tmc_ncn_20231101T0125121377_d_img_d18.img",
        Window(col_off=416, row_off=101611, width=2048, height=2048),
    )
    if tmc_fore is None or tmc_nadir is None:
        raise RuntimeError("Could not load the audited TMC-2 windows")

    return [
        ("synthetic_gentle", synthetic_ref, synthetic_sec, "OHRC", 0.26),
        ("OHRC_pair", ohrc_ref, ohrc_sec, "OHRC", 0.26),
        ("TMC2_fore_nadir", tmc_fore, tmc_nadir, "TMC-2", 4.47),
    ]


def no_secondary_chain(*args, **kwargs):
    """Prevent _align_core from silently replacing the matcher under test."""

    class Result:
        pts_src = np.empty((0, 2), dtype=np.float32)
        pts_ref = np.empty((0, 2), dtype=np.float32)
        error_msg = "Disabled: validation isolates one matcher per run"
        matcher_name = "disabled"

    class Chain:
        def __init__(self, *chain_args, **chain_kwargs):
            pass

        def match(self, *match_args, **match_kwargs):
            return Result()

    return Chain


def disabled_rift(*args, **kwargs):
    class Matcher:
        def __init__(self, *matcher_args, **matcher_kwargs):
            pass

        def match(self, *match_args, **match_kwargs):
            raise RuntimeError("RIFT2 disabled for isolated SIFT baseline")

    return Matcher


def run_one(pair, matcher_name, ref, sec, sensor, pixel_scale, artifact_path=None):
    import app
    import chandra_align.matcher as matcher_module
    import chandra_align.matching.deep_matchers as deep_module

    original_match_fn = app.match_pair_hf
    original_rift = matcher_module.RIFT2Matcher
    original_chain = deep_module.DeepMatcherChain
    state = {"correspondences": 0, "error": ""}

    def selected_matcher(img_ref, img_sec, threshold=3.0):
        started = time.perf_counter()
        try:
            if matcher_name == "RIFT2":
                points_ref, points_sec = original_rift(npt=2048).match(img_ref, img_sec)
                engine = "RIFT2 (Phase Congruency)"
            elif matcher_name == "LightGlue_ALIKED":
                from chandra_align.matching.deep_matchers import LightGlueALIKEDMatcher

                result = LightGlueALIKEDMatcher(max_keypoints=2048).match(img_ref, img_sec)
                points_ref, points_sec = result.pts_src, result.pts_ref
                if result.status == "FAILED":
                    state["error"] = result.error_msg or "LightGlue/ALIKED failed"
                engine = "LightGlue/ALIKED"
            else:
                # Invoke the production fallback implementation directly after
                # suppressing only its two preceding matcher attempts.
                matcher_module.RIFT2Matcher = disabled_rift()
                points_ref, points_sec, engine, diag = original_match_fn(
                    img_ref, img_sec, threshold
                )
                if diag.get("sift_error"):
                    state["error"] = diag["sift_error"]
            points_ref = np.asarray(points_ref, dtype=np.float32).reshape(-1, 2)
            points_sec = np.asarray(points_sec, dtype=np.float32).reshape(-1, 2)
            state["correspondences"] = len(points_ref)
            if len(points_ref) < 3:
                state["error"] = state["error"] or f"Only {len(points_ref)} correspondences"
            diagnostics = {
                "primary_engine": engine,
                "registration_engine": engine,
                "fallback_triggered": False,
                "fallback_reason": None,
            }
            return points_ref, points_sec, engine, diagnostics
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
            state["correspondences"] = 0
            return (
                np.empty((0, 2), np.float32), np.empty((0, 2), np.float32),
                matcher_name, {"primary_engine": matcher_name,
                               "registration_engine": matcher_name,
                               "fallback_triggered": False,
                               "fallback_reason": None},
            )
        finally:
            state["matcher_runtime"] = time.perf_counter() - started

    app.match_pair_hf = selected_matcher
    deep_module.DeepMatcherChain = no_secondary_chain()
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cv2.setRNGSeed(0)
        started = time.perf_counter()
        result = app._align_core(
            ref, sec, pixel_scale_m=pixel_scale, enable_clahe=True,
            enable_shadow_suppression=False, enable_wallis=False,
            enforce_uniform_distribution=True,
            sensor_pair_mode="Optical <-> Optical",
            secondary_sensor_name=sensor, reference_sensor_name=sensor,
            max_image_dimension=4096,
        )
        if artifact_path is not None and result.get("affine_matrix") is not None:
            np.savez_compressed(
                artifact_path,
                affine_matrix=np.asarray(result["affine_matrix"], dtype=np.float64),
                pts_ref_inliers=np.asarray(result["pts_ref_inliers"], dtype=np.float64),
                pts_src_inliers=np.asarray(result["pts_sec_inliers"], dtype=np.float64),
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - started
        quadrants = result.get("quadrant_metrics", {})
        counts = [int(quadrants.get(f"Q{i}", {}).get("inlier_count", 0)) for i in range(1, 5)]
        return {
            "pair": pair,
            "matcher": matcher_name,
            "correspondences": state["correspondences"],
            "rmse_px": result.get("rmse_px"),
            "inliers": result.get("inlier_cnt", 0),
            "entropy": result.get("quadrant_spatial_entropy", 0.0),
            "quadrants": f"{sum(value > 0 for value in counts)}/4 ({','.join(map(str, counts))})",
            "gate": result.get("status_message", result.get("status_code", "UNKNOWN")),
            "runtime_s": wall,
        }
    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - started
        return {
            "pair": pair,
            "matcher": matcher_name,
            "correspondences": state["correspondences"],
            "rmse_px": "",
            "inliers": 0,
            "entropy": 0.0,
            "quadrants": "0/4 (0,0,0,0)",
            "gate": f"REJECTED: {state['error'] or exc}",
            "runtime_s": wall,
        }
    finally:
        app.match_pair_hf = original_match_fn
        matcher_module.RIFT2Matcher = original_rift
        deep_module.DeepMatcherChain = original_chain


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; no matcher was run")
    torch.cuda.init()
    torch.cuda.synchronize()
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair, ref, sec, sensor, pixel_scale in load_pairs():
        for matcher_name in ("RIFT2", "LightGlue_ALIKED", "SIFT_RANSAC"):
            print(f"Running {pair} / {matcher_name} ({ref.shape} vs {sec.shape})", flush=True)
            row = run_one(pair, matcher_name, ref, sec, sensor, pixel_scale)
            rows.append(row)
            print(row, flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
