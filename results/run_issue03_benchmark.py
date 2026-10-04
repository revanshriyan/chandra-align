"""Run isolated LightGlue/ALIKED and SIFT/RANSAC comparisons on Issue #3 crops."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RAW_PATH = ROOT / "results" / "issue03_benchmark_runs.json"


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight failed (torch.cuda.is_available() is False); no matcher was run")
    torch.cuda.init()
    torch.cuda.synchronize()


def no_secondary_chain(*args, **kwargs):
    class Result:
        pts_src = np.empty((0, 2), dtype=np.float32)
        pts_ref = np.empty((0, 2), dtype=np.float32)
        error_msg = "Disabled to isolate the matcher under test"
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
            raise RuntimeError("RIFT2 disabled for the isolated SIFT baseline")

    return Matcher


def load_manifest(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No benchmark pairs in {path}")
    for row in rows:
        for key in ("reference_crop", "source_crop"):
            image_path = ROOT / row[key]
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (int(row["crop_size"]), int(row["crop_size"])):
                raise ValueError(f"Invalid crop {image_path}")
            row[key + "_array"] = image
    return rows


def run_one(pair: dict, matcher_name: str, ref: np.ndarray, src: np.ndarray,
            sensor: str, pixel_scale: float) -> dict:
    import app
    import chandra_align.matcher as matcher_module
    import chandra_align.matching.deep_matchers as deep_module

    original_match_fn = app.match_pair_hf
    original_rift = matcher_module.RIFT2Matcher
    original_chain = deep_module.DeepMatcherChain
    state = {"correspondences": 0, "error": ""}

    def selected_matcher(img_ref, img_src, threshold=3.0):
        started = time.perf_counter()
        try:
            if matcher_name == "LightGlue_ALIKED":
                from chandra_align.matching.deep_matchers import LightGlueALIKEDMatcher
                result = LightGlueALIKEDMatcher(max_keypoints=2048).match(img_ref, img_src)
                points_ref, points_src = result.pts_src, result.pts_ref
                if result.status == "FAILED":
                    state["error"] = result.error_msg or "LightGlue/ALIKED failed"
                engine = "LightGlue/ALIKED"
            else:
                matcher_module.RIFT2Matcher = disabled_rift()
                points_ref, points_src, engine, diag = original_match_fn(img_ref, img_src, threshold)
                if diag.get("sift_error"):
                    state["error"] = diag["sift_error"]
            points_ref = np.asarray(points_ref, np.float32).reshape(-1, 2)
            points_src = np.asarray(points_src, np.float32).reshape(-1, 2)
            state["correspondences"] = len(points_ref)
            if len(points_ref) < 3:
                state["error"] = state["error"] or f"Only {len(points_ref)} correspondences"
            diagnostics = {"primary_engine": engine, "registration_engine": engine,
                           "fallback_triggered": False, "fallback_reason": None}
            return points_ref, points_src, engine, diagnostics
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
            state["correspondences"] = 0
            return (np.empty((0, 2), np.float32), np.empty((0, 2), np.float32),
                    matcher_name, {"primary_engine": matcher_name,
                                   "registration_engine": matcher_name,
                                   "fallback_triggered": False, "fallback_reason": None})
        finally:
            state["matcher_runtime_s"] = time.perf_counter() - started

    app.match_pair_hf = selected_matcher
    deep_module.DeepMatcherChain = no_secondary_chain()
    try:
        torch.cuda.synchronize()
        cv2.setRNGSeed(0)
        started = time.perf_counter()
        result = app._align_core(
            ref, src, pixel_scale_m=pixel_scale, enable_clahe=True,
            enable_shadow_suppression=False, enable_wallis=False,
            enforce_uniform_distribution=True, sensor_pair_mode="Optical <-> Optical",
            secondary_sensor_name=sensor, reference_sensor_name=sensor,
            max_image_dimension=2048,
        )
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        quads = result.get("quadrant_metrics")
        if isinstance(quads, dict):
            counts = [int(quads.get(f"Q{i}", {}).get("inlier_count", 0)) for i in range(1, 5)]
        else:
            counts = [int(v) for v in result.get("judge_metrics", {}).get("quadrant_counts", [0, 0, 0, 0])]
        heldout = result.get("heldout_error") or {}
        heldout_rmse = result.get("rmse_heldout_px")
        if heldout_rmse is None or not isinstance(heldout_rmse, (int, float)):
            heldout_rmse = ""
        return {
            "pair_id": pair["pair_id"], "sensor": sensor, "matcher": matcher_name,
            "correspondences": state["correspondences"],
            "rmse_insample": result.get("rmse_in_sample_px", ""),
            "rmse_heldout": heldout_rmse,
            "n_heldout_points": int(heldout.get("n_check_points", 0)),
            "inliers": int(result.get("inlier_cnt", 0)),
            "entropy": float(result.get("quadrant_spatial_entropy", 0.0)),
            "quadrant_counts": counts,
            "verdict": result.get("status_code", result.get("status_message", "UNKNOWN")),
            "final_engine": result.get("engine_name", matcher_name),
            "secondary_fallback": bool(result.get("execution_diagnostics", {}).get("fallback_triggered", False)),
            "runtime_s": wall, "gt_rmse": "",
            "notes": state["error"] if state["error"] else "Independent ground truth pending; browse-image crop mapping is approximate.",
        }
    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {
            "pair_id": pair["pair_id"], "sensor": sensor, "matcher": matcher_name,
            "correspondences": state["correspondences"], "rmse_insample": "",
            "rmse_heldout": "", "n_heldout_points": 0, "inliers": 0,
            "entropy": 0.0, "quadrant_counts": [0, 0, 0, 0],
            "verdict": "REJECTED", "runtime_s": time.perf_counter() - started,
            "final_engine": "", "secondary_fallback": False,
            "gt_rmse": "", "notes": f"{state['error'] or type(exc).__name__ + ': ' + str(exc)}",
        }
    finally:
        app.match_pair_hf = original_match_fn
        matcher_module.RIFT2Matcher = original_rift
        deep_module.DeepMatcherChain = original_chain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "benchmark_pairs.csv")
    args = parser.parse_args()
    require_cuda()
    pairs = load_manifest(args.manifest)
    rows = []
    pixel_scales = {"OHRC": 0.26, "TMC-2": 4.47}
    for pair in pairs:
        ref, src = pair.pop("reference_crop_array"), pair.pop("source_crop_array")
        for matcher in ("LightGlue_ALIKED", "SIFT_RANSAC"):
            print(f"Running {pair['pair_id']} / {matcher}; identical arrays {ref.shape}", flush=True)
            row = run_one(pair, matcher, ref, src, pair["sensor"], pixel_scales[pair["sensor"]])
            rows.append(row)
            print({k: v for k, v in row.items() if k != "notes"}, flush=True)
            torch.cuda.empty_cache()
    RAW_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {RAW_PATH}")


if __name__ == "__main__":
    main()
