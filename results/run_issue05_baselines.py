"""Run RoMa, LightGlue/ALIKED, and SIFT/RANSAC on identical issue #3 crops.

Install the optional baseline with ``python -m pip install romatch==0.1.2``.
RoMa's public outdoor and DINOv2 checkpoints are cached by torch.hub on first run.
"""

from __future__ import annotations

import csv
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT if (ROOT / "data" / "qa" / "lite").exists() else ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results"))
CSV_PATH = ROOT / "results" / "table_issue05_baselines.csv"
FIELDS = [
    "pair", "matcher", "correspondences", "rmse_px", "rmse_insample_px",
    "rmse_heldout_px", "inliers", "verdict", "runtime_s", "final_engine",
    "secondary_fallback", "notes",
]


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight failed; no matcher was run")
    torch.cuda.init()
    torch.cuda.synchronize()


def load_pairs():
    import run_issue01_gpu_validation as issue01
    import run_issue03_benchmark as issue03

    # Canonical image inputs are ignored local assets and may only be present
    # in the primary checkout when this runner is executed from a linked worktree.
    issue01.ROOT = DATA_ROOT
    issue01.LITE = DATA_ROOT / "data" / "qa" / "lite" / "chandra-align-images"
    issue01.FULL = DATA_ROOT / "data" / "qa" / "fullres"
    issue03.ROOT = ROOT
    canonical = issue01.load_pairs()
    synthetic = next(row for row in canonical if row[0] == "synthetic_gentle")
    benchmark = issue03.load_manifest(ROOT / "data" / "benchmark_pairs.csv")
    if len(benchmark) != 12:
        raise RuntimeError(f"Expected 12 issue #3 crops, found {len(benchmark)}")
    pairs = [(synthetic[0], synthetic[1], synthetic[2], synthetic[3], synthetic[4])]
    scales = {"OHRC": 0.26, "TMC-2": 4.47}
    for row in benchmark:
        pairs.append((
            row["pair_id"], row.pop("reference_crop_array"),
            row.pop("source_crop_array"), row["sensor"], scales[row["sensor"]],
        ))
    return pairs


def _uint8_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        if array.ndim == 2:
            return cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
        if array.ndim == 3 and array.shape[2] == 4:
            return cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
        if array.ndim == 3 and array.shape[2] == 1:
            return cv2.cvtColor(array[:, :, 0], cv2.COLOR_GRAY2RGB)
        if array.ndim == 3 and array.shape[2] >= 3:
            return np.ascontiguousarray(array[:, :, :3])
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
        elif array.shape[2] == 1:
            array = array[:, :, 0]
        elif array.shape[2] >= 3:
            return np.clip(array[:, :, :3], 0, 255).astype(np.uint8)
    if array.ndim != 2:
        raise ValueError(f"Expected grayscale or RGB image, got {array.shape}")
    array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    low, high = np.percentile(array, [1, 99])
    if high <= low:
        low, high = float(array.min()), float(array.max())
    gray = np.zeros(array.shape, dtype=np.uint8) if high <= low else np.clip(
        (array - low) * (255.0 / (high - low)), 0, 255
    ).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def roma_correspondences(model, image_ref, image_src, output_shape):
    """Extract top-certainty RoMa dense correspondences in source pixel units."""
    rgb_ref, rgb_src = _uint8_rgb(image_ref), _uint8_rgb(image_src)
    pil_ref, pil_src = Image.fromarray(rgb_ref), Image.fromarray(rgb_src)
    with torch.inference_mode():
        warp, certainty = model.match(pil_ref, pil_src, device="cuda")
    half_width = warp.shape[2] // 2
    flow = warp[0, :, :half_width]
    scores = certainty[0, :, :half_width].reshape(-1)
    valid = torch.isfinite(scores) & (scores > 0.05)
    valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_idx.numel() == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty.copy()
    count = min(4096, int(valid_idx.numel()))
    ranked = torch.topk(scores[valid_idx], k=count, sorted=True).indices
    selected = valid_idx[ranked]
    height, width = flow.shape[:2]
    y = torch.div(selected, width, rounding_mode="floor")
    x = selected % width
    normalized = flow.reshape(-1, 4)[selected]
    h_src, w_src = output_shape
    # RoMa emits normalized [-1, 1] coordinates for align_corners=False grids.
    reference_points = torch.stack(((normalized[:, 0] + 1) * w_src / 2 - 0.5,
                                    (normalized[:, 1] + 1) * h_src / 2 - 0.5), dim=1)
    source_points = torch.stack(((normalized[:, 2] + 1) * w_src / 2 - 0.5,
                                 (normalized[:, 3] + 1) * h_src / 2 - 0.5), dim=1)
    points_ref = reference_points.detach().cpu().numpy().astype(np.float32)
    points_src = source_points.detach().cpu().numpy().astype(np.float32)
    in_bounds = (
        (points_ref[:, 0] >= 0) & (points_ref[:, 0] < w_src) &
        (points_ref[:, 1] >= 0) & (points_ref[:, 1] < h_src) &
        (points_src[:, 0] >= 0) & (points_src[:, 0] < w_src) &
        (points_src[:, 1] >= 0) & (points_src[:, 1] < h_src)
    )
    return points_ref[in_bounds], points_src[in_bounds]


def no_secondary_chain(*args, **kwargs):
    class EmptyResult:
        pts_src = np.empty((0, 2), dtype=np.float32)
        pts_ref = np.empty((0, 2), dtype=np.float32)
        error_msg = "Disabled for isolated matcher baseline"
        matcher_name = "disabled"

    class Chain:
        def __init__(self, *chain_args, **chain_kwargs):
            pass

        def match(self, *match_args, **match_kwargs):
            return EmptyResult()

    return Chain


def disabled_rift(*args, **kwargs):
    class Matcher:
        def __init__(self, *matcher_args, **matcher_kwargs):
            pass

        def match(self, *match_args, **match_kwargs):
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    return Matcher


def run_one(model, pair_id, matcher_name, ref, src, sensor, pixel_scale):
    import app
    import chandra_align.matcher as matcher_module
    import chandra_align.matching.deep_matchers as deep_module

    original_match_fn = app.match_pair_hf
    original_rift = matcher_module.RIFT2Matcher
    original_chain = deep_module.DeepMatcherChain
    state = {"correspondences": 0, "error": ""}

    def selected_matcher(image_ref, image_src, threshold=3.0):
        try:
            if matcher_name == "RoMa":
                points_ref, points_src = roma_correspondences(
                    model, image_ref, image_src, image_ref.shape[:2]
                )
                engine = "RoMa outdoor"
            elif matcher_name == "LightGlue_ALIKED":
                from chandra_align.matching.deep_matchers import LightGlueALIKEDMatcher
                result = LightGlueALIKEDMatcher(max_keypoints=2048).match(image_ref, image_src)
                points_ref, points_src = result.pts_src, result.pts_ref
                engine = "LightGlue/ALIKED"
                if result.status == "FAILED":
                    state["error"] = result.error_msg or "LightGlue/ALIKED failed"
            else:
                points_ref, points_src, count = app._match_sift_ransac(
                    image_ref, image_src, threshold
                )
                engine = "SIFT + Brute-Force (RANSAC)"
                if count is None:
                    state["error"] = "SIFT produced no RANSAC inlier set"
            points_ref = np.asarray(points_ref, dtype=np.float32).reshape(-1, 2)
            points_src = np.asarray(points_src, dtype=np.float32).reshape(-1, 2)
            state["correspondences"] = len(points_ref)
            if len(points_ref) < 3:
                state["error"] = state["error"] or f"Only {len(points_ref)} correspondences"
            return points_ref, points_src, engine, {
                "primary_engine": engine, "registration_engine": engine,
                "fallback_triggered": False, "fallback_reason": None,
            }
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
            state["correspondences"] = 0
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty.copy(), matcher_name, {
                "primary_engine": matcher_name, "registration_engine": matcher_name,
                "fallback_triggered": False, "fallback_reason": None,
            }

    app.match_pair_hf = selected_matcher
    deep_module.DeepMatcherChain = no_secondary_chain()
    try:
        torch.cuda.synchronize()
        cv2.setRNGSeed(0)
        started = time.perf_counter()
        result = app._align_core(
            ref, src, pixel_scale_m=pixel_scale, enable_clahe=True,
            enable_shadow_suppression=False, enable_wallis=False,
            enforce_uniform_distribution=True,
            sensor_pair_mode="Optical <-> Optical",
            secondary_sensor_name=sensor, reference_sensor_name=sensor,
            max_image_dimension=2048,
        )
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        metric = lambda value: "" if value is None else value
        return {
            "pair": pair_id, "matcher": matcher_name,
            "correspondences": state["correspondences"],
            "rmse_px": metric(result.get("rmse_px", "")),
            "rmse_insample_px": metric(result.get("rmse_in_sample_px", "")),
            "rmse_heldout_px": metric(result.get("rmse_heldout_px", "")),
            "inliers": int(result.get("inlier_cnt", 0)),
            "verdict": result.get("status_code", "UNKNOWN"),
            "runtime_s": wall,
            "final_engine": result.get("engine_name", matcher_name),
            "secondary_fallback": bool(result.get("execution_diagnostics", {}).get("fallback_triggered", False)),
            "notes": state["error"],
        }
    except Exception as exc:
        torch.cuda.synchronize()
        return {
            "pair": pair_id, "matcher": matcher_name,
            "correspondences": state["correspondences"], "rmse_px": "",
            "rmse_insample_px": "", "rmse_heldout_px": "", "inliers": 0,
            "verdict": "REJECTED", "runtime_s": time.perf_counter() - started,
            "final_engine": matcher_name, "secondary_fallback": False,
            "notes": state["error"] or f"{type(exc).__name__}: {exc}",
        }
    finally:
        app.match_pair_hf = original_match_fn
        matcher_module.RIFT2Matcher = original_rift
        deep_module.DeepMatcherChain = original_chain


def run_group(group: str, output_path: Path) -> None:
    require_cuda()
    pairs = load_pairs()
    rows = []
    if group == "roma":
        from romatch import roma_outdoor
        print("Initializing RoMa outdoor model (public checkpoints are cached on first run)", flush=True)
        # Reduced matcher working resolution keeps RoMa within the RTX 5070
        # Windows memory budget while preserving the original input arrays.
        roma_model = roma_outdoor(
            device="cuda", amp_dtype=torch.float16,
            coarse_res=280, upsample_res=432,
        )
        selected = (("RoMa", roma_model),)
    else:
        selected = (("LightGlue_ALIKED", None), ("SIFT_RANSAC", None))
    for matcher_name, model in selected:
        for pair_id, ref, src, sensor, pixel_scale in pairs:
            print(f"Running {pair_id} / {matcher_name}; shared arrays {ref.shape}", flush=True)
            row = run_one(model, pair_id, matcher_name, ref, src, sensor, pixel_scale)
            rows.append(row)
            print(row, flush=True)
            torch.cuda.empty_cache()
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_path} ({len(rows)} rows)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("classic", "roma"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.group:
        if args.output is None:
            parser.error("--output is required with --group")
        run_group(args.group, args.output)
        return

    # Separate processes clear CUDA allocator state between the classic
    # baselines and RoMa, which otherwise exhausts Windows GPU memory.
    with tempfile.TemporaryDirectory(prefix="issue05-") as temporary:
        temp = Path(temporary)
        outputs = [temp / "classic.csv", temp / "roma.csv"]
        for group, output in zip(("classic", "roma"), outputs):
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--group", group,
                 "--output", str(output)],
                cwd=ROOT, check=False,
            )
            if not output.exists():
                if group == "roma":
                    # Preserve a visible failure row if a native RoMa/CUDA
                    # crash prevents the child process from writing its CSV.
                    failed_rows = []
                    for pair_id, *_ in load_pairs():
                        failed_rows.append({
                            field: (pair_id if field == "pair" else "RoMa" if field == "matcher"
                                    else "REJECTED" if field == "verdict"
                                    else f"RoMa child process exited {completed.returncode} before writing output"
                                    if field == "notes" else "")
                            for field in FIELDS
                        })
                    with output.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=FIELDS)
                        writer.writeheader()
                        writer.writerows(failed_rows)
                else:
                    raise RuntimeError(
                        f"{group} baseline child exited {completed.returncode} without writing results"
                    )
        all_rows = []
        for output in outputs:
            with output.open(newline="", encoding="utf-8") as handle:
                all_rows.extend(csv.DictReader(handle))
        with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote {CSV_PATH} ({len(all_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
