"""Run illumination-sensitivity checks on a real LOLA DEM patch.

The input is a bounded HTTP-range window from the official USGS LOLA 118 m
global DEM. It is rendered with a simplified Hapke-style photometric model;
this is an illumination sensitivity experiment, not a physical image simulator.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
DEM_URL = (
    "https://planetarymaps.usgs.gov/mosaic/"
    "Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"
)
DEM_VSICURL = "/vsicurl/" + DEM_URL
DEM_NPZ = ROOT / "results" / "issue11_lola_dem_tile.npz"
RAW_JSON = ROOT / "results" / "issue11_sunangle_runs.json"
CSV_PATH = ROOT / "results" / "table_issue11_sunangle.csv"
PLOT_PATH = ROOT / "docs" / "images" / "sunangle-degradation.png"
SCALE_M = 0.5
PIXEL_SIZE_M = 118.45
WINDOW = Window(45952, 22912, 128, 128)
UPSCALE = 4

# Azimuth is clockwise from north; elevation is above the local horizon.
ILLUMINATIONS = [
    ("reference", 0.0, 30.0),
    ("az_minus45_el30", -45.0, 30.0),
    ("az_plus45_el30", 45.0, 30.0),
    ("az0_el15", 0.0, 15.0),
    ("az0_el60", 0.0, 60.0),
    ("az_minus45_el60", -45.0, 60.0),
    ("az_plus45_el15", 45.0, 15.0),
]
FIELDS = [
    "pair_id", "matcher", "solar_azimuth_deg", "solar_elevation_deg",
    "correspondences", "inliers", "rmse_heldout_split_px",
    "n_heldout_points", "identity_gt_rmse_px", "entropy", "quadrants",
    "verdict", "registration_engine", "fallback_path", "runtime_s", "notes",
]


def load_dem_tile() -> tuple[np.ndarray, str]:
    if DEM_NPZ.exists():
        with np.load(DEM_NPZ) as data:
            raw = data["raw_dn"].astype(np.int16, copy=True)
            digest = str(data["source_sha256"].item())
        if raw.shape != (int(WINDOW.height), int(WINDOW.width)):
            raise ValueError(f"Saved DEM tile has unexpected shape {raw.shape}")
        return raw.astype(np.float32) * SCALE_M, digest

    with rasterio.open(DEM_VSICURL) as dataset:
        raw = dataset.read(1, window=WINDOW)
        scale = float(dataset.scales[0])
        offset = float(dataset.offsets[0])
        nodata = dataset.nodata
    if raw.shape != (128, 128) or scale != SCALE_M or offset != 0.0:
        raise ValueError(
            f"Unexpected source tile metadata: shape={raw.shape}, scale={scale}, offset={offset}"
        )
    if nodata is not None and np.any(raw == nodata):
        raise ValueError("Selected LOLA tile contains nodata; choose a valid tile before rendering")
    digest = hashlib.sha256(raw.tobytes()).hexdigest()
    DEM_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DEM_NPZ, raw_dn=raw.astype(np.int16), source_sha256=np.array(digest))
    return raw.astype(np.float32) * scale + offset, digest


def hapke_style_render(height_m: np.ndarray, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Simplified nadir-view Hapke-style reflectance rendered to uint8."""
    dz_drow, dz_dcol = np.gradient(height_m, PIXEL_SIZE_M, PIXEL_SIZE_M)
    # The array's rows increase southward in this near-equatorial patch.
    east = -dz_dcol
    north = dz_drow
    up = np.ones_like(height_m, dtype=np.float32)
    norm = np.sqrt(east * east + north * north + up * up)
    east, north, up = east / norm, north / norm, up / norm

    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    sun_east = np.cos(el) * np.sin(az)
    sun_north = np.cos(el) * np.cos(az)
    sun_up = np.sin(el)
    mu0 = np.maximum(east * sun_east + north * sun_north + up * sun_up, 0.0)
    mu = np.maximum(up, 1e-5)  # nadir viewing geometry

    w = 0.55  # single-scattering albedo, fixed for every lighting variant
    h0 = (1.0 + 2.0 * mu0) / (1.0 + 2.0 * mu0 * np.sqrt(1.0 - w))
    h = (1.0 + 2.0 * mu) / (1.0 + 2.0 * mu * np.sqrt(1.0 - w))
    phase_angle = np.pi / 2.0 - el
    phase_function = 1.0 + 0.2 * np.cos(phase_angle)
    reflectance = (w / (4.0 * np.pi)) * (mu0 / np.maximum(mu0 + mu, 1e-6))
    reflectance *= phase_function + h0 * h - 1.0
    reflectance[mu0 <= 0.0] = 0.0
    return reflectance.astype(np.float32)


def to_common_uint8(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    joined = np.concatenate([image.ravel() for image in images.values()])
    low, high = np.percentile(joined, [1.0, 99.0])
    if not np.isfinite(high - low) or high <= low:
        raise ValueError("Rendered illumination variants have no usable radiometric range")
    return {
        name: cv2.resize(
            np.clip((image - low) * (255.0 / (high - low)), 0.0, 255.0).astype(np.uint8),
            (image.shape[1] * UPSCALE, image.shape[0] * UPSCALE),
            interpolation=cv2.INTER_CUBIC,
        )
        for name, image in images.items()
    }


def isolated_match(app, matcher_name: str, ref: np.ndarray, src: np.ndarray):
    """Return repository matcher points using the app's regular entry points."""
    if matcher_name == "LightGlue_ALIKED":
        from chandra_align.matching.deep_matchers import LightGlueALIKEDMatcher

        result = LightGlueALIKEDMatcher(max_keypoints=1024).match(ref, src)
        return (
            np.asarray(result.pts_src, np.float32).reshape(-1, 2),
            np.asarray(result.pts_ref, np.float32).reshape(-1, 2),
            "LightGlue/ALIKED",
            {
                "primary_engine": "LightGlue/ALIKED",
                "registration_engine": "LightGlue/ALIKED",
                "fallback_triggered": False,
                "fallback_reason": None,
                "fallback_error": result.error_msg if result.status == "FAILED" else None,
            },
        )

    pts_ref, pts_sec, inlier_count = app._match_sift_ransac(ref, src, 3.0)
    if inlier_count is None:
        pts_ref = np.empty((0, 2), np.float32)
        pts_sec = np.empty((0, 2), np.float32)
    return (
        np.asarray(pts_ref, np.float32).reshape(-1, 2),
        np.asarray(pts_sec, np.float32).reshape(-1, 2),
        "SIFT + Brute-Force (RANSAC)",
        {
            "primary_engine": "SIFT + Brute-Force (RANSAC)",
            "registration_engine": "SIFT + Brute-Force (RANSAC)",
            "fallback_triggered": False,
            "fallback_reason": None,
            "fallback_error": None if inlier_count is not None else "SIFT/RANSAC found no candidate",
        },
    )


def identity_rmse(affine: np.ndarray, size: int) -> float:
    # These nine grid coordinates are independent of detected/matched points;
    # each rendered image comes from the same DEM pixels, so the true map is I.
    coords = np.array(
        [[x, y] for x in (0.15, 0.5, 0.85) for y in (0.15, 0.5, 0.85)],
        dtype=np.float64,
    ) * float(size - 1)
    estimated = coords @ affine[:, :2].T + affine[:, 2]
    return float(np.sqrt(np.mean(np.sum((estimated - coords) ** 2, axis=1))))


def run() -> list[dict]:
    np.random.seed(0)
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("CUDA unavailable; LightGlue will use its installed device policy.", flush=True)
    else:
        torch.cuda.manual_seed_all(0)
    height_m, digest = load_dem_tile()
    rendered = {
        name: hapke_style_render(height_m, az, el)
        for name, az, el in ILLUMINATIONS
    }
    images = to_common_uint8(rendered)

    sys.path.insert(0, str(ROOT))
    import app

    original_match = app.match_pair_hf
    rows = []
    try:
        for name, azimuth, elevation in ILLUMINATIONS[1:]:
            for matcher in ("LightGlue_ALIKED", "SIFT_RANSAC"):
                state = {"correspondences": 0}

                def selected(ref, src, threshold=3.0):
                    pts_ref, pts_src, engine, diagnostics = isolated_match(
                        app, matcher, ref, src
                    )
                    state["correspondences"] = len(pts_ref)
                    return pts_ref, pts_src, engine, diagnostics

                app.match_pair_hf = selected
                cv2.setRNGSeed(0)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                try:
                    result = app._align_core(
                        images["reference"], images[name], pixel_scale_m=PIXEL_SIZE_M / UPSCALE,
                        enable_clahe=False, enable_shadow_suppression=False,
                        enable_wallis=False, enforce_uniform_distribution=True,
                        sensor_pair_mode="Optical <-> Optical",
                        secondary_sensor_name="OHRC", reference_sensor_name="OHRC",
                        max_image_dimension=1024,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    heldout = result.get("heldout_error") or {}
                    judge_metrics = result.get("judge_metrics", {})
                    quad = result.get("quadrant_metrics", {})
                    counts = [int(quad.get(f"Q{i}", {}).get("inlier_count", 0)) for i in range(1, 5)]
                    if not quad:
                        counts = [int(value) for value in judge_metrics.get("quadrant_counts", counts)]
                    assessment = result.get("confidence_assessment", {})
                    diagnostics = result.get("execution_diagnostics", {})
                    affine = result.get("affine_matrix")
                    identity_error = (
                        identity_rmse(np.asarray(affine, dtype=np.float64), images[name].shape[0])
                        if affine is not None else ""
                    )
                    rows.append({
                        "pair_id": name, "matcher": matcher,
                        "solar_azimuth_deg": azimuth,
                        "solar_elevation_deg": elevation,
                        "correspondences": state["correspondences"],
                        "inliers": int(result.get("inlier_cnt", 0)),
                        "rmse_heldout_split_px": heldout.get("rmse_px", ""),
                        "n_heldout_points": int(heldout.get("n_check_points", 0)),
                        "identity_gt_rmse_px": identity_error,
                        "entropy": float(result.get("quadrant_spatial_entropy", 0.0)),
                        "quadrants": counts,
                        "verdict": result.get("status_code", "UNKNOWN"),
                        "registration_engine": diagnostics.get("registration_engine", "Unknown"),
                        "fallback_path": json.dumps(assessment.get("fallback_path", {}), sort_keys=True),
                        "runtime_s": time.perf_counter() - started,
                        "notes": "Exact identity GT on shared DEM grid; simulated illumination, not orbital imagery.",
                    })
                except Exception as exc:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    rows.append({
                        "pair_id": name, "matcher": matcher,
                        "solar_azimuth_deg": azimuth,
                        "solar_elevation_deg": elevation,
                        "correspondences": state["correspondences"], "inliers": 0,
                        "rmse_heldout_split_px": "", "n_heldout_points": 0,
                        "identity_gt_rmse_px": "", "entropy": 0.0,
                        "quadrants": [0, 0, 0, 0], "verdict": "REJECTED",
                        "registration_engine": "None", "fallback_path": "{}",
                        "runtime_s": time.perf_counter() - started,
                        "notes": f"{type(exc).__name__}: {exc}",
                    })
                print(f"{name}/{matcher}: {rows[-1]['verdict']}", flush=True)
    finally:
        app.match_pair_hf = original_match

    RAW_JSON.write_text(json.dumps({"source_sha256": digest, "runs": rows}, indent=2), encoding="utf-8")
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["quadrants"] = "/".join(map(str, row["quadrants"]))
            out["fallback_path"] = row["fallback_path"]
            for key in ("rmse_heldout_split_px", "identity_gt_rmse_px", "entropy", "runtime_s"):
                if isinstance(out[key], (float, int)):
                    out[key] = f"{out[key]:.5f}"
            writer.writerow(out)
    make_plot(rows, images)
    print(f"Wrote {len(rows)} rows; DEM tile SHA-256 {digest}")
    return rows


def make_plot(rows: list[dict], images: dict[str, np.ndarray]) -> None:
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["pair_id"].replace("az_", "") for row in rows if row["matcher"] == "LightGlue_ALIKED"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#101820")
    for axis, matcher, color in zip(axes, ("LightGlue_ALIKED", "SIFT_RANSAC"), ("#35b7e8", "#f5a142")):
        subset = [row for row in rows if row["matcher"] == matcher]
        values = [row["rmse_heldout_split_px"] if isinstance(row["rmse_heldout_split_px"], (int, float)) else np.nan for row in subset]
        axis.plot(range(len(values)), values, marker="o", color=color, linewidth=2)
        missing = [index for index, value in enumerate(values) if not np.isfinite(value)]
        if missing:
            axis.scatter(missing, [0.12] * len(missing), marker="x", color="#ff6b6b",
                         label="No usable split RMSE", zorder=5)
        axis.set_ylim(0, max([value for value in values if np.isfinite(value)] + [1.0]) * 1.15)
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8)
        axis.set_title(matcher.replace("_", "/"), color="white")
        axis.set_ylabel("Split held-out RMSE (px)", color="white")
        axis.grid(alpha=0.25)
        axis.tick_params(colors="white")
        axis.set_facecolor("#17232d")
        if missing:
            axis.legend(loc="upper left", fontsize=8, facecolor="#17232d", labelcolor="white")
    figure.suptitle("LOLA DEM illumination sensitivity — simplified Hapke-style renders", color="white")
    figure.text(0.5, 0.01, "128×128 real 118 m LOLA DEM crop, upscaled for matching; same-grid identity truth. Simulated radiometry.", ha="center", color="#c9d1d9", fontsize=8)
    figure.tight_layout(rect=(0, 0.05, 1, 0.92))
    figure.savefig(PLOT_PATH, dpi=130, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    run()
