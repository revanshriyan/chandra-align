#!/usr/bin/env python3
"""Bounded local and optional live QA checks for Chandra-Align.

Run ``python scripts/test_full_system.py --live`` to include a small number of
sequential Hugging Face requests. Live calls are deliberately serialized to
avoid creating a quota or concurrency stress event on the hosted Space.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHECKS: list[tuple[str, str, str, str]] = []


def module_from_script(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_check(module: str, target: str, check: Callable[[], str]) -> None:
    started = time.monotonic()
    try:
        metric = check()
        status = "PASS"
    except Exception as exc:
        status = "FAIL"
        metric = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    elapsed = time.monotonic() - started
    CHECKS.append((module, target, status, f"{metric} ({elapsed:.2f}s)"))


def skip_check(module: str, target: str, reason: str) -> None:
    CHECKS.append((module, target, "SKIP", reason))


def check_polar_ode() -> str:
    from xml.etree.ElementTree import Element, SubElement

    ode = module_from_script("qa_get_search_results", "scripts/get_search_results.py")
    corners = {
        "UL": (-89.94, -170.0),
        "UR": (-89.94, -60.0),
        "LR": (-89.20, 60.0),
        "LL": (-89.20, 170.0),
    }
    ring = ode._unwrap_ring(corners)
    span = max(lon for lon, _ in ring) - min(lon for lon, _ in ring)
    if span <= 180:
        raise AssertionError(f"Expected polar longitude span >180°, got {span:.2f}°")
    poly = ode.corners_to_polygon(corners)
    if poly.is_empty or not poly.is_valid:
        raise AssertionError("Polar footprint did not form a valid polygon")

    class Response:
        def json(self):
            return {"ODEResults": {"Products": {"Product": [
                {"pdsid": "same-tile"}, {"pdsid": "same-tile"}
            ]}}}

        def raise_for_status(self):
            return None

        def close(self):
            return None

    class Session:
        calls: list[dict] = []

        def get(self, url, params, timeout):
            self.calls.append(dict(params))
            return Response()

    session = Session()
    products = ode.query_ode(session, -89.94, -89.20, -170.0, 170.0)
    boxes = [(call["westlon"], call["eastlon"]) for call in session.calls]
    if boxes != [(190.0, 360.0), (0.0, 170.0)]:
        raise AssertionError(f"ODE polar query split was invalid: {boxes}")
    if len(products) != 1:
        raise AssertionError(f"Expected duplicate ODE results to collapse, got {len(products)}")

    # Exercise PDS4 coordinate parsing against an in-memory namespace-local tree.
    geometry = Element("Geometry_Parameters")
    container = SubElement(geometry, "Refined_Corner_Coordinates")
    for key, (lat, lon) in corners.items():
        prefix = {"UL": "upper_left", "UR": "upper_right", "LR": "lower_right", "LL": "lower_left"}[key]
        SubElement(container, prefix + "_latitude").text = str(lat)
        SubElement(container, prefix + "_longitude").text = str(lon)
    parsed_ring = ode._unwrap_ring({key: (lat, lon) for key, (lat, lon) in corners.items()})
    if len(parsed_ring) != 4:
        raise AssertionError("PDS4 corner ring was malformed")
    return f"span={span:.1f}°, ODE boxes={boxes}, unique candidates={len(products)}"


def check_raw_staging() -> str:
    raw_nac = ROOT / "data" / "raw" / "nac"
    if not raw_nac.is_dir():
        raise AssertionError(f"Reference staging directory missing: {raw_nac}")
    rasters = [p for p in raw_nac.rglob("*") if p.is_file() and p.suffix.lower() == ".img"]
    if not rasters:
        raise AssertionError("No staged NAC IMG rasters found")
    return f"{len(rasters)} IMG reference tiles staged"


def check_gsd_factors() -> str:
    reg = module_from_script("qa_register_images_gsd", "scripts/register_images.py")
    expected = {"ohrc": (2.0, 1.0), "tmc2": (1.0, 10.0), "iirs": (1.0, 160.0)}
    observed = {sensor: reg.gsd_pyramid_factors(sensor) for sensor in expected}
    if observed != expected:
        raise AssertionError(f"Scale factors differ: {observed}")
    try:
        reg.gsd_pyramid_factors("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown sensor should be rejected")
    return str(observed)


def check_preprocessing() -> str:
    import numpy as np

    reg = module_from_script("qa_register_images_preprocess", "scripts/register_images.py")
    image = np.full((320, 320), 13.0, np.float32)
    yy, xx = np.indices(image.shape)
    image += (xx / 320.0) * 18.0 + (yy / 320.0) * 7.0
    for x, y, radius in ((80, 90, 22), (225, 80, 16), (165, 235, 31)):
        image[(xx - x) ** 2 + (yy - y) ** 2 < radius ** 2] += 10.0
    gray = reg._to_gray8(image)
    shadow_like = reg._preprocess_for_matching(gray, "ohrc")
    nonpolar = reg._preprocess_for_matching(gray, "tmc2")
    if gray.dtype != np.uint8 or shadow_like.shape != image.shape or nonpolar.shape != image.shape:
        raise AssertionError("Preprocessing changed shape or failed uint8 normalization")
    if int(gray.min()) != 0 or int(gray.max()) != 255 or np.std(shadow_like) < 1:
        raise AssertionError("Percentile stretch/CLAHE/Sobel did not preserve useful contrast")
    return f"uint8 range={gray.min()}..{gray.max()}, CLAHE/Sobel std={np.std(shadow_like):.1f}"


def check_iirs_and_multimodal_signatures() -> str:
    import inspect
    import numpy as np
    from chandra_align.preprocessing import preprocess_iirs_raster, resize_to_common_ground_sample

    iirs_parameters = inspect.signature(preprocess_iirs_raster).parameters
    resize_parameters = inspect.signature(resize_to_common_ground_sample).parameters
    if "contrast_percentiles" not in iirs_parameters or "clahe_clip_limit" not in iirs_parameters:
        raise AssertionError("IIRS preprocessing does not expose configurable contrast/CLAHE controls")
    if "minimum_long_side" not in resize_parameters:
        raise AssertionError("GSD resizer does not accept minimum_long_side")
    cube_band = np.random.default_rng(125).normal(100, 15, (384, 384)).astype(np.float32)
    iirs = preprocess_iirs_raster(
        cube_band, contrast_percentiles=(1.0, 99.0), clahe_clip_limit=4.5
    ).processed_2d_raster
    ref = np.random.default_rng(5).integers(0, 256, (512, 512), dtype=np.uint8)
    resized, scale = resize_to_common_ground_sample(ref, 0.5, 10.0, 192)
    if iirs.shape != cube_band.shape or resized.shape[0] < 192 or resized.shape[1] < 192:
        raise AssertionError(f"Preprocess/resize shape contract failed: {iirs.shape}/{resized.shape}")
    return f"IIRS CLAHE output={iirs.shape}; common-GSD resized={resized.shape}, scale={scale:.3f}"


def check_affine_and_gate() -> str:
    import cv2
    import numpy as np

    reg = module_from_script("qa_register_images_affine", "scripts/register_images.py")
    rng = np.random.default_rng(26166)
    source = rng.uniform(0.0, 800.0, (36, 2)).astype(np.float32)
    angle, scale = math.radians(7.5), 1.013
    truth = np.array([
        [scale * math.cos(angle), -scale * math.sin(angle), 14.25],
        [scale * math.sin(angle), scale * math.cos(angle), -8.75],
    ], np.float32)
    target = source @ truth[:, :2].T + truth[:, 2]
    target[-8:] += rng.uniform(70.0, 180.0, (8, 2)).astype(np.float32)
    estimated, mask = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=1.0,
        maxIters=10000, confidence=0.999, refineIters=10,
    )
    inliers = mask.ravel().astype(bool)
    if estimated is None or int(inliers.sum()) < 24:
        raise AssertionError(f"Affine RANSAC did not reject injected outliers: {inliers.sum()} inliers")
    projected = source[inliers] @ estimated[:, :2].T + estimated[:, 2]
    residual = np.linalg.norm(projected - target[inliers], axis=1)
    median_error = float(np.median(residual))
    if median_error >= 0.25:
        raise AssertionError(f"Affine inlier median residual too large: {median_error:.4f}px")
    if not reg.is_registration_acceptable(estimated, 8):
        raise AssertionError("Eight-inlier result should pass the acceptance gate")
    if reg.is_registration_acceptable(estimated, 7):
        raise AssertionError("Seven-inlier result should be rejected")
    if reg.is_registration_acceptable(np.full((2, 3), np.nan), 20):
        raise AssertionError("Non-finite transform should be rejected")
    return f"inliers={int(inliers.sum())}/36, median reprojection={median_error:.4f}px, gate=8"


def check_app_partial_affine_contract() -> str:
    import numpy as np
    from chandra_align.metrics import get_sensor_pixel_scale
    from chandra_align.registration_transform import (
        estimate_partial_affine, homogeneous_affine, is_valid_registration,
    )

    if get_sensor_pixel_scale("TMC-2") != 5.0:
        raise AssertionError("TMC-2 default ground sample distance must be 5.0 m/px")

    rng = np.random.default_rng(26166)
    source = rng.uniform(0, 500, (40, 2)).astype(np.float32)
    truth = np.array([[0.999, -0.035, 12.5], [0.035, 0.999, -7.25]], np.float32)
    target = source @ truth[:, :2].T + truth[:, 2]
    target[-6:] += rng.normal(0, 100, (6, 2)).astype(np.float32)
    matrix, inliers = estimate_partial_affine(source, target)
    if matrix is None or matrix.shape != (2, 3):
        raise AssertionError("App transform helper did not return a 2x3 Partial Affine")
    count = int(inliers.sum())
    if count < 8 or not is_valid_registration(matrix, count):
        raise AssertionError(f"App transform safety gate rejected valid synthetic data ({count} inliers)")
    homography = homogeneous_affine(matrix)
    if homography.shape != (3, 3) or not np.allclose(homography[2], (0, 0, 1)):
        raise AssertionError("Affine homogeneous serialization is malformed")
    if is_valid_registration(np.full((2, 3), np.nan), 20) or is_valid_registration(matrix, 7):
        raise AssertionError("Invalid transform or sub-threshold inliers passed the gate")
    return f"model=partial_affine_4dof, inliers={count}/40, TMC-2=5.0 m/px"


def check_rift2_runtime() -> str:
    import warnings
    import cv2
    import numpy as np
    from chandra_align.matcher import RIFT2Matcher

    rng = np.random.default_rng(26166)
    image = rng.normal(112.0, 24.0, (384, 384)).clip(0, 255).astype(np.uint8)
    for _ in range(55):
        x, y = rng.integers(18, 366, size=2)
        radius = int(rng.integers(5, 22))
        cv2.circle(image, (int(x), int(y)), radius, int(rng.integers(20, 240)), 2)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Module 'pyfftw'.*")
        matcher = RIFT2Matcher(npt=2048)
        points_a, points_b = matcher.match(image, image)
    if len(points_a) < 4 or len(points_a) != len(points_b):
        raise AssertionError(f"RIFT2 self-match returned only {len(points_a)} correspondences")
    return f"RIFT2 self-match={len(points_a)} correspondences"


def _peak_working_set_mb() -> float:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.WinDLL("Psapi.dll")
        kernel = ctypes.WinDLL("Kernel32.dll")
        get_info = psapi.GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        if not get_info(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return counters.PeakWorkingSetSize / (1024 * 1024)

    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def check_iirs_band_and_memory() -> str:
    reg = module_from_script("qa_register_images_iirs", "scripts/register_images.py")
    src = ROOT / reg.SOURCES["iirs"]
    if not src.is_file():
        raise AssertionError(f"IIRS cube not staged at {src}")
    preset = reg.BENCHMARK_PRESETS["iirs"]
    values, width, height, band = reg._read_band(src, 4096, preset["band"])
    if band != 125 or values.ndim != 2 or values.size == 0:
        raise AssertionError(f"Expected a streamed 2-D IIRS band 125, got band={band}, shape={values.shape}")
    label = src.with_suffix(".xml")
    if label.is_file():
        import xml.etree.ElementTree as ET
        wanted = None
        for node in ET.parse(label).getroot().iter():
            if node.tag.rsplit("}", 1)[-1] == "Band_Bin":
                record = {child.tag.rsplit("}", 1)[-1]: (child.text or "").strip() for child in node}
                if record.get("band_number") == "125":
                    wanted = float(record["center_wavelength"])
                    break
        if wanted is None or not math.isclose(wanted, 2802.0, abs_tol=0.1):
            raise AssertionError(f"Unexpected band 125 wavelength in label: {wanted}")
    peak_mb = _peak_working_set_mb()
    if peak_mb >= 500.0:
        raise AssertionError(f"Peak working set exceeded target: {peak_mb:.1f} MiB")
    return f"band={band}, wavelength=2.802µm, preview={values.shape}, source={width}x{height}, peak process WS={peak_mb:.1f}MiB"


def check_registration_outputs() -> str:
    import cv2

    reg = module_from_script("qa_register_images_outputs", "scripts/register_images.py")
    details = []
    for sensor, preset in reg.BENCHMARK_PRESETS.items():
        path = ROOT / preset["output"]
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty output: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or tuple(image.shape[:2]) != tuple(preset["shape_hw"]):
            raise AssertionError(f"Unexpected output shape for {sensor}: {None if image is None else image.shape}")
        details.append(f"{sensor}={image.shape[1]}x{image.shape[0]}")
    return ", ".join(details)


def check_full_registration_run() -> str:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "register_images.py")],
        cwd=ROOT, text=True, capture_output=True, timeout=240,
    )
    output = result.stdout + result.stderr
    if result.returncode:
        raise AssertionError(f"register_images.py exited {result.returncode}:\n{output[-5000:]}")
    lines = [line.strip() for line in output.splitlines() if "RANSAC inliers" in line or "Registration complete" in line]
    return "; ".join(lines[-4:])


def _validate_live_result(result, case: str) -> str:
    import numpy as np
    import requests
    from PIL import Image

    if not isinstance(result, (list, tuple)) or len(result) != 7:
        raise AssertionError(f"{case}: expected 7 output values, got {type(result).__name__}/{len(result) if hasattr(result, '__len__') else '?'}")
    image, report, csv_file, json_file, geotiff_file, png_file, dossier = result
    if not isinstance(report, str) or not report.strip():
        raise AssertionError(f"{case}: missing report text")

    def payload_bytes(item):
        if isinstance(item, Image.Image):
            import io
            buff = io.BytesIO()
            item.save(buff, format="PNG")
            return buff.getvalue()
        if isinstance(item, np.ndarray):
            import cv2
            success, encoded = cv2.imencode(".png", item)
            return encoded.tobytes() if success else None
        if isinstance(item, dict):
            item = item.get("url") or item.get("path")
        if not item:
            return None
        value = str(item)
        candidate = Path(value)
        if candidate.is_file():
            return candidate.read_bytes()
        if value.startswith(("http://", "https://")):
            response = requests.get(value, timeout=30)
            response.raise_for_status()
            return response.content
        return None

    image_bytes = payload_bytes(image)
    if "REGISTRATION COMPLETE" in report:
        # Gradio may transcode the Image preview to WebP. The dedicated PNG
        # DownloadButton is the stable PNG artifact contract for /predict.
        png_bytes = payload_bytes(png_file)
        if not image_bytes:
            raise AssertionError(f"{case}: successful registration did not return an image preview")
        if not png_bytes or not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise AssertionError(f"{case}: successful registration did not return a valid PNG download")
        json_bytes = payload_bytes(json_file)
        if not json_bytes:
            raise AssertionError(f"{case}: successful registration did not return transform JSON")
        try:
            metadata = json.loads(json_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AssertionError(f"{case}: transform JSON is invalid: {exc}") from exc
        if not isinstance(metadata, dict):
            raise AssertionError(f"{case}: transform metadata is not an object")
        inlier_count = int(metadata.get("inlier_count", 0))
        model = metadata.get("transform_model")
        affine = metadata.get("partial_affine_matrix_2x3")
        if model != "partial_affine_4dof" or inlier_count < 8:
            raise AssertionError(f"{case}: invalid transform contract model={model!r}, inliers={inlier_count}")
        if not isinstance(affine, list) or len(affine) != 2 or any(not isinstance(row, list) or len(row) != 3 for row in affine):
            raise AssertionError(f"{case}: missing 2x3 Partial Affine matrix")
        return f"success, PNG={len(png_bytes)} bytes, model={model}, inliers={inlier_count}"
    if image is not None or any((csv_file, json_file, geotiff_file, png_file, dossier)):
        raise AssertionError(f"{case}: failure response leaked partial output artifacts")
    if not any(word in report.lower() for word in ("failed", "insufficient", "error")):
        raise AssertionError(f"{case}: unexpected safe-failure report: {report[:180]}")
    return f"safe rejection, report={report.splitlines()[0][:100]}"


def check_live_space() -> str:
    from gradio_client import Client, handle_file
    import cv2
    import numpy as np

    client = Client("revanshriyan/chandra-align", verbose=False)
    available = client.view_api(return_format="dict")
    if not available:
        raise AssertionError("Hugging Face Space published no Gradio API schema")
    temp = tempfile.TemporaryDirectory(prefix="chandra_live_qa_")
    tmp = Path(temp.name)

    # Hosted requests use seeded synthetic imagery only. Real local ISRO/NASA
    # rasters are tested offline; public uploads require separate authorization.
    rng = np.random.default_rng(26166)
    synthetic = rng.normal(112.0, 24.0, (640, 640)).clip(0, 255).astype(np.uint8)
    for _ in range(90):
        x, y = rng.integers(24, 616, size=2)
        radius = int(rng.integers(6, 30))
        cv2.circle(synthetic, (int(x), int(y)), radius, int(rng.integers(25, 235)), 2)
        cv2.circle(synthetic, (int(x), int(y)), max(2, radius // 2), int(rng.integers(25, 235)), 1)
    affine = cv2.getRotationMatrix2D((320, 320), 2.5, 1.01)
    affine[:, 2] += (6.0, -4.0)
    transformed = cv2.warpAffine(synthetic, affine, (640, 640), borderMode=cv2.BORDER_REFLECT_101)
    ref_path, sec_path, identical_path = tmp / "synthetic_ref.png", tmp / "synthetic_sec.png", tmp / "synthetic_identical.png"
    cv2.imwrite(str(ref_path), synthetic)
    cv2.imwrite(str(sec_path), transformed)
    cv2.imwrite(str(identical_path), synthetic)

    modality_modes = [
        ("OHRC", "OHRC", "Optical <-> Optical", 0.25),
        ("TMC-2", "TMC-2", "Optical <-> Optical", 5.0),
        ("IIRS", "IIRS", "Optical <-> Infrared", 80.0),
    ]
    observations = []
    for sensor, secondary, pair_mode, scale in modality_modes:
        args = (
            handle_file(str(identical_path)), handle_file(str(identical_path)), sensor, secondary,
            # Use identical synthetic pixel grids and disable scene-dependent
            # shadow masks; preprocessing and scale factors have dedicated
            # deterministic local checks above.
            pair_mode, False, scale, False, 3.0, False, False, 128.0, 50.0,
        )
        try:
            result = client.submit(*args, api_name="/predict").result(timeout=180)
            contract = _validate_live_result(result, sensor)
            if not contract.startswith("success"):
                raise AssertionError(f"{sensor}: identical-image sensor probe was rejected: {contract}")
            observations.append(f"{sensor} mode: {contract}")
        except Exception as exc:
            observations.append(f"{sensor} mode: request exception {type(exc).__name__}: {exc}")

    blank_path = tmp / "blank.png"
    cv2.imwrite(str(blank_path), np.zeros((256, 256), dtype=np.uint8))
    try:
        result = client.submit(
            handle_file(str(blank_path)), handle_file(str(blank_path)),
            "OHRC", "OHRC", "Optical <-> Optical", False,
            0.25, False, 3.0, False, False, 128.0, 50.0,
            api_name="/predict",
        ).result(timeout=180)
        observations.append(f"featureless pair: {_validate_live_result(result, 'blank pair')}")
    except Exception as exc:
        observations.append(f"featureless pair: request exception {type(exc).__name__}: {exc}")

    temp.cleanup()
    failed = [value for value in observations if "exception" in value]
    if failed:
        raise AssertionError("; ".join(observations))
    return "; ".join(observations)


def print_results() -> None:
    headers = ("MODULE NAME", "TARGET / SENSOR", "STATUS", "OBSERVED METRIC")
    rows = [headers, *CHECKS]
    widths = [max(len(str(row[i])) for row in rows) for i in range(4)]
    divider = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print("\n" + divider)
    for row_index, row in enumerate(rows):
        print("| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)) + " |")
        if row_index == 0:
            print(divider)
    print(divider)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Call the public Hugging Face Space sequentially")
    parser.add_argument("--skip-registration-run", action="store_true", help="Verify outputs without rerunning all real registrations")
    args = parser.parse_args(argv)

    run_check("ODE ingestion", "Polar south-footprint", check_polar_ode)
    run_check("ODE ingestion", "data/raw/nac staging", check_raw_staging)
    run_check("Registration", "OHRC/TMC-2/IIRS GSD pyramid", check_gsd_factors)
    run_check("Registration", "Percentile + CLAHE + OHRC Sobel", check_preprocessing)
    run_check("Preprocessing API", "IIRS contrast controls / minimum match extent", check_iirs_and_multimodal_signatures)
    run_check("Registration", "Partial Affine stability / 8-inlier gate", check_affine_and_gate)
    run_check("App transform", "/predict 4-DOF matrix / 8-inlier contract", check_app_partial_affine_contract)
    run_check("Matcher", "Vendored RIFT2 runtime dependency", check_rift2_runtime)
    run_check("Registration", "IIRS band 125 / peak working set", check_iirs_band_and_memory)
    if args.skip_registration_run:
        skip_check("Registration", "Full real-data run", "disabled by --skip-registration-run")
    else:
        run_check("Registration", "OHRC/TMC-2/IIRS full pipeline", check_full_registration_run)
    run_check("Registration", "Processed PNG outputs", check_registration_outputs)
    if args.live:
        run_check("Hugging Face Space", "3 modality requests + featureless pair", check_live_space)
    else:
        skip_check("Hugging Face Space", "Live endpoint", "pass --live to make hosted requests")
    print_results()
    failures = sum(row[2] == "FAIL" for row in CHECKS)
    skips = sum(row[2] == "SKIP" for row in CHECKS)
    print(f"\nChecks: {len(CHECKS) - skips - failures} passed, {failures} failed, {skips} skipped.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
