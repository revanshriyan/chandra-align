"""Headless safety checks for the Gradio pitch-demo contract.

This suite intentionally avoids importing the full app stack so it can run on a
headless developer machine without OpenCV, Gradio, or ZeroGPU installed.
"""

from __future__ import annotations

import ast
import contextlib
import io
import os
import py_compile
import re
import sys
import traceback
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
APP_SOURCE = APP_PATH.read_text(encoding="utf-8")
APP_TREE = ast.parse(APP_SOURCE, filename=str(APP_PATH))
RESULTS: list[tuple[str, str, str]] = []


def record(name: str, metric: str, check) -> None:
    try:
        detail = check()
        text = str(detail if detail is not None else metric)
        RESULTS.append((name, "SKIP" if text.startswith("SKIP:") else "PASS", text))
    except unittest.SkipTest as exc:
        RESULTS.append((name, "SKIP", str(exc)))
    except Exception as exc:  # report all cases before returning a failure code
        RESULTS.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))


def function_node(name: str) -> ast.FunctionDef:
    for node in APP_TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"app.py is missing function {name}")


def execute_function(name: str, namespace: dict):
    node = function_node(name)
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace[name]


class FakeCV2:
    IMREAD_UNCHANGED = -1
    INTER_AREA = 3

    def __init__(self):
        self.paths: list[str] = []

    def imread(self, path, _flags):
        self.paths.append(path)
        return np.arange(48, dtype=np.uint16).reshape(6, 8)

    @staticmethod
    def resize(image, dimensions, interpolation=None):
        del interpolation
        width, height = dimensions
        return np.zeros((height, width), dtype=image.dtype)


def test_path_inputs() -> str:
    fake_cv = FakeCV2()
    namespace = {
        "np": np,
        "Image": Image,
        "os": os,
        "cv2": fake_cv,
        "MAX_IMAGE_DIMENSION": 4096,
    }
    reader = execute_function("_read_grayscale_image", namespace)
    cases = [
        Path("nested") / "reference.png",
        {"path": str(Path("nested") / "secondary.png")},
    ]
    for value in cases:
        result = reader(value)
        assert result.shape == (6, 8)
        assert result.dtype == np.uint8
    expected = [os.fspath(cases[0]), cases[1]["path"]]
    assert fake_cv.paths == expected, (fake_cv.paths, expected)
    return "str/Path/dict file objects decoded; uint16 normalized to uint8"


def test_callback_arity() -> str:
    namespace = {
        "np": np,
        "sys": sys,
        "re": re,
        "traceback": traceback,
        "MAX_IMAGE_DIMENSION": 4096,
        "IIRS_MAX_IMAGE_DIMENSION": 12000,
        "_failed_judge_metrics_summary": lambda *_args: {
            "status_code": "FAILED", "status_message": "REJECTED"
        },
        "_safe_rejection_output_tuple": lambda summary, status_message=None, **_kwargs: tuple(
            [None, None, None, None, status_message, summary, "", None, None, None, None, None, None, None]
        ),
        "_read_grayscale_image": lambda *_args: (_ for _ in ()).throw(ValueError("bad image")),
    }
    callback = execute_function("process_alignment", namespace)
    with contextlib.redirect_stderr(io.StringIO()):
        missing = callback(None, None)
        invalid = callback("bad-reference.png", "bad-secondary.png")
    assert len(missing) == 14, len(missing)
    assert len(invalid) == 14, len(invalid)
    return "missing and invalid inputs both return 14 outputs without UnboundLocalError"


def test_safe_rejection_arity() -> str:
    namespace = {
        "np": np,
        "Image": Image,
        "sys": sys,
        "traceback": traceback,
        "_failed_judge_metrics_summary": lambda: {"status_code": "FAILED"},
        "_rejected_output_tuple": lambda *_args, **_kwargs: tuple(range(12)),
    }
    execute_function("_minimal_rejection_output_tuple", namespace)
    helper = execute_function("_safe_rejection_output_tuple", namespace)
    with contextlib.redirect_stderr(io.StringIO()):
        outputs = helper({}, "Rejected in headless test")
    assert len(outputs) == 14, len(outputs)
    assert "N/A" in outputs[6]
    return "banner failure fallback returns 14 safe values and masked telemetry"


def test_wrapper_catches_gpu_cpu_and_append_errors() -> str:
    calls = []

    def fail_gpu(*_args):
        calls.append("gpu")
        raise RuntimeError("simulated ZeroGPU failure")

    def fail_cpu(*_args):
        calls.append("cpu")
        raise RuntimeError("simulated CPU failure")

    def emergency(message, summary=None):
        calls.append("fallback")
        return (message, summary) + (None,) * 12

    namespace = {
        "sys": sys,
        "traceback": traceback,
        "run_alignment_on_gpu": fail_gpu,
        "_run_alignment_core": fail_cpu,
        "_minimal_rejection_output_tuple": emergency,
        "_append_execution_status": lambda *_args: (_ for _ in ()).throw(
            RuntimeError("simulated report formatter failure")
        ),
        "_zerogpu_runtime_enabled": lambda: False,
        "GPU_EXECUTION_STATUS": "GPU",
        "CPU_FALLBACK_STATUS": "CPU",
    }
    wrapper = execute_function("process_wrapper", namespace)
    with contextlib.redirect_stderr(io.StringIO()):
        result = wrapper(*([None] * 13))
    assert len(result) == 14, len(result)
    assert calls == ["gpu", "cpu", "fallback", "fallback"], calls
    return "GPU, CPU, and output formatting exceptions are contained at the Gradio boundary"


def test_assets_and_examples() -> str:
    names = (
        "nac_reference_ohrc.png", "ohrc_secondary.png",
        "nac_reference_tmc2.png", "tmc2_secondary.png",
        "nac_reference_iirs.png", "iirs_band125_secondary.png",
        "synthetic_groundtruth_reference.png", "synthetic_groundtruth_secondary.png",
    )
    paths = [ROOT / "docs" / "assets" / "examples" / name for name in names]
    for path in paths:
        assert path.is_file(), f"missing example asset: {path}"
    pointer_paths = []
    for path in paths:
        with path.open("rb") as stream:
            if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
                pointer_paths.append(path.name)
    if pointer_paths:
        raise unittest.SkipTest(
            "git-lfs assets are pointers (" + ", ".join(pointer_paths)
            + "); run `git lfs pull` to inspect images"
        )
    for path in paths:
        with Image.open(path) as image:
            image.verify()
    with Image.open(paths[-2]) as reference, Image.open(paths[-1]) as secondary:
        ref_array = np.asarray(reference.convert("L"))
        sec_array = np.asarray(secondary.convert("L"))
    assert ref_array.shape == sec_array.shape
    assert not np.array_equal(ref_array, sec_array), "synthetic pair must be distinct"
    assert all(name in APP_SOURCE for name in names)
    assert 'label="Select Lunar Data Pair Presets"' in APP_SOURCE
    assert "cache_examples=False" in APP_SOURCE
    assert "examples_per_page=4" in APP_SOURCE
    assert "interface.launch(ssr_mode=False)" in APP_SOURCE
    return "8 tracked-format assets valid; synthetic pair distinct; Examples caching disabled"


def test_fixed_contract_and_primary_ratio() -> str:
    node = function_node("process_alignment")
    tuple_returns = [
        ret for ret in ast.walk(node)
        if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Tuple)
    ]
    assert tuple_returns, "no tuple return paths found"
    assert all(len(ret.value.elts) == 14 for ret in tuple_returns)
    assert "for ratio in (0.75, 0.80, 0.85)" in APP_SOURCE
    assert re.search(r"first\.distance\s*<\s*ratio\s*\*\s*second\.distance", APP_SOURCE)
    return f"{len(tuple_returns)} literal tuple path(s) have 14 values; Lowe ratio 0.75"


def test_warp_is_initialized_before_visualization() -> str:
    node = function_node("_align_core")
    warp_assignment = next(
        (child for child in ast.walk(node)
         if isinstance(child, ast.Assign)
         and any(isinstance(target, ast.Name) and target.id == "warped_sec"
                 for target in child.targets)),
        None,
    )
    assert warp_assignment is not None, "_align_core no longer assigns warped_sec"
    reads = [
        child.lineno for child in ast.walk(node)
        if isinstance(child, ast.Name) and child.id == "warped_sec"
        and isinstance(child.ctx, ast.Load)
    ]
    assert reads, "_align_core does not consume the warped raster"
    first_read = min(reads)
    assert warp_assignment.lineno < first_read, (
        f"warped_sec read on line {first_read} before assignment on line {warp_assignment.lineno}"
    )
    return f"warped_sec initialized on line {warp_assignment.lineno} before its first read on line {first_read}"


def test_compile() -> str:
    targets = [
        APP_PATH,
        ROOT / "chandra_align" / "preprocessing" / "general.py",
        ROOT / "chandra_align" / "preprocessing" / "__init__.py",
        ROOT / "chandra_align" / "matcher" / "__init__.py",
        ROOT / "chandra_align" / "matching" / "deep_matchers.py",
    ]
    for path in targets:
        py_compile.compile(str(path), doraise=True)
    return f"compiled {len(targets)} modified modules"


def main() -> int:
    checks = (
        ("String/path image decoding", "image path dtypes", test_path_inputs),
        ("Callback exception arity", "14 outputs", test_callback_arity),
        ("Rejection fallback arity", "14 outputs", test_safe_rejection_arity),
        ("Gradio callback master guard", "14 outputs", test_wrapper_catches_gpu_cpu_and_append_errors),
        ("Native example assets", "8 valid PNGs", test_assets_and_examples),
        ("Tuple and ratio contract", "14 tuple / 0.75 ratio", test_fixed_contract_and_primary_ratio),
        ("Warped raster initialization order", "assign before read", test_warp_is_initialized_before_visualization),
        ("Python compilation", "5 modules", test_compile),
    )
    for name, metric, check in checks:
        record(name, metric, check)
    print("[CHECK] | [STATUS] | [DETAIL]")
    print("-" * 100)
    for name, status, detail in RESULTS:
        print(f"{name} | {status} | {detail}")
    failed = sum(status == "FAIL" for _name, status, _detail in RESULTS)
    skipped = sum(status == "SKIP" for _name, status, _detail in RESULTS)
    print(f"\n{len(RESULTS) - failed - skipped} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
