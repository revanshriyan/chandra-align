"""Run canonical Issue #1 fits and evaluate human-reviewed ground-truth points."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
FITS = ROOT / "results" / "issue02_fits"
OUT = ROOT / "results" / "issue13_gt_matcher_runs.json"
EVALUATOR = ROOT / "scripts" / "eval_ground_truth.py"
PAIRS = {
    "OHRC_pair": {
        "gt": ROOT / "data" / "ground_truth" / "gt_ohrc_pair_human_verified.csv",
        "ref": "ch2_ohr_nrp_20240425T1209509264_d_img_d18",
        "src": "ch2_ohr_nrp_20240425T1406019344_d_img_d18",
        "sensor": "OHRC", "scale": 0.26,
    },
    "TMC2_fore_nadir": {
        "gt": ROOT / "data" / "ground_truth" / "gt_tmc2_fore_nadir_human_verified.csv",
        "ref": "ch2_tmc_ncf_20231101T0125121344_d_img_d18",
        "src": "ch2_tmc_ncn_20231101T0125121377_d_img_d18",
        "sensor": "TMC-2", "scale": 4.47,
    },
}


def load_issue1_runner():
    path = ROOT / "results" / "run_issue01_gpu_validation.py"
    spec = importlib.util.spec_from_file_location("issue01_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def find_fullres(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit else [ROOT / "data" / "qa" / "fullres", ROOT.parent / "data" / "qa" / "fullres"]
    for candidate in candidates:
        if candidate is not None and (candidate / (PAIRS["OHRC_pair"]["ref"] + ".img")).exists():
            return candidate.resolve()
    raise FileNotFoundError("Canonical data/qa/fullres inputs are missing; pass --fullres explicitly")


def read_gt_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def load_real_pairs(fullres: Path, runner):
    import app

    loaded = {}
    ohrc = PAIRS["OHRC_pair"]
    loaded["OHRC_pair"] = (
        app.load_lunar_raster(str(fullres / f"{ohrc['ref']}.img"), max_dimension=4096),
        app.load_lunar_raster(str(fullres / f"{ohrc['src']}.img"), max_dimension=4096),
    )
    tmc = PAIRS["TMC2_fore_nadir"]
    def read_tmc(product: str, window: Window):
        path = fullres / f"{product}.img"
        with rasterio.open(path.with_suffix(".xml")) as dataset:
            raw = dataset.read(1, window=window, masked=True)
            raw = np.asarray(raw.astype(np.float32).filled(np.nan))
        return app.load_lunar_raster(raw, max_dimension=4096)
    loaded["TMC2_fore_nadir"] = (
        read_tmc(tmc["ref"], Window(361, 93328, 2048, 2048)),
        read_tmc(tmc["src"], Window(416, 101611, 2048, 2048)),
    )
    return loaded


def run(fullres: Path | None = None) -> list[dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this validation requires the Issue #1 GPU matcher setup")
    torch.cuda.init()
    torch.cuda.synchronize()
    fullres = find_fullres(fullres)
    runner = load_issue1_runner()
    pairs = load_real_pairs(fullres, runner)
    import app

    prior_table = ROOT / "results" / "table_issue02_ground_truth.csv"
    previous_synthetic = []
    previous_unmeasured_rift = []
    if prior_table.exists():
        with prior_table.open(newline="", encoding="utf-8-sig") as handle:
            prior_rows = list(csv.DictReader(handle))
            previous_synthetic = [row for row in prior_rows if row["pair"] == "synthetic_gentle"]
            previous_unmeasured_rift = [
                row for row in prior_rows
                if row["pair"] in PAIRS and row["matcher"] == "RIFT2"
            ]

    FITS.mkdir(parents=True, exist_ok=True)
    runs = []
    original_core = app._align_core
    for pair, (ref, src) in pairs.items():
        if ref is None or src is None:
            raise RuntimeError(f"Issue #1 loader returned an empty input for {pair}")
        metadata = PAIRS[pair]
        for matcher in ("LightGlue_ALIKED", "SIFT_RANSAC"):
            artifact = FITS / f"{pair}_{matcher}.npz"
            captured = {}

            def capture_core(*args, **kwargs):
                result = original_core(*args, **kwargs)
                captured["result"] = result
                return result

            app._align_core = capture_core
            try:
                row = runner.run_one(
                    pair, matcher, ref, src, metadata["sensor"], metadata["scale"],
                    artifact_path=artifact,
                )
            finally:
                app._align_core = original_core
            result = captured.get("result", {})
            heldout = result.get("heldout_error") or {}
            split_rmse = result.get("rmse_heldout_px")
            if not isinstance(split_rmse, (int, float)) or not np.isfinite(float(split_rmse)):
                split_rmse = ""
            status = str(result.get("status_code", row.get("gate", "UNMEASURED")))
            record = {
                "pair": pair, "matcher": matcher, "fit_artifact": str(artifact.relative_to(ROOT)),
                "ground_truth_csv": str(metadata["gt"].relative_to(ROOT)),
                "n_gt_points": read_gt_count(metadata["gt"]),
                "rmse_heldout_split": split_rmse,
                "n_heldout_points": int(heldout.get("n_check_points", 0)),
                "split_verdict": status,
                "registration_engine": result.get("registration_engine", result.get("engine_name", "UNKNOWN")),
                "fallback_triggered": bool((result.get("execution_diagnostics") or {}).get("fallback_triggered", False)),
                "fallback_reason": (result.get("execution_diagnostics") or {}).get("fallback_reason"),
                "fit_available": artifact.exists(),
                "fit_inliers": int(row.get("inliers", 0)),
                "matcher_correspondences": int(row.get("correspondences", 0)),
                "gate_message": row.get("gate", ""),
                "runtime_s": row.get("runtime_s", ""),
                "circularity_guard": "not yet evaluated",
            }
            if artifact.exists():
                completed = subprocess.run(
                    [sys.executable, str(EVALUATOR), pair, "--ground-truth", str(metadata["gt"]),
                     "--fit-artifact", str(artifact)],
                    cwd=ROOT, text=True, capture_output=True, check=False,
                )
                record["eval_stdout"] = completed.stdout.strip()
                record["eval_stderr"] = completed.stderr.strip()
                if completed.returncode == 0:
                    for line in completed.stdout.splitlines():
                        key, sep, value = line.partition(":")
                        if sep:
                            record[key.strip()] = value.strip()
                    record["circularity_guard"] = "PASS"
                else:
                    record["eval_error"] = completed.stderr.strip() or completed.stdout.strip()
                    record["circularity_guard"] = "FAIL"
            else:
                record["eval_error"] = "No valid fitted transform/inlier artifact was produced"
                record["circularity_guard"] = "NOT_RUN_NO_FIT"
            runs.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            torch.cuda.empty_cache()

    app._align_core = original_core
    try:
        fullres_record = fullres.relative_to(ROOT).as_posix()
    except ValueError:
        fullres_record = "../data/qa/fullres" if fullres == ROOT.parent / "data" / "qa" / "fullres" else "external fullres directory (not committed)"
    payload = {
        "device": torch.cuda.get_device_name(0),
        "fullres_root": fullres_record,
        "previous_synthetic_rows": previous_synthetic,
        "previous_unmeasured_rift_rows": previous_unmeasured_rift,
        "runs": runs,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fullres", type=Path, help="canonical PDS product directory; defaults to worktree or sibling data/qa/fullres")
    run(parser.parse_args().fullres)
