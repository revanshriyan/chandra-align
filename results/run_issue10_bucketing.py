"""Re-run Issue #3's 12 identical crop pairs with 4x4 detector bucketing."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from run_issue03_benchmark import load_manifest, require_cuda, run_one

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results" / "issue10_bucketing_runs.json"


def main() -> None:
    # Check CUDA before loading any pair or initializing either matcher.
    require_cuda()
    pairs = load_manifest(ROOT / "data" / "benchmark_pairs.csv")
    if len(pairs) != 12:
        raise RuntimeError(f"Expected the canonical 12-pair set, found {len(pairs)}")
    output = []
    pixel_scales = {"OHRC": 0.26, "TMC-2": 4.47}
    for pair in pairs:
        ref, src = pair.pop("reference_crop_array"), pair.pop("source_crop_array")
        for matcher in ("LightGlue_ALIKED", "SIFT_RANSAC"):
            for variant, enabled in (("before", "0"), ("after", "1")):
                os.environ["CHANDRA_GRID_BUCKETING"] = enabled
                print(f"Running {variant} {pair['pair_id']} / {matcher}; shared arrays {ref.shape}", flush=True)
                row = run_one(pair, matcher, ref, src, pair["sensor"], pixel_scales[pair["sensor"]])
                row["variant"] = variant
                output.append(row)
                print({key: value for key, value in row.items() if key != "notes"}, flush=True)
                torch.cuda.empty_cache()
    os.environ.pop("CHANDRA_GRID_BUCKETING", None)
    RAW_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {RAW_PATH}")


if __name__ == "__main__":
    main()
