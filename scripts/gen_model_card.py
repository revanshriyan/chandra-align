"""Generate docs/model-card.md from the canonical results table.

Run: python scripts/gen_model_card.py
The model card is a generated file; edit this script, not the markdown.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANONICAL = ROOT / "results" / "table_canonical_v1.csv"
OUT = ROOT / "docs" / "model-card.md"


def load():
    with CANONICAL.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    rows = load()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = []
    add = lines.append

    add("# CHANDRA-ALIGN — Honest Benchmark Card")
    add("")
    add(f"*Generated {stamp} UTC from `results/table_canonical_v1.csv` by*")
    add("`scripts/gen_model_card.py`. *Numbers are measured, not claimed.*")
    add("")
    add("## What this pipeline is")
    add("")
    add("A lunar image registration pipeline for Chandrayaan-2 OHRC, TMC-2, and IIRS")
    add("products. Matcher cascade: LightGlue/ALIKED (GPU, validated path) → SIFT +")
    add("Brute-Force RANSAC (CPU fallback). A fail-closed spatial gate classifies every")
    add("result as ACCEPT (sub-pixel), COARSE ADVISORY (regional fit), or REJECT /")
    add("DEGENERATE_FAILURE. Rejected fits mask transform telemetry.")
    add("")
    add("## Measured results (canonical table v1)")
    add("")
    add("| Benchmark | Pair | Matcher | Corr | Inliers | RMSE (px) | Held-out (px) | GT RMSE (px) | Verdict |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for bench in ["gpu_validation", "ground_truth", "iirs_first_contact"]:
        for row in rows:
            if row["benchmark"] != bench:
                continue
            rmse = row["rmse_px"][:6] if row["rmse_px"] else "—"
            held = row["rmse_heldout_px"][:6] if row["rmse_heldout_px"] else "—"
            gt = row["gt_rmse_px"][:6] if row["gt_rmse_px"] else "—"
            verdict = row["verdict"].split(":")[0].split("(")[0].strip()
            add(f"| {bench} | {row['pair']} | {row['matcher']} | {row['correspondences']} "
                f"| {row['inliers']} | {rmse} | {held} | {gt} | {verdict} |")
    add("")
    add("### 12-pair south-polar crop benchmark")
    add("")
    for row in rows:
        if row["benchmark"] == "crop_benchmark_12pair":
            add(f"- **{row['matcher']}**: {row['verdict']}")
    add("")
    add("## What the numbers mean — and don't")
    add("")
    add("- **Synthetic pairs ACCEPT sub-pixel** (0.37–0.40 px). The pipeline works when")
    add("  correspondences exist.")
    add("- **Real OHRC/TMC-2 pairs are COARSE at best** (1.5–2.2 px in-sample; held-out")
    add("  1.5–2.6 px). Independent human landmarks disagree with the fitted transforms")
    add("  (GT RMSE 5.7–27.4 px): the pipeline finds *a* consistent fit, not necessarily")
    add("  *the* true one. Treat real-pair outputs as advisories, not measurements.")
    add("- **IIRS↔TMC-2 cross-modal: no successful registration yet.** Both matchers hit")
    add("  DEGENERATE_FAILURE on the verified-overlap crop.")
    add("- **RIFT2 is non-functional** in this codebase (0–1 correspondences per pair).")
    add("  It is vendored but disabled by default.")
    add("- Held-out RMSE is computed on 3–25 check points depending on the pair; small")
    add("  held-out sets are noisy — see the per-table `n_heldout_points` columns.")
    add("")
    add("## Provenance")
    add("")
    add("- GPU validation: NVIDIA RTX 5070.")
    add("- Full per-run tables: `results/table_issue01_gpu_validation.csv`,")
    add("  `results/table_issue02_ground_truth.csv`, `results/table_issue03_benchmark.csv`,")
    add("  `results/table_issue05_baselines.csv`, `results/table_issue12_iirs.csv`.")
    add("- Canonical rollup: `results/table_canonical_v1.csv` (+ `.meta.json`).")
    add("- README numbers are machine-checked by `scripts/check_readme_numbers.py`.")
    add("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
