#!/usr/bin/env python3
"""Re-run ONE arm's cells in a fresh process (sys.path collision workaround).

Root cause (proven by diagnose_frontend_collision.py): anymatch,
matchanything and xoftr each vendor a `src/` package and do
sys.path.insert(0, <root>). Whichever root is imported first wins
sys.modules['src']; the other two frontends then resolve src.* against
the wrong root and fail instantly (exception swallowed -> (None, None)
in 0.0s). In the full run anymatch was called first per pair, so its
src won and every matchanything/xoftr cell failed. In a fresh process
running ONLY one of these arms, that arm's src wins and it works (xoftr
alone on N4 reproduced Oct-15's n_corr=504 exactly).

Usage: python3 run_arm_cells.py <arm>   # arm in {xoftr, matchanything}
Writes arm_cells_<arm>_2026-10-26.csv + rows_<arm>.jsonl in OUT_DIR.
Merge into the main table afterwards (replace that arm's rows).
"""
import csv
import importlib.util
import json
import os
import sys

OUT_DIR = os.path.expanduser("~/workspace/guardon_19pair_2026-10-26")
REPO = os.path.expanduser("~/workspace/chandra-align")
RUNNER = os.path.join(OUT_DIR, "rerun_arm_19pair_2026-10-26.py")

sys.path.insert(0, REPO)
spec = importlib.util.spec_from_file_location("arm_runner", RUNNER)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main():
    arm = sys.argv[1]
    assert arm in ("xoftr", "matchanything"), arm
    from chandra_align.metrics.spatial_support import strip_guard_enabled
    assert strip_guard_enabled(), "guard must be default-ON"
    pairs = mod.load_pairs()
    ref = mod.load_reference()
    csv_path = os.path.join(OUT_DIR, f"arm_cells_{arm}_2026-10-26.csv")
    jsonl_path = os.path.join(OUT_DIR, f"rows_{arm}.jsonl")
    with open(csv_path, "w", newline="") as fcsv, open(jsonl_path, "w") as fjson:
        writer = csv.DictWriter(fcsv, fieldnames=mod.CSV_FIELDS)
        writer.writeheader()
        for pair, iirs, tmc in pairs:
            print(f"{pair} x {arm} ...", flush=True)
            row, detail = mod.run_cell(pair, arm, iirs, tmc, ref.get((pair, arm)))
            writer.writerow(row)
            fcsv.flush()
            fjson.write(json.dumps(detail) + "\n")
            fjson.flush()
            print(f"  -> {row['verdict']} rmse={row['rmse_gate_px']} "
                  f"inliers={row['inliers']} n_corr={row['n_corr']} "
                  f"match_s={row['match_s']} guard={row['guard_fired']} "
                  f"changed={row['verdict_changed_vs_oct15']} "
                  f"attr={row['attribution']}", flush=True)
    print(f"done: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
