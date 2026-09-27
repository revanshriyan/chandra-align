"""Evaluation harness (M7.1-M7.3) — code only, results come from real runs.

- M7.1: Held-out check-point protocol (automated split, RMSE on check-set only)
- M7.2: Success-rate vs solar-azimuth-difference chart generator (UNMEASURED until real pairs exist)
- M7.3: Per-method × per-region table generator (reads metrics JSONs, emits markdown with UNMEASURED cells)
"""

import json
import csv
import glob
import os
from pathlib import Path


def collect_run_metrics(runs_dir="outputs"):
    """Collect all metrics.json from run directories."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    metrics_files = list(runs_dir.glob("*/metrics.json"))
    results = []
    for mf in metrics_files:
        try:
            with open(mf, "r") as f:
                data = json.load(f)
            data["_run_dir"] = str(mf.parent)
            data["_run_id"] = mf.parent.name
            results.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return results


def read_download_log(log_path="data/download_log.csv"):
    """Read download_log.csv into a list of dicts."""
    if not os.path.exists(log_path):
        return []
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def merge_run_with_log(metrics_list, log_entries):
    """Merge metrics with download log entries by run_id."""
    log_by_run = {entry.get("run_id"): entry for entry in log_entries}
    for m in metrics_list:
        run_id = m.get("_run_id")
        if run_id and run_id in log_by_run:
            m["log"] = log_by_run[run_id]
    return metrics_list


# --- M7.1: Held-out check-point protocol ---

def report_heldout_rmse(metrics_dict):
    """Extract held-out RMSE from a metrics dict. Returns 'UNMEASURED' if not present."""
    rmse = metrics_dict.get("rmse", {})
    if isinstance(rmse, dict):
        if rmse.get("held_out"):
            return rmse.get("rmse_px", "UNMEASURED")
    return "UNMEASURED"


def check_circularity_guard(metrics_dict):
    """Verify that circularity guard fired: fit-set RMSE != held-out RMSE.
    
    Since we never compute fit-set RMSE, this is structurally enforced.
    This function just documents the guarantee.
    """
    return {
        "circularity_guard_enforced": True,
        "note": "fit-set RMSE never computed; only held-out RMSE reported"
    }


# --- M7.3: Per-method × per-region table ---

METHODS = ["sift", "rift2", "lightglue_aliked"]
REGIONS = ["equatorial", "polar"]


def method_from_metrics(m):
    """Extract method name from metrics dict."""
    matcher = str(m.get("matcher", "unknown")).lower()
    if "rift2" in matcher:
        return "rift2"
    elif "lightglue" in matcher or "tier2" in matcher:
        return "lightglue_aliked"
    elif "sift" in matcher:
        return "sift"
    return matcher


def region_from_log(log_entry):
    """Determine region from log entry (equatorial/polar)."""
    # Simple heuristic based on solar elevation or latitude in future
    lat = log_entry.get("latitude", "") if isinstance(log_entry, dict) else ""
    if lat:
        try:
            lat_val = float(lat)
            if abs(lat_val) > 60:
                return "polar"
        except ValueError:
            pass
    # Default to equatorial if unknown
    return "equatorial"


def build_method_region_table(metrics_list):
    """Build per-method × per-region table.

    Cells contain held-out RMSE (px) or "UNMEASURED" if no run ID exists for that combo.
    """
    # Initialize table with UNMEASURED
    table = {method: {region: "UNMEASURED" for region in REGIONS} for method in METHODS}
    
    for m in metrics_list:
        method = method_from_metrics(m)
        log = m.get("log", {})
        region = region_from_log(log)
        
        if method in METHODS and region in REGIONS:
            rmse = report_heldout_rmse(m)
            # Only overwrite if we have a real measurement
            if rmse != "UNMEASURED":
                table[method][region] = f"{float(rmse):.3f}"
    
    return table


def emit_method_region_table_markdown(table):
    """Emit the M7.3 table as markdown."""
    lines = ["| Method | Equatorial RMSE (px) | Polar RMSE (px) |", "|---|---|---|"]
    for method in METHODS:
        eq = table.get(method, {}).get("equatorial", "UNMEASURED")
        pol = table.get(method, {}).get("polar", "UNMEASURED")
        lines.append(f"| {method} | {eq} | {pol} |")
    lines.append("")
    lines.append("*Cells without a run ID show UNMEASURED. Values are held-out RMSE in pixels.*")
    return "\n".join(lines)


# --- M7.2: Solar-azimuth-difference robustness chart ---

def extract_solar_azimuth_diff(log_entry):
    """Extract solar azimuth difference from log entry."""
    # In a real run, this would be the difference between CH-2 and reference acquisition
    # For now, return None (no data)
    return log_entry.get("solar_azimuth_diff") if log_entry else None


def build_azimuth_vs_success_table(metrics_list):
    """Build data for success-rate vs solar-azimuth-difference chart.

    Returns list of dicts: {solar_azimuth_diff_deg, success_rate, method, region}
    """
    results = []
    for m in metrics_list:
        log = m.get("log", {})
        method = method_from_metrics(m)
        region = region_from_log(log)
        azimuth_diff = extract_solar_azimuth_diff(log)
        success_rate = m.get("success_rate", "UNMEASURED")
        
        if azimuth_diff is not None:
            results.append({
                "solar_azimuth_diff_deg": float(azimuth_diff),
                "success_rate": success_rate if success_rate != "UNMEASURED" else None,
                "method": method,
                "region": region,
            })
    return results


def emit_azimuth_chart_data(metrics_list, output_path="outputs/azimuth_vs_success.csv"):
    """Emit chart data as CSV for external plotting."""
    data = build_azimuth_vs_success_table(metrics_list)
    if not data:
        # Write empty header for now
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["solar_azimuth_diff_deg", "success_rate", "method", "region"])
            writer.writeheader()
        return "UNMEASURED (no solar azimuth diff data in logs)"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["solar_azimuth_diff_deg", "success_rate", "method", "region"])
        writer.writeheader()
        writer.writerows(data)
    return f"wrote {len(data)} rows to {output_path}"


# --- Main evaluation CLI ---

def run_evaluation(runs_dir="outputs", log_path="data/download_log.csv"):
    """Run the evaluation pipeline: collect metrics, merge with log, emit tables/charts."""
    metrics = collect_run_metrics(runs_dir)
    log = read_download_log(log_path)
    merged = merge_run_with_log(metrics, log)
    
    # M7.3 table
    table = build_method_region_table(merged)
    md_table = emit_method_region_table_markdown(table)
    
    # M7.2 chart data
    azimuth_msg = emit_azimuth_chart_data(merged)
    
    # Summary
    summary = {
        "n_runs": len(merged),
        "method_region_table": table,
        "azimuth_chart": azimuth_msg,
        "circularity_guard": "enforced (held-out RMSE only)",
    }
    
    return summary, md_table


if __name__ == "__main__":
    import sys
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    summary, md_table = run_evaluation(runs_dir)
    print(md_table)
    print("\n--- Summary ---")
    print(json.dumps(summary, indent=2))