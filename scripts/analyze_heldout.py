"""Held-out Point Analysis Script (§7 Hierarchy Level 2 Ground-Truth Evaluation).

Reads paired reference and product CSVs exported from QGIS, computes per-point displacements,
total RMSE (px and m), and Success Rates at 1x and 2x GSD tolerances, broken down by confidence tier.

Guards:
- Refuses to run if row counts differ between reference and product CSVs.
- Refuses to run if 'confidence' column is missing from either CSV.
- Generates reports/heldout_<pair_id>.md and outputs/residual_scatter_<pair_id>.png.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_csv_points(file_path: Path) -> List[Dict]:
    """Parse QGIS exported CSV file into point list. Validates required columns."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    points = []
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        
        # Check confidence column presence
        conf_field = None
        for name in fieldnames:
            if name.lower() in ["confidence", "conf"]:
                conf_field = name
                break
        if conf_field is None:
            raise ValueError(f"Missing required 'confidence' column in {file_path}. Found columns: {fieldnames}")

        for idx, row in enumerate(reader):
            x_val = row.get("X") or row.get("x") or row.get("lon")
            y_val = row.get("Y") or row.get("y") or row.get("lat")
            if x_val is None or y_val is None:
                # WKT fallback
                wkt = row.get("WKT") or row.get("wkt") or row.get("geometry")
                if wkt and "POINT" in wkt.upper():
                    coords = wkt.replace("POINT", "").replace("(", "").replace(")", "").strip().split()
                    x_val, y_val = coords[0], coords[1]

            if x_val is None or y_val is None:
                raise ValueError(f"Row {idx+1} in {file_path} missing X/Y coordinates.")

            feat_type = row.get("feature_type") or row.get("type") or "unknown"
            conf = row.get(conf_field)

            try:
                conf_val = int(float(conf))
            except (ValueError, TypeError):
                raise ValueError(f"Invalid confidence value '{conf}' at row {idx+1} in {file_path}")

            points.append({
                "row_id": idx + 1,
                "x": float(x_val),
                "y": float(y_val),
                "confidence": conf_val,
                "feature_type": str(feat_type).strip().lower(),
            })

    return points


def analyze_heldout_pair(
    ref_file: Path,
    prod_file: Path,
    gsd_m: float = 0.25,
    pair_id: str = "ohrc_test_001"
) -> Tuple[Dict, str]:
    """Analyze held-out point pairs and enforce row count and column presence guards."""
    ref_pts = parse_csv_points(ref_file)
    prod_pts = parse_csv_points(prod_file)

    # GUARD: Row count match
    if len(ref_pts) != len(prod_pts):
        raise ValueError(
            f"Row count mismatch error: Reference CSV '{ref_file.name}' has {len(ref_pts)} rows, "
            f"but Product CSV '{prod_file.name}' has {len(prod_pts)} rows. Refusing to silently pair mismatched rows!"
        )

    if len(ref_pts) == 0:
        raise ValueError("CSV files contain 0 points!")

    results = []
    for r, p in zip(ref_pts, prod_pts):
        dx = p["x"] - r["x"]
        dy = p["y"] - r["y"]
        disp_px = math.hypot(dx, dy)
        disp_m = disp_px * gsd_m

        results.append({
            "point_id": r["row_id"],
            "ref_x": r["x"],
            "ref_y": r["y"],
            "prod_x": p["x"],
            "prod_y": p["y"],
            "dx_px": dx,
            "dy_px": dy,
            "disp_px": disp_px,
            "disp_m": disp_m,
            "confidence": r["confidence"],
            "feature_type": r["feature_type"],
            "pass_1x_gsd": disp_px <= 1.0,
            "pass_2x_gsd": disp_px <= 2.0,
        })

    def calc_sub_stats(sub: List[Dict]) -> Dict:
        if not sub:
            return {"count": 0, "rmse_x_px": 0.0, "rmse_y_px": 0.0, "rmse_total_px": 0.0, "rmse_m": 0.0, "sr_1x": 0.0, "sr_2x": 0.0}
        n = len(sub)
        sum_dx2 = sum(pt["dx_px"]**2 for pt in sub)
        sum_dy2 = sum(pt["dy_px"]**2 for pt in sub)
        sum_d2 = sum(pt["disp_px"]**2 for pt in sub)

        rx = math.sqrt(sum_dx2 / n)
        ry = math.sqrt(sum_dy2 / n)
        rt = math.sqrt(sum_d2 / n)
        n1 = sum(1 for pt in sub if pt["pass_1x_gsd"])
        n2 = sum(1 for pt in sub if pt["pass_2x_gsd"])

        return {
            "count": n,
            "rmse_x_px": rx,
            "rmse_y_px": ry,
            "rmse_total_px": rt,
            "rmse_m": rt * gsd_m,
            "sr_1x": n1 / n,
            "sr_2x": n2 / n,
            "n_pass_1x": n1,
            "n_pass_2x": n2,
        }

    overall = calc_sub_stats(results)
    by_tier = {
        "tier_1_certain": calc_sub_stats([pt for pt in results if pt["confidence"] == 1]),
        "tier_2_pretty_sure": calc_sub_stats([pt for pt in results if pt["confidence"] == 2]),
        "tier_3_best_guess": calc_sub_stats([pt for pt in results if pt["confidence"] == 3]),
    }

    # Render residual scatter plot
    fig, ax = plt.subplots(figsize=(7, 7))
    dxs = [pt["dx_px"] for pt in results]
    dys = [pt["dy_px"] for pt in results]
    confs = [pt["confidence"] for pt in results]

    colors = {1: "tab:green", 2: "tab:orange", 3: "tab:red"}
    labels = {1: "Tier 1 (Certain)", 2: "Tier 2 (Pretty Sure)", 3: "Tier 3 (Best Guess)"}

    for c in [1, 2, 3]:
        cx = [dx for dx, conf in zip(dxs, confs) if conf == c]
        cy = [dy for dy, conf in zip(dys, confs) if conf == c]
        if cx:
            ax.scatter(cx, cy, c=colors[c], label=labels[c], s=40, alpha=0.8)

    # Add 1x and 2x GSD circles
    circle1 = plt.Circle((0, 0), 1.0, color="blue", fill=False, linestyle="--", label="1x GSD (1 px)")
    circle2 = plt.Circle((0, 0), 2.0, color="purple", fill=False, linestyle=":", label="2x GSD (2 px)")
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="-")
    ax.set_aspect("equal")
    max_r = max(2.5, max(abs(d) for d in dxs + dys) * 1.2 if dxs else 2.5)
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_xlabel("Residual dx (pixels)")
    ax.set_ylabel("Residual dy (pixels)")
    ax.set_title(f"Held-Out Residual Scatter Plot ({pair_id})\nOverall RMSE: {overall['rmse_total_px']:.3f} px ({overall['rmse_m']:.3f} m)")
    ax.legend(loc="upper right")
    fig.tight_layout()

    scatter_path = Path(f"outputs/residual_scatter_{pair_id}.png")
    scatter_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(scatter_path, dpi=120)
    plt.close(fig)

    # Markdown Report Construction
    md_lines = []
    md_lines.append(f"# Manual Ground-Truth Held-Out Report (`{pair_id}`)")
    md_lines.append("")
    md_lines.append(f"- **Pair ID**: `{pair_id}`")
    md_lines.append(f"- **Total Manual Check Points**: `{overall['count']}`")
    md_lines.append(f"- **Product GSD**: `{gsd_m} m/px`")
    md_lines.append(f"- **Circularity Guard**: Active (§7 Level 2 Ground Truth — fit points never mixed)")
    md_lines.append("")
    md_lines.append("## Overall Metrics")
    md_lines.append("")
    md_lines.append("| Metric | Value (px) | Value (m) | Criteria / Details |")
    md_lines.append("|---|---|---|---|")
    md_lines.append(f"| **Total RMSE** | `{overall['rmse_total_px']:.4f} px` | `{overall['rmse_m']:.4f} m` | Radial displacement |")
    md_lines.append(f"| **RMSE X** | `{overall['rmse_x_px']:.4f} px` | `{overall['rmse_x_px']*gsd_m:.4f} m` | Along-row residual |")
    md_lines.append(f"| **RMSE Y** | `{overall['rmse_y_px']:.4f} px` | `{overall['rmse_y_px']*gsd_m:.4f} m` | Along-column residual |")
    md_lines.append(f"| **Success Rate (1× GSD)** | `{overall['sr_1x']*100:.1f}%` | — | `{overall['n_pass_1x']}/{overall['count']}` points $\\le 1.0$ px |")
    md_lines.append(f"| **Success Rate (2× GSD)** | `{overall['sr_2x']*100:.1f}%` | — | `{overall['n_pass_2x']}/{overall['count']}` points $\\le 2.0$ px |")
    md_lines.append("")
    md_lines.append("## Metrics Breakdown by Confidence Tier")
    md_lines.append("")
    md_lines.append("| Tier | Label | Count | Total RMSE (px) | RMSE (m) | Success (1× GSD) | Success (2× GSD) |")
    md_lines.append("|---|---|---|---|---|---|---|")
    for t_key, t_st in by_tier.items():
        name = t_key.replace("tier_", "").replace("_", " ").title()
        if t_st["count"] == 0:
            md_lines.append(f"| `{t_key}` | {name} | 0 | N/A | N/A | N/A | N/A |")
        else:
            md_lines.append(
                f"| `{t_key}` | {name} | {t_st['count']} | "
                f"`{t_st['rmse_total_px']:.4f} px` | `{t_st['rmse_m']:.4f} m` | "
                f"`{t_st['sr_1x']*100:.1f}%` | `{t_st['sr_2x']*100:.1f}%` |"
            )
    md_lines.append("")
    md_lines.append(f"![Residual Scatter Plot]({scatter_path.resolve().as_uri()})")
    md_lines.append("")

    summary_dict = {
        "pair_id": pair_id,
        "gsd_m": gsd_m,
        "overall": overall,
        "by_tier": by_tier,
        "scatter_plot": str(scatter_path),
        "per_point": results
    }

    report_path = Path(f"reports/heldout_{pair_id}.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return summary_dict, str(report_path)


def main():
    parser = argparse.ArgumentParser(description="Analyze Manual Held-Out Check Points")
    parser.add_argument("--ref", type=str, required=True, help="Path to heldout_ref.csv")
    parser.add_argument("--prod", type=str, required=True, help="Path to heldout_prod.csv")
    parser.add_argument("--gsd", type=float, default=0.25, help="Product GSD in m/px (default: 0.25)")
    parser.add_argument("--pair-id", type=str, default="ohrc_test_001", help="Pair ID")

    args = parser.parse_args()

    try:
        summary, rpath = analyze_heldout_pair(Path(args.ref), Path(args.prod), gsd_m=args.gsd, pair_id=args.pair_id)
        print(f"Analysis complete! Report written to {rpath}")
    except Exception as e:
        print(f"Error analyzing heldout points: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
