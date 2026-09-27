"""Manual Ground-Truth (§7 Hierarchy Level 2) Held-Out Point Evaluation Harness.

Computes:
- Per-point displacement (in pixels and meters)
- RMSE (x, y, total)
- Success Rate with tolerance 1x GSD and 2x GSD (<= 1 px and <= 2 px)
- Confidence tier breakdown (Tier 1: Certain, Tier 2: Pretty Sure, Tier 3: Best Guess)
- Feature type breakdown (crater_rim, boulder, crater_center, ridge)

Note: NEVER mix held-out check points into any fit/transformation calculation.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_points_from_csv(file_path: Path) -> List[Dict]:
    """Load points from a QGIS-exported CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    points = []
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # QGIS exports geometry as X, Y or WKT, or custom x, y columns
            x_val = row.get("X") or row.get("x") or row.get("lon") or row.get("X84")
            y_val = row.get("Y") or row.get("y") or row.get("lat") or row.get("Y84")
            
            if x_val is None or y_val is None:
                # Check if WKT geometry field exists (e.g. "POINT (x y)")
                wkt = row.get("WKT") or row.get("wkt") or row.get("geometry")
                if wkt and "POINT" in wkt.upper():
                    try:
                        coords = wkt.replace("POINT", "").replace("(", "").replace(")", "").strip().split()
                        x_val, y_val = coords[0], coords[1]
                    except Exception:
                        pass

            if x_val is None or y_val is None:
                raise ValueError(f"Could not parse X/Y coordinates at row {idx+1} in {file_path}. Row keys: {list(row.keys())}")

            feat_type = row.get("feature_type") or row.get("type") or "unknown"
            conf = row.get("confidence") or row.get("conf") or "1"
            notes = row.get("notes") or row.get("note") or ""

            points.append({
                "index": idx,
                "x": float(x_val),
                "y": float(y_val),
                "feature_type": str(feat_type).strip().lower(),
                "confidence": int(float(conf)),
                "notes": str(notes).strip()
            })
    return points


def evaluate_manual_pairs(
    ref_points: List[Dict],
    prod_points: List[Dict],
    gsd_m: float = 0.25
) -> Dict:
    """Compute per-point displacement, RMSE, and Success Rates from paired reference/product CSVs."""
    if len(ref_points) != len(prod_points):
        raise ValueError(
            f"Row count mismatch: reference CSV has {len(ref_points)} rows, "
            f"product CSV has {len(prod_points)} rows. Both files must pair by row order!"
        )
    if len(ref_points) == 0:
        raise ValueError("No points found to evaluate!")

    per_point_results = []
    
    for i, (ref, prod) in enumerate(zip(ref_points, prod_points)):
        dx = prod["x"] - ref["x"]
        dy = prod["y"] - ref["y"]
        disp_px = math.hypot(dx, dy)
        disp_m = disp_px * gsd_m
        
        # Use confidence and feature_type from reference point (or fallback to product)
        conf = ref.get("confidence", prod.get("confidence", 1))
        feat_type = ref.get("feature_type", prod.get("feature_type", "unknown"))

        per_point_results.append({
            "point_id": i + 1,
            "ref_x": ref["x"],
            "ref_y": ref["y"],
            "prod_x": prod["x"],
            "prod_y": prod["y"],
            "dx_px": dx,
            "dy_px": dy,
            "disp_px": disp_px,
            "disp_m": disp_m,
            "confidence": conf,
            "feature_type": feat_type,
            "pass_1x_gsd": disp_px <= 1.0,
            "pass_2x_gsd": disp_px <= 2.0,
        })

    def compute_stats_for_subset(subset: List[Dict]) -> Dict:
        if not subset:
            return {
                "count": 0,
                "rmse_x_px": None,
                "rmse_y_px": None,
                "rmse_total_px": None,
                "rmse_m": None,
                "success_rate_1x_gsd": None,
                "success_rate_2x_gsd": None,
            }
        
        n = len(subset)
        sum_dx2 = sum(p["dx_px"] ** 2 for p in subset)
        sum_dy2 = sum(p["dy_px"] ** 2 for p in subset)
        sum_disp2 = sum(p["disp_px"] ** 2 for p in subset)

        rmse_x = math.sqrt(sum_dx2 / n)
        rmse_y = math.sqrt(sum_dy2 / n)
        rmse_total = math.sqrt(sum_disp2 / n)
        rmse_m = rmse_total * gsd_m

        n_pass_1x = sum(1 for p in subset if p["pass_1x_gsd"])
        n_pass_2x = sum(1 for p in subset if p["pass_2x_gsd"])

        return {
            "count": n,
            "rmse_x_px": round(rmse_x, 4),
            "rmse_y_px": round(rmse_y, 4),
            "rmse_total_px": round(rmse_total, 4),
            "rmse_m": round(rmse_m, 4),
            "success_rate_1x_gsd": round(n_pass_1x / n, 4),
            "success_rate_2x_gsd": round(n_pass_2x / n, 4),
            "n_pass_1x_gsd": n_pass_1x,
            "n_pass_2x_gsd": n_pass_2x,
        }

    # Overall stats
    overall_stats = compute_stats_for_subset(per_point_results)

    # By Confidence Tier (1, 2, 3)
    by_confidence = {}
    for tier in [1, 2, 3]:
        sub = [p for p in per_point_results if p["confidence"] == tier]
        tier_names = {1: "1_certain", 2: "2_pretty_sure", 3: "3_best_guess"}
        by_confidence[tier_names[tier]] = compute_stats_for_subset(sub)

    # By Feature Type
    feature_types = set(p["feature_type"] for p in per_point_results)
    by_feature = {}
    for ft in sorted(feature_types):
        sub = [p for p in per_point_results if p["feature_type"] == ft]
        by_feature[ft] = compute_stats_for_subset(sub)

    return {
        "hierarchy_level": 2,
        "evaluation_type": "Manual Ground-Truth Check Set",
        "gsd_m": gsd_m,
        "total_points": len(per_point_results),
        "overall": overall_stats,
        "by_confidence_tier": by_confidence,
        "by_feature_type": by_feature,
        "per_point_details": per_point_results,
    }


def generate_markdown_report(report: Dict) -> str:
    """Generate clean Markdown table output for human review."""
    lines = []
    lines.append("# Manual Ground-Truth Held-Out Point Evaluation (§7 Hierarchy Level 2)")
    lines.append("")
    lines.append(f"- **Total Check Points**: `{report['total_points']}`")
    lines.append(f"- **GSD (Ground Sample Distance)**: `{report['gsd_m']} m/px`")
    lines.append(f"- **Evaluation Protocol**: Row-paired manual features on Reference vs Registered Product")
    lines.append("")
    
    # Overall summary table
    lines.append("## Overall Metrics")
    lines.append("")
    ov = report["overall"]
    lines.append("| Metric | Value (px) | Value (meters) | Notes / Criteria |")
    lines.append("|---|---|---|---|")
    lines.append(f"| **RMSE (Total)** | `{ov['rmse_total_px']} px` | `{ov['rmse_m']} m` | Radial displacement |")
    lines.append(f"| **RMSE (X)** | `{ov['rmse_x_px']} px` | `{round(ov['rmse_x_px']*report['gsd_m'], 4)} m` | Easting / Along-row |")
    lines.append(f"| **RMSE (Y)** | `{ov['rmse_y_px']} px` | `{round(ov['rmse_y_px']*report['gsd_m'], 4)} m` | Northing / Along-col |")
    lines.append(f"| **Success Rate (1× GSD)** | `{ov['success_rate_1x_gsd']*100:.1f}%` | — | `{ov['n_pass_1x_gsd']}/{ov['count']}` points $\\le 1.0$ px |")
    lines.append(f"| **Success Rate (2× GSD)** | `{ov['success_rate_2x_gsd']*100:.1f}%` | — | `{ov['n_pass_2x_gsd']}/{ov['count']}` points $\\le 2.0$ px |")
    lines.append("")

    # Confidence Tier breakdown table
    lines.append("## Metrics by Confidence Tier")
    lines.append("")
    lines.append("| Tier | Label | Count | RMSE (px) | RMSE (m) | Success (1× GSD) | Success (2× GSD) |")
    lines.append("|---|---|---|---|---|---|---|")
    for tier_key, stats in report["by_confidence_tier"].items():
        if stats["count"] == 0:
            lines.append(f"| `{tier_key}` | — | 0 | N/A | N/A | N/A | N/A |")
        else:
            lines.append(
                f"| `{tier_key}` | `{tier_key.split('_', 1)[1].replace('_', ' ').title()}` | {stats['count']} | "
                f"`{stats['rmse_total_px']} px` | `{stats['rmse_m']} m` | "
                f"`{stats['success_rate_1x_gsd']*100:.1f}%` | `{stats['success_rate_2x_gsd']*100:.1f}%` |"
            )
    lines.append("")

    # Feature Type breakdown table
    lines.append("## Metrics by Feature Type")
    lines.append("")
    lines.append("| Feature Type | Count | RMSE (px) | RMSE (m) | Success (1× GSD) | Success (2× GSD) |")
    lines.append("|---|---|---|---|---|---|")
    for ft_key, stats in report["by_feature_type"].items():
        lines.append(
            f"| `{ft_key}` | {stats['count']} | "
            f"`{stats['rmse_total_px']} px` | `{stats['rmse_m']} m` | "
            f"`{stats['success_rate_1x_gsd']*100:.1f}%` | `{stats['success_rate_2x_gsd']*100:.1f}%` |"
        )
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> **Circularity Guard**: These ground-truth points were picked manually and evaluated exclusively on the final registered product. They were NEVER included in the registration fitting set.")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Manual Ground-Truth Held-Out Point Evaluation")
    parser.add_argument("--ref", type=str, help="Path to heldout_ref.csv (reference image picks)")
    parser.add_argument("--prod", type=str, help="Path to heldout_prod.csv (registered product picks)")
    parser.add_argument("--single", type=str, help="Path to single paired CSV (e.g. ref_x, ref_y, prod_x, prod_y)")
    parser.add_argument("--gsd", type=float, default=0.25, help="Ground Sample Distance in meters per pixel (default: 0.25)")
    parser.add_argument("--out-json", type=str, default="outputs/heldout_metrics.json", help="Path to output JSON")
    parser.add_argument("--out-md", type=str, default="outputs/heldout_report.md", help="Path to output Markdown")

    args = parser.parse_args()

    if args.single:
        # Load single paired file
        single_path = Path(args.single)
        if not single_path.exists():
            print(f"Error: Single CSV file '{single_path}' does not exist.")
            sys.exit(1)
        ref_pts, prod_pts = [], []
        with open(single_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for idx, r in enumerate(reader):
                ref_pts.append({
                    "x": float(r["ref_x"]), "y": float(r["ref_y"]),
                    "confidence": int(float(r.get("confidence", 1))),
                    "feature_type": r.get("feature_type", "unknown")
                })
                prod_pts.append({
                    "x": float(r["prod_x"]), "y": float(r["prod_y"]),
                    "confidence": int(float(r.get("confidence", 1))),
                    "feature_type": r.get("feature_type", "unknown")
                })
    elif args.ref and args.prod:
        ref_pts = load_points_from_csv(Path(args.ref))
        prod_pts = load_points_from_csv(Path(args.prod))
    else:
        print("Error: Must specify either (--ref AND --prod) or --single")
        sys.exit(1)

    res = evaluate_manual_pairs(ref_pts, prod_pts, gsd_m=args.gsd)
    md = generate_markdown_report(res)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)

    print(md)
    print(f"\nSaved metrics to {args.out_json} and report to {args.out_md}")


if __name__ == "__main__":
    main()
