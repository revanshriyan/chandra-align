import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple, Union

from chandra_align.ingestion.pds4 import parse_pds4_metadata
from chandra_align.pipeline.router import evaluate_route


@dataclass
class BatchPairResult:
    pair_id: str
    source_pds4: str
    ref_pds4: str
    primary_strategy: str
    delta_azimuth_deg: float
    scale_ratio: float
    status: str
    error_msg: str = ""


def load_manifest(manifest_path: Union[str, Path], base_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Loads explicit (source, reference) XML pairs from CSV or JSON manifest.
    CSV format: source_xml,ref_xml
    JSON format: [{"source": "src.xml", "reference": "ref.xml"}, ...]
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    pairs: List[Tuple[Path, Path]] = []

    if manifest_path.suffix.lower() == ".json":
        try:
            with open(manifest_path, "r") as f:
                data = json.load(f)
                for item in data:
                    if "source" not in item or "reference" not in item:
                        raise ValueError(f"JSON manifest items must contain 'source' and 'reference' keys: {item}")
                    src = base_dir / item["source"]
                    ref = base_dir / item["reference"]
                    pairs.append((src, ref))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in manifest file '{manifest_path}': {e}")
    elif manifest_path.suffix.lower() == ".csv":
        with open(manifest_path, "r") as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader, 1):
                if not row or row[0].startswith("#") or row[0].lower() == "source":
                    continue
                if len(row) < 2:
                    raise ValueError(f"Row {row_idx} in CSV manifest has fewer than 2 columns: {row}")
                src = base_dir / row[0].strip()
                ref = base_dir / row[1].strip()
                pairs.append((src, ref))
    else:
        raise ValueError(f"Unsupported manifest format: {manifest_path.suffix}. Use .json or .csv")

    return pairs


def process_single_pair(pair_paths: Tuple[str, str]) -> BatchPairResult:
    """Processes a single explicit source/reference XML metadata pair with fault isolation."""
    src_path_str, ref_path_str = pair_paths
    src_xml = Path(src_path_str)
    ref_xml = Path(ref_path_str)
    pair_id = f"{src_xml.stem}_VS_{ref_xml.stem}"

    try:
        src_meta = parse_pds4_metadata(src_xml)
        ref_meta = parse_pds4_metadata(ref_xml)
        route = evaluate_route(src_meta, ref_meta)

        return BatchPairResult(
            pair_id=pair_id,
            source_pds4=src_xml.name,
            ref_pds4=ref_xml.name,
            primary_strategy=route.primary_strategy.value,
            delta_azimuth_deg=round(route.delta_azimuth_deg, 1),
            scale_ratio=round(route.scale_ratio, 2),
            status="SUCCESS",
        )
    except Exception as e:
        return BatchPairResult(
            pair_id=pair_id,
            source_pds4=src_xml.name if src_xml else "UNKNOWN",
            ref_pds4=ref_xml.name if ref_xml else "UNKNOWN",
            primary_strategy="FAILED",
            delta_azimuth_deg=0.0,
            scale_ratio=1.0,
            status="FAILED",
            error_msg=str(e),
        )


def run_batch_pipeline(
    manifest_path: Path,
    input_dir: Path,
    output_dir: Path,
    max_workers: int = 4,
    use_threads: bool = False
) -> List[BatchPairResult]:
    """Runs batch routing/registration for explicit pairs defined in manifest."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_manifest(manifest_path, input_dir)
    if not pairs:
        print(f"⚠️ No valid image pairs loaded from manifest: {manifest_path}")
        return []

    pair_str_tuples = [(str(src), str(ref)) for src, ref in pairs]

    results: List[BatchPairResult] = []
    print(f"🚀 Launching Batch Pipeline for {len(pairs)} manifest pairs using {max_workers} {'threads' if use_threads else 'processes'}...")

    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    with executor_class(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_pair, p): p for p in pair_str_tuples}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            status_symbol = "✓" if res.status == "SUCCESS" else "❌"
            print(f"  {status_symbol} [{res.pair_id}] Strategy={res.primary_strategy} | Status={res.status}")

    # Export Batch Reports
    json_path = output_dir / "batch_summary.json"
    md_path = output_dir / "batch_summary.md"

    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    md_lines = [
        "# 📦 Chandra-Align Batch Ingestion Summary\n",
        f"Processed Pairs: **{len(results)}**\n",
        "| Pair ID | Source Label | Reference Label | Strategy | ΔAzimuth | Scale | Status | Error Info |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: x.pair_id):
        status_icon = "✅" if r.status == "SUCCESS" else "❌"
        err_msg = r.error_msg if r.error_msg else "N/A"
        md_lines.append(
            f"| **{r.pair_id}** | `{r.source_pds4}` | `{r.ref_pds4}` | `{r.primary_strategy}` | "
            f"{r.delta_azimuth_deg}° | {r.scale_ratio}x | {status_icon} {r.status} | `{err_msg}` |"
        )

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n✅ Batch summary reports saved to:\n  - {json_path}\n  - {md_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Chandra-Align Manifest-Driven Batch CLI")
    parser.add_argument("--manifest", "-m", required=True, type=str, help="CSV or JSON file listing explicit (source, ref) pairs")
    parser.add_argument("--input-dir", "-i", required=True, type=str, help="Directory containing PDS4 XML labels")
    parser.add_argument("--output-dir", "-o", default="outputs/batch_run", type=str, help="Output directory for reports")
    parser.add_argument("--max-workers", "-w", default=4, type=int, help="Number of parallel execution workers")

    args = parser.parse_args()
    run_batch_pipeline(Path(args.manifest), Path(args.input_dir), Path(args.output_dir), max_workers=args.max_workers)


if __name__ == "__main__":
    main()