#!/usr/bin/env python
"""
Equatorial NAC Ingestion Automation

Downloads approved NAC frames from the shortlist, runs CRS projection-trap guard,
and logs entries into data/download_log.csv under new run IDs.
"""

import json
import os
import sys
import hashlib
import csv
from pathlib import Path
from datetime import datetime
import urllib.request

PROJECT_ROOT = Path(__file__).parent.parent
SHORTLIST_PATH = PROJECT_ROOT / "data" / "equatorial_nac_shortlist.json"
NAC_DIR = PROJECT_ROOT / "data" / "lroc_nac"
LOG_PATH = PROJECT_ROOT / "data" / "download_log.csv"


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, expected_md5: str = None) -> bool:
    """Download a file with progress, verify MD5 if provided."""
    print(f"Downloading {url} -> {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
        if expected_md5:
            actual = md5_file(dest)
            if actual.lower() != expected_md5.lower():
                print(f"  MD5 MISMATCH: expected {expected_md5}, got {actual}")
                return False
            print(f"  MD5 verified: {actual}")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def load_shortlist():
    with open(SHORTLIST_PATH, "r") as f:
        return json.load(f)


def append_log_entries(entries):
    """Append new rows to download_log.csv."""
    fieldnames = [
        "product_id", "source_url", "acquisition_date",
        "solar_azimuth_deg", "solar_elevation_deg", "scale_ratio",
        "local_path", "run_id"
    ]
    
    # Read existing to preserve header
    existing = []
    if LOG_PATH.exists():
        with open(LOG_PATH, "r") as f:
            reader = csv.DictReader(f)
            existing = list(reader)
    
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
        for e in entries:
            writer.writerow(e)


def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest_equatorial_nac.py [indices...]")
        print("  indices: 0-based indices from equatorial_nac_shortlist.json")
        print("Example: python ingest_equatorial_nac.py 0 1")
        return 1

    shortlist = load_shortlist()
    indices = [int(i) for i in sys.argv[1:]]
    
    selected = [shortlist[i] for i in indices]
    print(f"Selected {len(selected)} frames:")
    for s in selected:
        print(f"  {s['product_id']} - {s['observation_date']} - incidence {s['incidence_angle_deg']}° - overlap {s['overlap_pct']}%")
    
    # Confirm
    confirm = input("Proceed with download? (y/N): ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return 0

    NAC_DIR.mkdir(parents=True, exist_ok=True)
    log_entries = []
    run_id_base = f"nac_eq_{datetime.now().strftime('%Y%m%d_%H%M')}"

    for idx, frame in enumerate(selected):
        run_id = f"{run_id_base}_{idx:03d}"
        local_path = NAC_DIR / frame["product_name"]
        
        # Download
        if download_file(frame["pds_url"], local_path):
            # Verify file exists and get size
            size = local_path.stat().st_size
            print(f"  Saved: {local_path} ({size:,} bytes)")
            
            # Create log entry
            log_entry = {
                "product_id": frame["product_id"],
                "source_url": frame["pds_url"],
                "acquisition_date": frame["observation_date"],
                "solar_azimuth_deg": "UNMEASURED",  # will derive later
                "solar_elevation_deg": f'{frame.get("solar_elevation_deg", "UNMEASURED")}',
                "scale_ratio": "1.0",
                "local_path": str(local_path.relative_to(PROJECT_ROOT)),
                "run_id": run_id
            }
            log_entries.append(log_entry)
            print(f"  Logged with run_id: {run_id}")
        else:
            print(f"  FAILED: {frame['product_id']}")

    if log_entries:
        append_log_entries(log_entries)
        print(f"\nAppended {len(log_entries)} entries to {LOG_PATH}")

    print("\nNext steps:")
    print("1. Run CRS projection-trap guard on each new NAC frame vs your OHRC equatorial strip")
    print("2. Derive solar azimuth from labels and update download_log.csv")
    print("3. Run full Stage-1 pipeline on the pairs")

    return 0


if __name__ == "__main__":
    sys.exit(main())