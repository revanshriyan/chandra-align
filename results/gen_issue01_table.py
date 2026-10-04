"""Print a Markdown view of table_issue01_gpu_validation.csv."""

import csv
from pathlib import Path


CSV_PATH = Path(__file__).with_name("table_issue01_gpu_validation.csv")
HEADERS = (
    ("pair", "Pair"),
    ("matcher", "Matcher"),
    ("correspondences", "Correspondences"),
    ("rmse_px", "RMSE (px)"),
    ("inliers", "Inliers"),
    ("entropy", "Entropy"),
    ("quadrants", "Quadrants"),
    ("gate", "Gate"),
    ("runtime_s", "Runtime (s)"),
)


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print("| " + " | ".join(label for _, label in HEADERS) + " |")
    print("| " + " | ".join("---" for _ in HEADERS) + " |")
    for row in rows:
        values = []
        for key, _ in HEADERS:
            value = row[key]
            if key == "rmse_px" and not value:
                value = "N/A"
            elif key in {"rmse_px", "entropy", "runtime_s"} and value:
                value = f"{float(value):.4f}"
            values.append(value.replace("|", "\\|"))
        print("| " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()
