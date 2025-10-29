import argparse
import glob
import json
import os
from typing import Dict, List


def _load_summary_file(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    averaged_scores = payload.get("averaged_scores", {})
    result = {
        "encoder": payload.get("encoder", os.path.basename(path)),
        "num_samples": payload.get("num_samples", 0),
    }
    result.update({metric: float(value) for metric, value in averaged_scores.items()})
    return result


def _compute_summary_from_full(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        return {"encoder": os.path.basename(path), "num_samples": 0}

    totals: Dict[str, float] = {}
    for entry in records:
        encoder = entry.get("encoder")
        for key, value in entry.items():
            if key in {"encoder", "target_path", "ref_path"}:
                continue
            totals.setdefault(key, 0.0)
            totals[key] += float(value)

    num_samples = len(records)
    averages = {k: v / num_samples for k, v in totals.items()}

    result = {"encoder": encoder or os.path.basename(path), "num_samples": num_samples}
    result.update(averages)
    return result


def gather_results(root: str) -> List[Dict]:
    summary_pattern = os.path.join(root, "benchmark_*_summary.json")
    summary_files = glob.glob(summary_pattern)

    if summary_files:
        return sorted((_load_summary_file(path) for path in summary_files), key=lambda x: x["encoder"])

    full_pattern = os.path.join(root, "benchmark_*.json")
    full_files = [path for path in glob.glob(full_pattern) if not path.endswith("_summary.json")]

    if not full_files:
        return []

    return sorted((_compute_summary_from_full(path) for path in full_files), key=lambda x: x["encoder"])


def format_results(results: List[Dict]) -> str:
    if not results:
        return "No retrieval benchmark files found."

    metric_keys = [key for key in results[0].keys() if key not in {"encoder", "num_samples"}]
    header = ["Encoder", "Samples"] + metric_keys

    rows: List[List[str]] = []
    for record in results:
        row = [record["encoder"], str(record.get("num_samples", 0))]
        for metric in metric_keys:
            value = record.get(metric, float("nan"))
            row.append(f"{value:.4f}")
        rows.append(row)

    column_widths = [
        max(len(row[idx]) for row in ([header] + rows)) for idx in range(len(header))
    ]

    def _format_row(values: List[str]) -> str:
        return " | ".join(values[idx].ljust(column_widths[idx]) for idx in range(len(values)))

    separator = "-+-".join("-" * width for width in column_widths)

    lines = [_format_row(header), separator]
    lines.extend(_format_row(row) for row in rows)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare retrieval benchmarks across encoders.")
    parser.add_argument(
        "--root",
        type=str,
        default="output/retrieval",
        help="Directory containing retrieval benchmark JSON files.",
    )
    args = parser.parse_args()

    results = gather_results(args.root)
    print(format_results(results))


if __name__ == "__main__":
    main()
