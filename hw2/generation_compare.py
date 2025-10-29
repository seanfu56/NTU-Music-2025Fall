import argparse
import glob
import json
import os
from typing import Dict, List


def _load_summary_file(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    averaged_scores = payload.get("averaged_scores", {})
    record = {
        "encoder": payload.get("encoder", os.path.basename(path)),
        "num_samples": payload.get("num_samples", 0),
    }
    record.update({metric: float(value) for metric, value in averaged_scores.items()})
    return record


def _compute_summary_from_full(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        return {"encoder": os.path.basename(path), "num_samples": 0}

    totals: Dict[str, float] = {}
    for entry in records:
        for key, value in entry.items():
            if key in {"encoder", "target_path", "generation_path"}:
                continue
            totals.setdefault(key, 0.0)
            totals[key] += float(value)

    num_samples = len(records)
    averages = {k: v / num_samples for k, v in totals.items()}

    encoder = records[0].get("encoder", os.path.basename(path))
    result = {"encoder": encoder, "num_samples": num_samples}
    result.update(averages)
    return result


def gather_results(root: str) -> List[Dict]:
    summary_pattern = os.path.join(root, "benchmark_*_summary.json")
    summary_files = glob.glob(summary_pattern)

    if summary_files:
        return sorted((_load_summary_file(path) for path in summary_files), key=lambda x: x["encoder"])

    full_pattern = os.path.join(root, "benchmark_*.json")
    full_files = glob.glob(full_pattern)

    if not full_files:
        return []

    return sorted((_compute_summary_from_full(path) for path in full_files), key=lambda x: x["encoder"])


def format_results(results: List[Dict]) -> str:
    if not results:
        return "No benchmark files found."

    metrics = [key for key in results[0].keys() if key not in {"encoder", "num_samples"}]
    header = ["Generator", "Samples"] + metrics

    rows = []
    for record in results:
        row = [
            record["encoder"],
            str(record.get("num_samples", 0)),
        ]
        for metric in metrics:
            value = record.get(metric, float("nan"))
            row.append(f"{value:.4f}")
        rows.append(row)

    col_widths = [max(len(row[i]) for row in ([header] + rows)) for i in range(len(header))]

    def format_row(values):
        return " | ".join(value.ljust(col_widths[idx]) for idx, value in enumerate(values))

    lines = [format_row(header), "-+-".join("-" * width for width in col_widths)]
    lines.extend(format_row(row) for row in rows)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare generation benchmarks across generators.")
    parser.add_argument(
        "--root",
        type=str,
        default="output/generation",
        help="Directory containing benchmark JSON files.",
    )
    args = parser.parse_args()

    results = gather_results(args.root)
    print(format_results(results))


if __name__ == "__main__":
    main()
