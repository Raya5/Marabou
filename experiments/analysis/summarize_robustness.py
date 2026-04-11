#!/usr/bin/env python3
"""
experiments/analysis/summarize_robustness.py

Summarize robustness results into a single JSON file and a paper-style stats file.

Expected input layout:
  experiments/results/robustness/full/point_<idx>/
    baseline/summary.json
    incremental/summary.json

Outputs:
  experiments/results/analysis/robustness_summary.json
  experiments/results/analysis/stats/robustness.txt
"""

import json
from pathlib import Path


INPUT_ROOT = Path("experiments/results/robustness/full")
OUTPUT_JSON = Path("experiments/results/analysis/robustness_summary.json")
OUTPUT_STATS = Path("experiments/results/analysis/stats/robustness.txt")

PRECISION = 0.001
TOTAL_TIMEOUT = 1200.0


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_point_idx(point_dir: Path) -> int:
    name = point_dir.name
    assert name.startswith("point_"), f"Unexpected point directory name: {name}"
    return int(name[len("point_"):])


def load_mode_summary(mode_dir: Path):
    summary_path = mode_dir / "summary.json"
    if not summary_path.exists():
        return None
    return load_json(summary_path)


def mean(xs):
    assert len(xs) > 0, "Cannot take mean of empty list"
    return sum(xs) / len(xs)


def is_solved(entry):
    return (
        entry["status"] != "SkippedMisclassified"
        and float(entry["time_sec"]) < TOTAL_TIMEOUT
        and float(entry["precision_width"]) < PRECISION
    )


def write_stats(records):
    OUTPUT_STATS.parent.mkdir(parents=True, exist_ok=True)

    baseline_times = [float(r["baseline"]["time_sec"]) for r in records]
    incremental_times = [float(r["incremental"]["time_sec"]) for r in records]

    baseline_solved = sum(1 for r in records if is_solved(r["baseline"]))
    incremental_solved = sum(1 for r in records if is_solved(r["incremental"]))

    incremental_tightenings = [
        float(r["incremental"]["total_incremental_tightenings"]) for r in records
    ]
    incremental_conflicts = [
        float(r["incremental"]["total_conflicts_recorded"]) for r in records
    ]

    baseline_time_avg = mean(baseline_times)
    incremental_time_avg = mean(incremental_times)
    speedup = baseline_time_avg / incremental_time_avg

    with OUTPUT_STATS.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("\\;\\textbf{Method}\\; &\n")
        f.write("\\;\\textbf{Time (s)}\\;  &\n")
        f.write("\\;\\textbf{Solved}\\;  &\n")
        f.write("\\;\\textbf{Propagations}\\;  &\n")
        f.write("\\;\\textbf{Conflicts} \\;\\\\\n")
        f.write("\\midrule\n")
        f.write(
            f"Non-incremental &\n"
            f"{baseline_time_avg:.1f} &\n"
            f"{baseline_solved} &\n"
            f"-- &\n"
            f"-- \\\\\n"
        )
        f.write(
            f"Incremental &\n"
            f"\\bf{{{incremental_time_avg:.1f}}} &\n"
            f"\\bf{{{incremental_solved}}} &\n"
            f"{mean(incremental_tightenings):.1f} &\n"
            f"{mean(incremental_conflicts):.1f}\\\\\n"
        )
        f.write("\\midrule\n")
        f.write(
            f"\\textbf{{Speedup}} &\n"
            f"\\textbf{{{speedup:.2f}$\\times$}} &\n"
            f"-- &\n"
            f"-- &\n"
            f"-- \\\\\n"
        )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Robustness radius evaluation on MNIST.}\n")
        f.write("\\label{tab:robustness-radius-results}\n")
        f.write("\\end{table}\n")


def main():
    assert INPUT_ROOT.exists(), f"Missing input root: {INPUT_ROOT}"

    point_dirs = sorted(
        [p for p in INPUT_ROOT.iterdir() if p.is_dir() and p.name.startswith("point_")],
        key=parse_point_idx,
    )

    included = []
    skipped = []
    records = []

    for point_dir in point_dirs:
        point_idx = parse_point_idx(point_dir)

        baseline = load_mode_summary(point_dir / "baseline")
        incremental = load_mode_summary(point_dir / "incremental")

        if baseline is None or incremental is None:
            skipped.append(point_idx)
            continue

        if (
            baseline.get("status") == "SkippedMisclassified"
            or incremental.get("status") == "SkippedMisclassified"
        ):
            skipped.append(point_idx)
            continue

        record = {
            "point": point_idx,
            "baseline": baseline,
            "incremental": incremental,
        }
        records.append(record)
        included.append(point_idx)

    output = {
        "input_root": str(INPUT_ROOT),
        "num_points_included": len(included),
        "num_points_skipped": len(skipped),
        "included_points": included,
        "skipped_points": skipped,
        "points": records,
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    assert len(records) > 0, "No included robustness records to summarize"
    write_stats(records)

    print(f"[summarize_robustness] done: {OUTPUT_JSON}")
    print(f"[summarize_robustness] done: {OUTPUT_STATS}")


if __name__ == "__main__":
    main()