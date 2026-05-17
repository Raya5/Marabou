#!/usr/bin/env python3
"""
experiments/analysis/summarize_explainability.py

Summarize explainability full-run results into a single JSON file and a paper-style stats file.

Expected input layout:
experiments/results/explainability/full/point_<idx>/
  baseline/
    summary.json
    progress.csv
  incremental/
    summary.json
    progress.csv

Important:
- Some points may only have baseline results, because incremental was skipped
  when baseline encountered no conflicts.
- Those points are skipped from the final analysis output.

Outputs:
  experiments/results/analysis/explainability_summary.json
  experiments/results/analysis/stats/explainability.txt
"""

import csv
import json
from pathlib import Path
import os


RESULTS_ROOT = Path("experiments/results/explainability/full")
if not RESULTS_ROOT.exists():
    RESULTS_ROOT = Path("experiments/results/explainability/subset")
OUT_JSON = Path("experiments/results/analysis/explainability_summary.json")
OUT_STATS = Path("experiments/results/analysis/stats/explainability.txt")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_progress_csv(path: Path):
    """
    Returns:
        {
            "progress_t": [...],
            "progress_size": [...],
            "progress_event": [...],
            "progress_pixel": [...]
        }
    """
    progress_t = []
    progress_size = []
    progress_event = []
    progress_pixel = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            progress_t.append(float(row["t_sec"]))
            progress_size.append(int(row["explanation_size"]))
            progress_event.append(row["event"])
            progress_pixel.append(row["pixel"])

    return {
        "progress_t": progress_t,
        "progress_size": progress_size,
        "progress_event": progress_event,
        "progress_pixel": progress_pixel,
    }


def load_mode(mode_dir: Path):
    summary_path = mode_dir / "summary.json"
    progress_path = mode_dir / "progress.csv"

    if not summary_path.exists() or not progress_path.exists():
        return None

    summary = load_json(summary_path)
    progress = load_progress_csv(progress_path)

    return {
        "time_elapsed_sec": summary["time_elapsed_sec"],
        "final_explanation_size": summary["final_explanation_size"],
        "total_incremental_tightenings": summary["total_incremental_tightenings"],
        "total_conflicts_recorded": summary["total_conflicts_recorded"],
        "total_conflicts_when_sat": summary["total_conflicts_when_sat"],
        "progress_t": progress["progress_t"],
        "progress_size": progress["progress_size"],
        "progress_event": progress["progress_event"],
        "progress_pixel": progress["progress_pixel"],
    }


def parse_point_idx(point_dir: Path) -> int:
    name = point_dir.name
    assert name.startswith("point_"), f"Unexpected point dir name: {name}"
    return int(name[len("point_"):])


def mean(xs):
    assert len(xs) > 0, "Cannot take mean of empty list"
    return sum(xs) / len(xs)


def write_stats(records):
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)

    baseline_sizes = [float(r["baseline"]["final_explanation_size"]) for r in records]
    incremental_sizes = [float(r["incremental"]["final_explanation_size"]) for r in records]

    incremental_tightenings = [
        float(r["incremental"]["total_incremental_tightenings"]) for r in records
    ]
    incremental_conflicts = [
        float(r["incremental"]["total_conflicts_recorded"]) for r in records
    ]

    with OUT_STATS.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("\\;\\textbf{Method}\\; &\n")
        f.write("\\;\\textbf{Explanation Size}\\; &\n")
        f.write("\\;\\textbf{Propagations}\\; &\n")
        f.write("\\;\\textbf{Conflicts} \\;\\\\\n")
        f.write("\\midrule\n")
        f.write(
            f"Non-incremental &\n"
            f"{mean(baseline_sizes):.2f} &\n"
            f"-- &\n"
            f"-- \\\\\n"
        )
        f.write(
            f"Incremental &\n"
            f"\\bf{{{mean(incremental_sizes):.2f}}} &\n"
            f"{mean(incremental_tightenings):.2f} &\n"
            f"{mean(incremental_conflicts):.2f} \\\\\n"
        )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Minimal sufficient feature set extraction on GTSRB.}\n")
        f.write("\\label{tab:feature-set-extraction-results}\n")
        f.write("\\end{table}\n")
        f.write("\\vspace{-1.5em}\n")


def main():
    assert RESULTS_ROOT.exists(), f"Missing results root: {RESULTS_ROOT}"

    included = []
    skipped = []
    records = []

    point_dirs = sorted(
        [p for p in RESULTS_ROOT.iterdir() if p.is_dir() and p.name.startswith("point_")],
        key=parse_point_idx,
    )

    for point_dir in point_dirs:
        point_idx = parse_point_idx(point_dir)

        baseline_dir = point_dir / "baseline"
        incremental_dir = point_dir / "incremental"

        baseline = load_mode(baseline_dir)
        incremental = load_mode(incremental_dir)

        # Skip points where incremental was not run / not saved
        if baseline is None or incremental is None:
            skipped.append(point_idx)
            continue

        record = {
            "point": point_idx,
            "baseline": baseline,
            "incremental": incremental,
        }
        records.append(record)
        if incremental["total_conflicts_when_sat"] == 0:
            skipped.append(point_idx)
            continue
        included.append(point_idx)

    output = {
        "num_points_included": len(included),
        "num_points_skipped": len(skipped),
        "included_points": included,
        "skipped_points": skipped,
        "points": records,
    }

    os.makedirs(OUT_JSON.parent, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    assert len(records) > 0, "No included explainability records to summarize"
    write_stats(records)

    print(f"[summarize_explainability] done: {OUT_JSON}")
    print(f"[summarize_explainability] done: {OUT_STATS}")


if __name__ == "__main__":
    main()