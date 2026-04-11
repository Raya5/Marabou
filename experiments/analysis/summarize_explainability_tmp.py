#!/usr/bin/env python3
"""
experiments/analysis/summarize_explainability.py

Summarize explainability full-run results into a single JSON file.

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
"""

import csv
import json
from pathlib import Path


RESULTS_ROOT = Path("experiments/results/explainability/full")
OUT_DIR = Path("experiments/results/analysis")
OUT_PATH = OUT_DIR / "explainability_summary.json"


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
    # expects point_<idx>
    name = point_dir.name
    assert name.startswith("point_"), f"Unexpected point dir name: {name}"
    return int(name[len("point_"):])


def main():
    assert RESULTS_ROOT.exists(), f"Missing results root: {RESULTS_ROOT}"

    # OUT_DIR.mkdir(parents=True, exist_ok=True)

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

        # skip points where incremental was not run / not saved
        if baseline is None or incremental is None:
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
        "num_points_included": len(included),
        "num_points_skipped": len(skipped),
        "included_points": included,
        "skipped_points": skipped,
        "points": records,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[summarize_explainability] included {len(included)} points")
    print(f"[summarize_explainability] skipped {len(skipped)} points")
    print(f"[summarize_explainability] included points: {included}")
    print(f"[summarize_explainability] skipped points: {skipped}")
    print(f"[summarize_explainability] wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()