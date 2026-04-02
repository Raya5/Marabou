#!/usr/bin/env python3
"""
experiments/analysis/summarize_robustness.py

Summarize robustness results into a single JSON file.

Expected input layout:
  experiments/results/<robustness_run_dir>/point_<idx>/
    baseline/summary.json
    incremental/summary.json

Example:
  experiments/results/robustness_4/full_0.001_180/point_38/
    baseline/summary.json
    incremental/summary.json

Output:
  experiments/results/analysis/robustness_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INPUT_ROOT = Path("experiments/results/robustness_4/full")
DEFAULT_OUTPUT = Path("experiments/results/analysis/robustness_summary.json")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root robustness results directory containing point_* folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write summarized robustness JSON.",
    )
    args = parser.parse_args()

    input_root: Path = args.input_root
    output_path: Path = args.output

    assert input_root.exists(), f"Missing input root: {input_root}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    point_dirs = sorted(
        [p for p in input_root.iterdir() if p.is_dir() and p.name.startswith("point_")],
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

        record = {
            "point": point_idx,
            "baseline": baseline,
            "incremental": incremental,
        }
        records.append(record)
        included.append(point_idx)

    output = {
        "input_root": str(input_root),
        "num_points_included": len(included),
        "num_points_skipped": len(skipped),
        "included_points": included,
        "skipped_points": skipped,
        "points": records,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[summarize_robustness] included {len(included)} points")
    print(f"[summarize_robustness] skipped {len(skipped)} points")
    print(f"[summarize_robustness] wrote: {output_path}")


if __name__ == "__main__":
    main()