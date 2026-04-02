#!/usr/bin/env python3
"""
experiments/analysis/plots_explainability.py

Create the explainability mean size-vs-time plot from:
    experiments/results/analysis/explainability_summary.json

Expected input structure per point:
{
  "point": ...,
  "baseline": {
    "time_elapsed_sec": ...,
    "final_explanation_size": ...,
    "total_conflicts": ...,
    "progress_t": [...],
    "progress_size": [...],
    ...
  },
  "incremental": {
    ...
  }
}
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path("experiments/results/analysis/explainability_summary.json")
DEFAULT_OUT_DIR = Path("experiments/results/analysis/plots")


def load_summary(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def step_on_grid(times, sizes, grid):
    """
    Convert a stepwise progress curve into values sampled on a fixed time grid.

    times: list of event times
    sizes: list of explanation sizes at those times
    grid: numpy array of target times

    Returns:
        numpy array same shape as grid
    """
    times = np.asarray(times, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    grid = np.asarray(grid, dtype=float)

    assert len(times) == len(sizes), "times and sizes must have same length"
    assert len(times) > 0, "progress curve must be non-empty"

    # For each grid point, find the last event time <= grid[t]
    idx = np.searchsorted(times, grid, side="right") - 1

    # Before first event, use the initial size
    idx[idx < 0] = 0

    return sizes[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--grid-points", type=int, default=500)
    parser.add_argument("--tmax", type=float, default=120.0)
    args = parser.parse_args()

    data = load_summary(args.input)
    points = data["points"]

    assert len(points) > 0, f"No points found in {args.input}"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    included_points = [p["point"] for p in points]
    print(f"[plots_explainability] included points: {included_points}")

    # We already skipped missing-incremental points in summarize_explainability.py,
    # so "points" here are exactly the filtered set we want.
    filtered = points

    grid = np.linspace(0, args.tmax, args.grid_points)
    inc_curves = []
    nor_curves = []

    for record in filtered:
        inc = record["incremental"]
        nor = record["baseline"]

        inc_curves.append(
            step_on_grid(inc["progress_t"], inc["progress_size"], grid)
        )
        nor_curves.append(
            step_on_grid(nor["progress_t"], nor["progress_size"], grid)
        )

    inc_curves = np.asarray(inc_curves, dtype=float)
    nor_curves = np.asarray(nor_curves, dtype=float)

    plt.figure()
    plt.plot(
        grid,
        np.mean(inc_curves, axis=0),
        label="incremental (mean)",
        linewidth=2,
    )
    plt.plot(
        grid,
        np.mean(nor_curves, axis=0),
        label="normal (mean)",
        linewidth=2.0,
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Explanation size")
    plt.title("Explanation size vs time")
    plt.legend()
    plt.xlim(0, args.tmax)
    plt.ylim(top=700)#, bottom=840)
    plt.tight_layout()

    p_png = args.out_dir / "plot2_size_vs_time_mean_filtered_sat_conflict.png"
    p_pdf = args.out_dir / "plot2_size_vs_time_mean_filtered_sat_conflict.pdf"

    plt.savefig(p_png, dpi=200, bbox_inches="tight")
    plt.savefig(p_pdf, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[plots_explainability] wrote: {p_png}")
    print(f"[plots_explainability] wrote: {p_pdf}")


if __name__ == "__main__":
    main()