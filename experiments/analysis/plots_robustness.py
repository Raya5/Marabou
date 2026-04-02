#!/usr/bin/env python3
"""
experiments/analysis/plots_robustness.py

Create the robustness runtime scatter plot from:
  experiments/results/analysis/robustness_summary.json

Input format (produced by summarize_robustness.py):
{
  "points": [
    {
      "point": 38,
      "baseline": {...},
      "incremental": {...}
    },
    ...
  ]
}

Outputs:
  experiments/results/analysis/plots/plot1_time_scatter.png
  experiments/results/analysis/plots/plot1_data_used.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path("experiments/results/analysis/robustness_summary.json")
DEFAULT_OUT_DIR = Path("experiments/results/analysis/plots")
DEFAULT_TOTAL_TIMEOUT = 1200.0
DEFAULT_TOTAL_TIMEOUT = 600.0
DEFAULT_PRECISION = 0.001
DEFAULT_PRECISION = 0.01


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clamp_time(t: float, total_timeout: float) -> float:
    return float(min(t, total_timeout))


def classify_timeout(inc_t: float, nor_t: float, total_timeout: float) -> str:
    inc_to = inc_t > total_timeout
    nor_to = nor_t > total_timeout

    if (not inc_to) and (not nor_to):
        return "both_under"
    if inc_to and nor_to:
        return "both_timeout"
    if nor_to and (not inc_to):
        return "nor_only"
    return "inc_only"


def write_plot1_csv(rows: List[Dict[str, Any]], out_path: Path):
    with out_path.open("w", encoding="utf-8", newline="") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--total-timeout", type=float, default=DEFAULT_TOTAL_TIMEOUT)
    parser.add_argument("--precision", type=float, default=DEFAULT_PRECISION)
    args = parser.parse_args()

    data = load_json(args.input)
    points = data["points"]

    assert len(points) > 0, f"No points found in {args.input}"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    xs = []
    ys = []
    cats = []

    for record in points:
        pt_idx = int(record["point"])
        inc = record["incremental"]
        nor = record["baseline"]

        # Skip points that were not actually run
        if inc.get("status") == "SkippedMisclassified" or nor.get("status") == "SkippedMisclassified":
            print(f"[plots_robustness] skipping misclassified point {pt_idx}")
            continue

        # Skip malformed/incomplete entries
        if "time_sec" not in inc or "time_sec" not in nor:
            print(f"[plots_robustness] skipping incomplete point {pt_idx}")
            continue

        inc_t_raw = float(inc["time_sec"])
        nor_t_raw = float(nor["time_sec"])

        cat = classify_timeout(inc_t_raw, nor_t_raw, args.total_timeout)
        if cat == "inc_only":
            print(
                f"Point {pt_idx} nor only timeout: "
                f"inc_time={inc_t_raw}, nor_time={nor_t_raw}"
            )

        if inc_t_raw < args.total_timeout:
            print(f"Point {pt_idx} inc under timeout: inc_time={inc_t_raw}, args.total_timeout={args.total_timeout}")
            print(f"  precision width: {inc['precision_width']}, precision threshold: {args.precision}")
            assert float(inc["precision_width"]) < args.precision
        if nor_t_raw < args.total_timeout:
            assert float(nor["precision_width"]) < args.precision

        inc_t = clamp_time(inc_t_raw, args.total_timeout)
        nor_t = clamp_time(nor_t_raw, args.total_timeout)

        xs.append(inc_t)
        ys.append(nor_t)
        cats.append(cat)

        rows.append({
            "point": pt_idx,
            "inc_status": inc["status"],
            "nor_status": nor["status"],
            "inc_time_raw": inc_t_raw,
            "nor_time_raw": nor_t_raw,
            "inc_time_clamped": inc_t,
            "nor_time_clamped": nor_t,
            "category": cat,
            "inc_precision_width": inc["precision_width"],
            "nor_precision_width": nor["precision_width"],
            "inc_tightenings": inc["total_incremental_tightenings"],
            "inc_conflicts": inc["total_conflicts_recorded"],
        })

    plot1_csv = args.out_dir / "plot1_data_used.csv"
    write_plot1_csv(rows, plot1_csv)

    plt.rcParams.update({
        "font.size": 20,
        "axes.labelsize": 28,
        "axes.titlesize": 28,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 18,
    })

    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()

    T = args.total_timeout
    min_val = 0.1
    max_val = T * 1.05

    # diagonal y=x
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color="black",
        linewidth=3,
    )

    # timeout lines
    ax.axvline(x=T, color="black", linestyle="-", linewidth=2)
    ax.axhline(y=T, color="black", linestyle="-", linewidth=2)

    color_map = {
        "both_under": "tab:blue",
        "both_timeout": "tab:blue",
        "nor_only": "tab:blue",
        "inc_only": "tab:blue",
    }
    label_map = {
        "both_under": "Both < timeout",
        "both_timeout": "Both timeout",
        "nor_only": "Only non-incremental timeout",
        "inc_only": "Only incremental timeout",
    }

    alpha = 0.45
    for cat in ["both_under", "both_timeout", "nor_only", "inc_only"]:
        xcat = [y for y, c in zip(ys, cats) if c == cat]  # non-inc
        ycat = [x for x, c in zip(xs, cats) if c == cat]  # inc
        if not xcat:
            continue
        ax.scatter(
            xcat,
            ycat,
            s=55,
            alpha=alpha,
            edgecolors="none",
            color=color_map[cat],
            label=label_map[cat],
        )

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ax.set_xlabel("Non-incremental total time (s)")
    ax.set_ylabel("Incremental total time (s)")
    ax.set_title("Incremental vs Non-incremental Runtimes")

    ticks = np.linspace(0, args.total_timeout, 7)
    plt.xticks(ticks[1:])  # skip 0 on x-axis
    plt.yticks(ticks)
    ax.grid(False)
    fig.tight_layout()

    out_path = args.out_dir / "plot1_time_scatter.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[plots_robustness] wrote: {plot1_csv}")
    print(f"[plots_robustness] wrote: {out_path}")


if __name__ == "__main__":
    main()