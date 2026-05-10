#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext

TIME_PATTERNS = [
    r"elapsed[:=]\s*([0-9.]+)",
    r"time[:=]\s*([0-9.]+)",
    r"runtime[:=]\s*([0-9.]+)",
    r"total time[:=]\s*([0-9.]+)",
    r"([0-9.]+)\s*seconds",
    r"([0-9.]+)\s*s\b",
]


def parse_time(log_path: Path):
    """
    Try to extract a runtime from a .log file.
    Adjust TIME_PATTERNS if your logs use a different format.
    """
    text = log_path.read_text(errors="replace").lower()

    matches = []
    for pattern in TIME_PATTERNS:
        for m in re.finditer(pattern, text):
            try:
                matches.append(float(m.group(1)))
            except ValueError:
                pass

    if not matches:
        return None

    # Usually the final timing is the most useful one.
    return matches[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="experiments/results/inputsplit",
                        help="Directory containing 'baseline' and 'incremental' subdirectories with .log files (default: experiments/results/inputsplit)")
    parser.add_argument("--output", default="experiments/results/analysis/plots/runtime_scatter.png")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    baseline_dir = results_dir / "baseline"
    incremental_dir = results_dir / "incremental"

    baseline_logs = {p.stem: p for p in baseline_dir.glob("*.log")}
    incremental_logs = {p.stem: p for p in incremental_dir.glob("*.log")}

    common_names = sorted(set(baseline_logs) & set(incremental_logs))

    xs = []
    ys = []
    labels = []

    skipped = []

    for name in common_names:
        baseline_time = parse_time(baseline_logs[name])
        incremental_time = parse_time(incremental_logs[name])

        if baseline_time is None or incremental_time is None:
            skipped.append(name)
            continue

        # Log axes cannot show zero.
        if baseline_time <= 0 or incremental_time <= 0:
            skipped.append(name)
            continue

        xs.append(baseline_time)
        ys.append(incremental_time)
        labels.append(name)

    if not xs:
        raise RuntimeError("No valid matching runtimes found.")

    plt.figure(figsize=(10, 10))
    plt.scatter(xs, ys, alpha=0.75)

    min_val = min(min(xs), min(ys))
    max_val = max(max(xs), max(ys))

    plt.xscale("log")
    plt.yscale("log")

    ax = plt.gca()

    ticks = [1e-1, 1e0, 1e1, 1e2, 1e3]

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

    ax.set_xlim(1e-1, 1e3)
    ax.set_ylim(1e-1, 1e3)

    # Optional: hide minor tick labels
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))

    plt.plot(
        [1e-1, 1e3],
        [1e-1, 1e3],
        linestyle="--",
        linewidth=1,
        label="equal runtime",
    )

    plt.xlabel("Baseline runtime (seconds, log scale)")
    plt.ylabel("Incremental runtime (seconds, log scale)")
    plt.title("Baseline vs Incremental Runtime")

    plt.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved plot to {args.output}")

    if skipped:
        print(f"Skipped {len(skipped)} logs because no valid runtime was found.")


if __name__ == "__main__":
    main()