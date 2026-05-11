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

SOLVED_PATTERNS = [
    r"\bsolved\b",
    r"\bsat\b",
    r"\bunsat\b",
    r"\bverified\b",
    r"\bsuccess\b",
]

UNSOLVED_PATTERNS = [
    r"\btimeout\b",
    r"\bfailed\b",
    r"\bunknown\b",
    r"\berror\b",
]

PROPAGATION_PATTERNS = [
    r"propagations[:=]\s*([0-9.]+)",
    r"propagation[:=]\s*([0-9.]+)",
]

CONFLICT_PATTERNS = [
    r"conflicts[:=]\s*([0-9.]+)",
    r"conflict[:=]\s*([0-9.]+)",
]


def parse_last_float(log_path: Path, patterns):
    text = log_path.read_text(errors="replace").lower()

    matches = []
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            try:
                matches.append(float(m.group(1)))
            except ValueError:
                pass

    if not matches:
        return None

    return matches[-1]


def parse_time(log_path: Path):
    """
    Try to extract a runtime from a .log file.
    Adjust TIME_PATTERNS if your logs use a different format.
    """
    return parse_last_float(log_path, TIME_PATTERNS)


def parse_propagations(log_path: Path):
    return parse_last_float(log_path, PROPAGATION_PATTERNS)


def parse_conflicts(log_path: Path):
    return parse_last_float(log_path, CONFLICT_PATTERNS)


def parse_solved(log_path: Path):
    """
    Returns True if the log appears solved, False otherwise.

    This is intentionally conservative:
    - timeout / failed / unknown / error means unsolved
    - solved / sat / unsat / verified / success means solved
    """
    text = log_path.read_text(errors="replace").lower()

    if any(re.search(pattern, text) for pattern in UNSOLVED_PATTERNS):
        return False

    if any(re.search(pattern, text) for pattern in SOLVED_PATTERNS):
        return True

    return False


def fmt_num(value, digits=1):
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def write_latex_table(
    output_path: Path,
    baseline_time,
    incremental_time,
    baseline_solved,
    incremental_solved,
    incremental_propagations=None,
    incremental_conflicts=None,
    caption="Evaluation of iterative input splitting for Lyapunov certificate verification.",
    label="tab:input-split-results",
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    speedup = baseline_time / incremental_time if incremental_time > 0 else None

    if speedup is None:
        speedup_text = "--"
    else:
        speedup_text = rf"\textbf{{{speedup:.2f}$\times$}}"

    propagations_text = fmt_num(incremental_propagations, digits=1)
    conflicts_text = fmt_num(incremental_conflicts, digits=1)

    table = rf"""\begin{{table}}[h]
\centering
\begin{{tabular}}{{lcccc}}
\toprule
\textbf{{Method}}  &
\textbf{{Time (s)}}  &
\textbf{{Solved}}  &
\textbf{{Propagations}}  &
\textbf{{Conflicts}}  \\
\midrule
Non-incremental &
{baseline_time:.1f} &
{baseline_solved} &
-- &
-- \\
Incremental &
\textbf{{{incremental_time:.1f}}} &
\textbf{{{incremental_solved}}} &
{propagations_text} &
{conflicts_text}\\
\midrule
\textbf{{Speedup}} &
{speedup_text} &
-- &
-- &
-- \\
\bottomrule
\end{{tabular}}
\vspace{{0.5em}}
\caption{{{caption}}}
\label{{{label}}}
\end{{table}}
"""

    output_path.write_text(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="experiments/results/inputsplit",
        help="Directory containing 'baseline' and 'incremental' subdirectories with .log files",
    )
    parser.add_argument(
        "--output",
        default="experiments/results/analysis/plots/runtime_scatter.png",
        help="Output path for the runtime scatter plot",
    )
    parser.add_argument(
        "--table-output",
        default="experiments/results/analysis/stats/input_split_results.txt",
        help="Output path for the LaTeX table",
    )

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

    baseline_solved = 0
    incremental_solved = 0

    incremental_propagations = []
    incremental_conflicts = []

    for name in common_names:
        baseline_log = baseline_logs[name]
        incremental_log = incremental_logs[name]

        baseline_time = parse_time(baseline_log)
        incremental_time = parse_time(incremental_log)

        if parse_solved(baseline_log):
            baseline_solved += 1

        if parse_solved(incremental_log):
            incremental_solved += 1

        prop = parse_propagations(incremental_log)
        conf = parse_conflicts(incremental_log)

        if prop is not None:
            incremental_propagations.append(prop)

        if conf is not None:
            incremental_conflicts.append(conf)

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

    plot_output = Path(args.output)
    plot_output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 10))
    plt.scatter(xs, ys, alpha=0.75)

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

    # Optional: hide minor tick labels.
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
    plt.savefig(plot_output, dpi=300)
    print(f"Saved plot to {plot_output}")

    baseline_total_time = sum(xs)
    incremental_total_time = sum(ys)

    avg_incremental_propagations = (
        sum(incremental_propagations) / len(incremental_propagations)
        if incremental_propagations
        else None
    )

    avg_incremental_conflicts = (
        sum(incremental_conflicts) / len(incremental_conflicts)
        if incremental_conflicts
        else None
    )

    table_output = Path(args.table_output)

    write_latex_table(
        output_path=table_output,
        baseline_time=baseline_total_time,
        incremental_time=incremental_total_time,
        baseline_solved=baseline_solved,
        incremental_solved=incremental_solved,
        incremental_propagations=avg_incremental_propagations,
        incremental_conflicts=avg_incremental_conflicts,
    )

    print(f"Saved LaTeX table to {table_output}")

    if skipped:
        print(f"Skipped {len(skipped)} logs because no valid runtime was found.")


if __name__ == "__main__":
    main()