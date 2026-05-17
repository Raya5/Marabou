#!/usr/bin/env python3
"""
Run every .ipq benchmark under a directory through both the input-split
baseline and the input-split incremental solver, capturing per-query logs.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = {
    "baseline": "marabou_inputsplit_baseline.py",
    "incremental": "marabou_inputsplit_incremental.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks-dir", default="benchmarks",
                        help="Directory to search for .ipq files (default: benchmarks)")
    parser.add_argument("--results-dir", default="experiments/results/inputsplit",
                        help="Directory to write per-query logs (default: experiments/results/inputsplit)")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--timeout-factor", type=float, default=1.5)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--smoke", action="store_true", help="Run a single query for a quick smoke test")
    parser.add_argument("--subset", action="store_true", help="Run a subset of queries for testing")
    parser.add_argument("--full", action="store_true", help="Run all queries (default)")
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    benchmarks_dir = Path(args.benchmarks_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    
    queries = sorted(benchmarks_dir.rglob("*.ipq"))
    if not queries:
        print(f"No .ipq files found under {benchmarks_dir}", file=sys.stderr)
        sys.exit(1)

    tier = "full"
    if args.smoke:
        queries = [queries[0]]
        tier = "smoke"
    elif args.subset:
        queries = queries[:10]  # Run only the first 10 queries
        tier = "subset"
    elif args.full:
        pass  # Run all queries

    # print(f"Found {len(queries)} queries under {benchmarks_dir}")
    # print(f"Writing logs to {results_dir}")

    for config, script_name in CONFIGS.items():
        (results_dir / config).mkdir(parents=True, exist_ok=True)

    for i, q in enumerate(queries, 1):
        name = q.stem
        for config, script_name in CONFIGS.items():
            log_path = results_dir / config / f"{name}.log"
            cmd = [
                sys.executable, str(scripts_dir / script_name), str(q),
                "--timeout", str(args.timeout),
                "--timeout-factor", str(args.timeout_factor),
                "--max-depth", str(args.max_depth),
            ]
            start = time.time()
            with open(log_path, "w") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
            elapsed = time.time() - start
            if config == "baseline":
                print(f"[{tier}-inp] index={i-1}")


if __name__ == "__main__":
    main()
