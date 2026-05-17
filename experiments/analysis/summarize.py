#!/usr/bin/env python3
"""
experiments/analysis/summarize.py

Run all summarization scripts.

Expected usage:
    python experiments/analysis/summarize.py

This generates:
    experiments/results/analysis/explainability_summary.json
    experiments/results/analysis/robustness_summary.json

Each use-case summarizer is also responsible for generating its own
paper-style tables.
"""

from summarize_explainability import main as summarize_explainability_main
from summarize_robustness import main as summarize_robustness_main


def main():
    summarize_explainability_main()
    summarize_robustness_main()


if __name__ == "__main__":
    main()