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
# from summarize_input_splitting import main as summarize_input_splitting_main  # TODO: implement input splitting summarization and tables


def main():
    summarize_explainability_main()
    summarize_robustness_main()
    # summarize_input_splitting_main()  # TODO: enable once implemented


if __name__ == "__main__":
    main()