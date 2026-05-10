#!/usr/bin/env python3
"""
experiments/analysis/plot.py

Run all plotting scripts.

Expected usage:
    python experiments/analysis/plot.py

This generates:
    experiments/results/analysis/plots/robustness.png
    experiments/results/analysis/plots/explainability.png

Each use-case plotting script is responsible for producing its own plots.
"""

from plots_explainability import main as plots_explainability_main
from plots_robustness import main as plots_robustness_main
from plots_inputsplit import main as plots_input_splitting_main  
from summarize_explainability import main as summarize_explainability_main
from summarize_robustness import main as summarize_robustness_main


def main():
    summarize_explainability_main()
    summarize_robustness_main()
    plots_explainability_main()
    plots_robustness_main()
    plots_input_splitting_main()  


if __name__ == "__main__":
    main()