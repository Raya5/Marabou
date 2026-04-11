
# Experiments: Incremental Neural Network Verification

This directory contains the experimental infrastructure for reproducing the results of the paper:

> *Incremental Neural Network Verification via Learned Conflicts*

It provides a clean and reproducible release of:

- Incremental Marabou (C++ modifications)
- Two use cases:
  - **Robustness** (robustness radius determination)
  - **Input Splitting** #TODO
  - **Explainability** (minimal sufficient feature set extraction)

---

## Scope

The Marabou repository is a general-purpose neural network verification framework.  
The `experiments/` directory contains **the fixed experimental setup used in the paper**.

This release supports:

- A single fixed experimental configuration (timeouts, epsilon, precision)
- One entry point per use case
- Automatic execution of both:
  - Incremental mode
  - Non-incremental baseline
- Output formats aligned with the paper’s plots and statistics

---

## Directory Structure

- `scripts/` – execution scripts (smoke, subset, SLURM)
- `results/` – generated outputs
- `analysis/` – plotting and statistics scripts
- `data/` – datasets or dataset download scripts

---

## Quick Start

### 0. Installation

### 1. Smoke Test

Run small sanity tests (< 1.5 hours):

```bash
bash experiments/scripts/run_smoke.sh
````

Or run each use case separately:

```bash
bash experiments/scripts/run_smoke_explainability.sh   # ~45–50 minutes
bash experiments/scripts/run_smoke_robustness.sh       # ~15–20 minutes
```

These verify that:

* Marabou builds correctly
* Incremental functionality works
* Outputs are generated in the expected format

---

### 2. Subset Sequential Run

Due to the long runtime of the full experiments, we provide a sequential run on a small subset of points that is enough to verify the approach and results.
For the full experiment suite, see section [3. Full Experiments (SLURM)](#3-full-experiments-slurm).

Run a subset of points sequentially:

```bash
bash experiments/scripts/run_subset.sh
```

This runs:

* `run_subset_explainability.sh` (~12 hours)
* `run_subset_robustness.sh` (~30 hours)

Total expected runtime: **~40–42 hours**

---

### 3. Full Experiments (SLURM)

Run the full experiment suite using SLURM:

```bash
bash experiments/scripts/run_full_slurm.sh
```

This submits two job arrays:

* Explainability: indices `0–999`
* Robustness: indices `0–999`

Recommended configuration (already defined in the script):

* Parallelism limit: `50` (e.g., `--array=0-999%50`)

Expected runtime: **~3 days**

---

## Outputs

All results are written to:

```
experiments/results/<usecase>/<smoke|full>/point_<idx>/
```

---

## Analysis

To reproduce plots and aggregated statistics:

```bash
python experiments/analysis/summarize.py
python experiments/analysis/plot.py
```

Check the plots in `experiments/results/analysis/plots/robustness.png` and `experiments/results/analysis/plots/input_splitting.png` and `experiments/results/analysis/plots/explainability.png`.

Check the table results in `experiments/results/analysis/stats/robustness.txt` and `experiments/results/analysis/stats/input_splitting.txt` and `experiments/results/analysis/stats/explainability.txt`.

---

## Notes

* Some inputs may be skipped internally (e.g., invalid or trivial cases), consistent with the paper

---

## Running Individual Examples

```bash
python experiments/scripts/run_explainability.py --index 0
python experiments/scripts/run_robustness.py --index 0
```

