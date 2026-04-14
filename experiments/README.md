
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

Expected output:

```
[run_smoke] start <Day> <DD> <Mon> <YYYY> <HH:MM:SS> <TZ>
[smoke-rob] start <Day> <DD> <Mon> <YYYY> <HH:MM:SS> <TZ>
[smoke-rob] running 10 points
[smoke-rob] (1/10) index=0
[done] wrote robustness outputs to: experiments/results/robustness/smoke/point_0
[smoke-rob] (<i>/<N>) index=<k>
[done] wrote robustness outputs to: experiments/results/robustness/smoke/point_<k>
...
[smoke-rob] (10/10) index=9
[done] wrote robustness outputs to: experiments/results/robustness/smoke/point_9
[smoke-expl] start <Day> <DD> <Mon> <YYYY> <HH:MM:SS> <TZ>
[smoke-expl] running 10 points
[smoke-expl] (1/10) index=27
[done] wrote explainability outputs under: experiments/results/explainability/smoke/point_27
[smoke-expl] (<i>/<N>) index=<k>
[done] wrote explainability outputs under: experiments/results/explainability/smoke/point_<k>
...
[smoke-expl] (10/10) index=66
[done] wrote explainability outputs under: experiments/results/explainability/smoke/point_66
[smoke-expl] success
[smoke-expl] done <Day> <DD> <Mon> <YYYY> <HH:MM:SS> <TZ>
[run_smoke] done <Day> <DD> <Mon> <YYYY> <HH:MM:SS> <TZ>
```

Or run each use case separately:

```bash
bash experiments/scripts/run_smoke_robustness.sh       # ~15–20 minutes
bash experiments/scripts/run_smoke_explainability.sh   # ~45-55 minutes
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

This must be run on a cluster with SLURM support and access to a token-based (floating) license of Gurobi

### 3.0. Installation

Check that you are on a cluster with SLURM support:
```bash
sinfo | head -n 5
```

We rocommend continuing in an interactive SLURM session for installation (not a must):
```bash
srun --mem=400m -c4 --time=1-00 --pty $SHELL
```

Clone the Marabou repository and check out the release branch:
```
git clone https://github.com/Raya5/Marabou.git
cd Marabou 
git checkout release
```

Install Gurobi and set up the license:
```
cd tools
bash install_gurobi.sh # ~1-2 minutes
export GUROBI_HOME=$(pwd)"/gurobi1103/linux64"
export PATH=$GUROBI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
cd ..
```

Set up your Gurobi license (Note that, unlike the student license acquired earlier, reproducing the full results requires a token-based (floating) Gurobi license (TYPE=TOKEN), provided via a Gurobi Token Server. This type of license is typically available in university-managed clusters and shared environments):
```
export GRB_LICENSE_FILE="/path/to/your/gurobi.lic"
```

Install Marabou with Gurobi:
```
bash install_marabou.sh # ~25-30 minutes
```

### 3.1.

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

After the experiments have completed, you can run the analysis scripts to reproduce results from the paper. To reproduce plots and aggregated statistics:

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

