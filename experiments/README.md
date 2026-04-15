
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

### 0. Installation (~ 1.5 hours)

#### 0.1. obtaining Gurobi license


To run the experiments in this repository, you must have a **Gurobi Optimizer** license. Gurobi provides free licenses to students and faculty.

1. Register for an Academic Account
    1.  Visit the Gurobi Registration Page [https://www.gurobi.com/registration/](https://www.gurobi.com/registration/).
    2.  Register using your institutional/university email address.
    3.  Select **"Academic"** under "Your Work" in the second page in registration.
    4.  Continue with the registration process according to the instructions.

2. Obtain your License Key
    1.  Log in to the Gurobi Website:[https://portal.gurobi.com/iam/login](https://portal.gurobi.com/iam/login).
    2.  Navigate to the Licenses page by clicking Licenses on the left or use the link: [https://portal.gurobi.com/iam/licenses/list](https://portal.gurobi.com/iam/licenses/list).
    3.  Click on the "Request" button or use the link: [https://portal.gurobi.com/iam/licenses/request](https://portal.gurobi.com/iam/licenses/request).
    4.  Choose the "WLS Academic" license and click "Generate Now". Accept the terms and click **Confirm Request**.
    5.  Download the license file (`gurobi.lic`) and save it in "<path_to_your_license_file>".

<!-- ```bash
# Example for Linux/macOS
export GRB_LICENSE_FILE="/path/to/your/gurobi.lic" -->

### 1. Smoke Test (~ 15 minutes)

Make sure that you are in the `Marabou` directory and run small sanity tests (< 15 minutes):

```bash
bash experiments/scripts/run_smoke.sh
````

Expected sucess output:

```
[run_smoke] start <Day> <DD> <Mon> <YYYY> <HH:MM:SS> <TZ>
[smoke-rob] start <Day> <DD> <Mon> <YYYY> <HH:MM:SS> <TZ>
[smoke-rob] index=0
[done] wrote robustness outputs to: experiments/results/robustness/smoke/point_0
[smoke-rob] index=1
[done] wrote robustness outputs to: experiments/results/robustness/smoke/point_1
[smoke-rob] success
[smoke-rob] done <Day> <DD> <Mon> <YYYY> <HH:MM:SS> <TZ>
[smoke-expl] start <Day> <DD> <Mon> <YYYY> <HH:MM:SS> <TZ>
[smoke-expl] index=27
[done] wrote explainability outputs under: experiments/results/explainability/smoke/point_27
[smoke-expl] success
[smoke-expl] done <Day> <DD> <Mon> <YYYY> <HH:MM:SS> <TZ>
[run_smoke] done <Day> <DD> <Mon> <YYYY> <HH:MM:SS> <TZ>
```

These verify that:

* Marabou builds correctly
* Incremental functionality works
* Outputs are generated in the expected format

---

### 2. Subset Sequential Run (~ 8 hours)

Due to the long runtime of the full experiments, we provide a sequential run on a small subset of points that as advised in the artifact evaluation instructions finishes within 8 hours.
For the full experiment suite, see section [3. Full Experiments](#3-full-experiments).

Run a subset of points:

```bash
bash experiments/scripts/run.sh 
```
This will take about 8 hours and runs a subset of points sequentially.


---

### 3. Full Experiments 

The subset run above is sufficient to verify that the code is working correctly and generates the expected outputs as in the paper.
This step is optional and it reproduces the full results from the paper. 

To run the full experiment suite:

```bash
bash experiments/scripts/run.sh --full
```
This will take at least 3 days machine time and might take longer.

### temp


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

