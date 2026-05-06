#!/usr/bin/env python3
"""
experiments/scripts/run_explainability.py

Combined runner + minimal VeriX implementation.

Outputs (mode-local; no inc_/nor_ prefixes):
experiments/results/explainability/<smoke|full|subset>/<run_id>/<baseline|incremental>/
  summary.json
  sat.txt
  unsat.txt
  timeout.txt
  progress.csv
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

# -------------------------
# Marabou imports
# -------------------------
# IMPORTANT:
#   Do NOT hardcode Marabou paths in the release.
#   Prefer either:
#     1) installing maraboupy in the environment, or
#     2) setting PYTHONPATH appropriately when running: export PYTHONPATH=$PWD/iv/Marabou:$PYTHONPATH
#
# Resolve iv/Marabou directory dynamically
CURRENT_FILE = Path(__file__).resolve()
SCRIPT_DIR = CURRENT_FILE.parent
MARABOU_ROOT = CURRENT_FILE.parents[2]  # .../Marabou

sys.path.insert(0, str(SCRIPT_DIR))     # local modules in experiments/scripts
sys.path.insert(0, str(MARABOU_ROOT))   # maraboupy

from maraboupy import Marabou, MarabouCore, MarabouUtils
from maraboupy.MarabouCore import StatisticsUnsignedAttribute

# Your IBP traversal implementation
from verix_traversal_ibp_onnx import traversal_order_ibp


# ============================================================
# VeriXMinimal
# ============================================================
class VeriXMinimal:
    """
    Minimal VeriX implementation supporting:
      - new_traversal(epsilon)
      - compute_explanation(epsilon) using Marabou queries + binary splitting
    """

    def __init__(
        self,
        *,
        dataset: str,
        image: np.ndarray,
        model_path_no_suffix: str,
        out_dir: str,
        incremental: bool,
    ):
        self.dataset = dataset.lower()
        self.image = np.asarray(image).astype(np.float32)

        self.out_dir = str(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.incremental = bool(incremental)
        self.overall_timeout_sec = 120.0 + 1

        # incremental IDs
        self._qid = 1

        # run counters
        self.total_incremental_tightenings = 0
        self.total_conflicts_recorded = 0
        self.total_conflicts_when_sat = 0

        # internal state
        self._start_time = None
        self.epsilon = None

        # output paths (MODE-LOCAL; NO PREFIX)
        self.progress_path = os.path.join(self.out_dir, "progress.csv")
        self.summary_path = os.path.join(self.out_dir, "summary.json")
        self.sat_path = os.path.join(self.out_dir, "sat.txt")
        self.unsat_path = os.path.join(self.out_dir, "unsat.txt")
        self.timeout_path = os.path.join(self.out_dir, "timeout.txt")

        # init progress.csv (overwrite)
        with open(self.progress_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_sec", "explanation_size", "event", "pixel"])

        # ORT session
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1

        self.onnx_path = model_path_no_suffix + ".onnx"
        self.onnx_model = onnx.load(self.onnx_path)
        self.onnx_session = ort.InferenceSession(
            self.onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
        )

        # Nominal label + logit ranking
        input_name = self.onnx_model.graph.input[0].name
        pred = self.onnx_session.run(None, {input_name: np.expand_dims(self.image, axis=0)})[0][0]
        self.label = int(pred.argmax())
        self.logit_rank = pred.argsort()[::-1]

        # Marabou model
        self.mara_model = Marabou.read_onnx(self.onnx_path)
        self.outputVars = self.mara_model.outputVars[0].flatten()

        # Property: add misclassification disjunction once
        self._add_misclassification_disjunction_once(margin=1e-6)

        # Stable IPQ
        self.ipq = self.mara_model.getInputQuery()
        self.ipq.push()  # base frame for the whole run

        if self.incremental:
            self.ica = MarabouCore.buildIncrementalConflictAnalyser()
            self.ipq.setIncrementalConflictAnalyser(self.ica)
        else:
            self.ica = None

        # Pixel indices (assumes (W,H,C))
        w, h = self.image.shape[0], self.image.shape[1]
        self.inputVars = np.arange(w * h, dtype=int)
        self.explanation_size = int(self.inputVars.size)

        # explanation sets
        self.unsat_set = []
        self.sat_set = []
        self.timeout_set = []

        # Marabou options
        self.options = Marabou.createOptions(
            numWorkers=2,
            timeoutInSeconds=120,
            verbosity=0,
        )
        self.options._incremental = bool(self.incremental)

    # ---------------------------
    # utilities
    # ---------------------------
    def next_id(self) -> int:
        assert self.incremental
        qid = self._qid
        self._qid += 1
        return qid

    def _maybe_next_id(self):
        return self.next_id() if self.incremental else None

    def _elapsed(self) -> float:
        assert self._start_time is not None
        return time.time() - self._start_time

    def _timed_out_globally(self) -> bool:
        return self._start_time is not None and (time.time() - self._start_time >= self.overall_timeout_sec)

    def _record_progress(self, *, time_elapsed: float, event: str, pixels=None):
        # pixels may be None/empty for init/done
        if pixels is None or (isinstance(pixels, (list, np.ndarray)) and len(pixels) == 0):
            assert event in ("init", "done")
            pixel_repr = ""
        else:
            assert event in ("unsat", "sat", "TIMEOUT")
            if isinstance(pixels, np.ndarray):
                pixel_repr = str(pixels.astype(int).tolist())
            else:
                pixel_repr = str(list(pixels))

        with open(self.progress_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([float(time_elapsed), int(self.explanation_size), str(event), pixel_repr])

    # ---------------------------
    # property
    # ---------------------------
    def _add_misclassification_disjunction_once(self, margin: float = 1e-6):
        """
        Adds: OR_{j != label} ( y_label - y_j <= -margin )
        i.e., exists j != label with y_j >= y_label + margin
        """
        y_label = int(self.outputVars[self.label])

        disjuncts = []
        for j in map(int, self.logit_rank):
            if j == int(self.label):
                continue
            y_j = int(self.outputVars[j])

            eq = MarabouUtils.Equation(MarabouCore.Equation.LE)
            eq.addAddend(1.0, y_label)
            eq.addAddend(-1.0, y_j)
            eq.setScalar(float(-margin))
            disjuncts.append([eq])

        self.mara_model.addDisjunctionConstraint(disjuncts)

    # ---------------------------
    # traversal
    # ---------------------------
    def new_traversal(self, *, epsilon: float):

        channels_first = False
        sorted_idx, sensitivity_map = traversal_order_ibp(
            onnx_path=self.onnx_path,
            image=self.image,
            epsilon=float(epsilon),
            label=self.label,
            input_min=0.0,
            input_max=1.0,
            channels_first=channels_first,
        )

        self.inputVars = sorted_idx.astype(int)
        self.sensitivity = sensitivity_map

    # ---------------------------
    # main procedure
    # ---------------------------
    def compute_explanation(self, *, epsilon: float):
        self.epsilon = float(epsilon)
        self.unsat_set = []
        self.sat_set = []
        self.timeout_set = []

        self.explanation_size = int(len(self.inputVars))

        # IMPORTANT: set start time before recording init so _elapsed works consistently
        self._start_time = time.time()
        self._record_progress(time_elapsed=0.0, event="init", pixels=None)

        self._binary_check_robust(self.inputVars, ancestors=None)

        # record done + summary
        t_done = self._elapsed()
        self._record_progress(time_elapsed=t_done, event="done", pixels=None)
        self._write_outputs(time_elapsed_done=t_done)

    # ---------------------------
    # binary splitting
    # ---------------------------
    def _mark_timeout_remaining(self, pixels: np.ndarray):
        pixels = np.asarray(pixels, dtype=int)
        if pixels.size == 0:
            return
        already = set(self.timeout_set)
        new = [int(p) for p in pixels.tolist() if int(p) not in already]
        self.timeout_set.extend(new)

    def _binary_check_robust(self, x, ancestors=None):
        # print(".", end="") 
        if self._timed_out_globally():
            self._mark_timeout_remaining(np.asarray(x, dtype=int))
            return

        if ancestors is None:
            ancestors = []

        x = np.asarray(x, dtype=int)
        if x.size == 0:
            return

        # leaf
        if x.size == 1:
            exit_code = self.check_robust(
                pixels=np.concatenate([np.asarray(self.unsat_set, dtype=int), x]),
                incremental=self.incremental,
                query_id=self._maybe_next_id(),
                ancestors=ancestors if self.incremental else None,
                added=x,
            )
            if exit_code == "unsat":
                self.unsat_set.extend(x.tolist())
                self.explanation_size -= int(x.size)
            elif exit_code == "sat":
                self.sat_set.extend(x.tolist())
            elif exit_code == "TIMEOUT":
                self.timeout_set.extend(x.tolist())
            else:
                raise RuntimeError(f"Unexpected exit_code {exit_code}")

            self._record_progress(time_elapsed=self._elapsed(), event=exit_code, pixels=x)
            return

        a, b = np.array_split(x, 2)

        # check a
        qid_a = self._maybe_next_id()
        exit_code_a = self.check_robust(
            pixels=np.concatenate([np.asarray(self.unsat_set, dtype=int), a]),
            incremental=self.incremental,
            query_id=qid_a,
            ancestors=ancestors if self.incremental else None,
            added=a,
        )

        if exit_code_a == "unsat":
            self.unsat_set.extend(a.tolist())
            self.explanation_size -= int(a.size)
            self._record_progress(time_elapsed=self._elapsed(), event="unsat", pixels=a)

            # then check b
            qid_b = self._maybe_next_id()
            exit_code_b = self.check_robust(
                pixels=np.concatenate([np.asarray(self.unsat_set, dtype=int), b]),
                incremental=self.incremental,
                query_id=qid_b,
                ancestors=ancestors if self.incremental else None,
                added=b,
            )

            if exit_code_b == "unsat":
                self.unsat_set.extend(b.tolist())
                self.explanation_size -= int(b.size)
                self._record_progress(time_elapsed=self._elapsed(), event="unsat", pixels=b)
            elif exit_code_b == "sat":
                if b.size == 1:
                    self.sat_set.extend(b.tolist())
                    self._record_progress(time_elapsed=self._elapsed(), event="sat", pixels=b)
                else:
                    new_anc = (ancestors + [qid_b]) if self.incremental else ancestors
                    self._binary_check_robust(b, ancestors=new_anc)
            elif exit_code_b == "TIMEOUT":
                self.timeout_set.extend(b.tolist())
                self._record_progress(time_elapsed=self._elapsed(), event="TIMEOUT", pixels=b)
            else:
                raise RuntimeError(f"Unexpected exit_code {exit_code_b}")
        elif exit_code_a == "sat":
            if a.size == 1:
                self.sat_set.extend(a.tolist())
                self._record_progress(time_elapsed=self._elapsed(), event="sat", pixels=a)
            else:
                new_anc = (ancestors + [qid_a]) if self.incremental else ancestors
                self._binary_check_robust(a, ancestors=new_anc)

            self._binary_check_robust(b, ancestors=ancestors)
        elif exit_code_a == "TIMEOUT":
            self.timeout_set.extend(a.tolist())
            self._record_progress(time_elapsed=self._elapsed(), event="TIMEOUT", pixels=a)
            # still try b (optional). We keep it simple and stop here:
            self._mark_timeout_remaining(b)
        else:
            raise RuntimeError(f"Unexpected exit_code {exit_code_a}")

    # ---------------------------
    # single query solve
    # ---------------------------
    def _get_conflicts_num_this_solve(self, stats) -> int:
        """
        Returns the number of conflicts of size >= 2 encountered in this solve.
        """
        return int(stats.getUnsignedAttribute(StatisticsUnsignedAttribute.NUM_CONFLICTS))

    def check_robust(self, *, pixels, incremental: bool, query_id, ancestors, added):
        """
        Solves the misclassification-disjunction once using IPQ bounds.
        Returns one of: "unsat", "sat", "TIMEOUT".
        """
        # print(".", end="")
        ipq = self.ipq
        assert ipq is not None

        if incremental:
            assert query_id is not None
            assert ancestors is not None

        eps = float(self.epsilon)

        w, h, c = self.image.shape[0], self.image.shape[1], self.image.shape[2]
        flat = self.image.reshape(w * h, c)

        pixels = np.asarray(pixels, dtype=int)
        fixed_pixels = np.setdiff1d(np.arange(w * h, dtype=int), pixels, assume_unique=False)

        # Push a frame for input bounds
        ipq.push()

        # input bounds
        assert self.dataset == "gtsrb", f"Only gtsrb is supported in this minimal version, got {self.dataset}"
        for i in pixels:
            r, g, b = map(float, flat[i])
            base = 3 * int(i)
            ipq.setLowerBound(base + 0, max(0.0, r - eps))
            ipq.setUpperBound(base + 0, min(1.0, r + eps))
            ipq.setLowerBound(base + 1, max(0.0, g - eps))
            ipq.setUpperBound(base + 1, min(1.0, g + eps))
            ipq.setLowerBound(base + 2, max(0.0, b - eps))
            ipq.setUpperBound(base + 2, min(1.0, b + eps))

        for i in fixed_pixels:
            r, g, b = map(float, flat[i])
            base = 3 * int(i)
            ipq.setLowerBound(base + 0, r)
            ipq.setUpperBound(base + 0, r)
            ipq.setLowerBound(base + 1, g)
            ipq.setUpperBound(base + 1, g)
            ipq.setLowerBound(base + 2, b)
            ipq.setUpperBound(base + 2, b)

        # incremental metadata
        if incremental:
            ica = ipq.getIncrementalConflictAnalyser()
            ica.setID(int(query_id))
            ica.setAncestors([int(a) for a in ancestors])
            ica.setRecordConflicts(len(added) >= 4)

        # solve once
        exit_code, vals, stats = MarabouCore.solve(ipq, self.options)

        # stats
        num_inc_tight = int(
            stats.getUnsignedAttribute(StatisticsUnsignedAttribute.NUM_INCREMENTAL_TIGHTENINGS)
        )
        self.total_incremental_tightenings += num_inc_tight
        if exit_code == "sat": # in ("TIMEOUT", "sat"):
            self.total_conflicts_when_sat += self._get_conflicts_num_this_solve(stats)



        if incremental:
            self.total_conflicts_recorded = int(
                ipq.getIncrementalConflictAnalyser().getRecordedConflictCount()
            )
        else:
            self.total_conflicts_recorded = 0

        # pop input-bounds frame
        ipq.pop()

        if exit_code in ("unsat", "sat", "TIMEOUT"):
            return exit_code
        raise RuntimeError(f"Unexpected Marabou exit code: {exit_code}")

    # ---------------------------
    # output writing
    # ---------------------------
    def _write_outputs(self, *, time_elapsed_done: float):
        # sat/unsat/timeout lists
        np.savetxt(self.unsat_path, np.asarray(self.unsat_set, dtype=int), fmt="%d")
        np.savetxt(self.sat_path, np.asarray(self.sat_set, dtype=int), fmt="%d")
        np.savetxt(self.timeout_path, np.asarray(self.timeout_set, dtype=int), fmt="%d")

        final_explanation_size = int(len(self.sat_set) + len(self.timeout_set))
        # In your logic, explanation consists of sat+timeout, freed are unsat.
        # Keep the invariant check:
        assert final_explanation_size == self.explanation_size, (
            f"Mismatch: final_explanation_size({final_explanation_size}) "
            f"!= explanation_size({self.explanation_size})"
        )

        summary = {
            "incremental": bool(self.incremental),
            "time_elapsed_sec": float(time_elapsed_done),
            "final_explanation_size": int(final_explanation_size),
            "total_incremental_tightenings": int(self.total_incremental_tightenings),
            "total_conflicts_recorded": int(self.total_conflicts_recorded),
            "total_conflicts_when_sat": int(self.total_conflicts_when_sat),
        }

        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)


# ============================================================
# Data loading helpers (GTSRB now; expand later if needed)
# ============================================================
def load_gtsrb_x_test(pickle_path: str) -> np.ndarray:
    with open(pickle_path, "rb") as handle:
        gtsrb = pickle.load(handle)
    x_test = gtsrb["x_test"].astype("float32") / 255.0
    return x_test


# ============================================================
# Runner
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True, help="Point index in the dataset (run_id).")
    parser.add_argument("--smoke", action="store_true", help="Write outputs under results/.../smoke/ instead of full/")
    parser.add_argument("--subset", action="store_true", help="Write outputs under results/.../subset/ instead of full/")

    args = parser.parse_args()

    # Fixed configuration (release)
    dataset = "gtsrb"
    model_name = "gtsrb-cnn"
    epsilon = 0.05
    index = int(args.index)
    indices = [22, 49, 57, 58, 61, 62, 79, 83, 97, 109, 122, 125, 142, 144, 154, 155, 178, 184, 197, 214, 215, 256, 268, 275, 291, 311, 323, 339, 404, 461, 462, 464, 465, 491, 493, 564, 604, 635, 637, 669, 673, 682, 702, 705, 708, 717, 737, 756, 765, 767, 769, 770, 771, 775, 780, 809, 811, 820, 837, 842, 844, 845, 858, 877, 885, 940, 961, 966, 969, 989]
    idx = indices[index]

    # Paths 
    gtsrb_pickle = "experiments/data/gtsrb.pickle"
    model_path_no_suffix = f"experiments/data/{model_name}"

    # Output root
    smoke_or_full_or_subset = "smoke" if args.smoke else "full" if not args.subset else "subset"
    run_id = str(idx)  # you locked: run_id = point index
    root = Path("experiments") / "results" / "explainability" / smoke_or_full_or_subset / f"point_{run_id}"
    root.mkdir(parents=True, exist_ok=True)


    baseline_dir = root / "baseline"
    incremental_dir = root / "incremental"
    baseline_dir.mkdir(exist_ok=True)
    incremental_dir.mkdir(exist_ok=True)

    # Load input
    assert dataset == "gtsrb", f"Only gtsrb is supported in this minimal version, got {dataset}"
    x_test = load_gtsrb_x_test(gtsrb_pickle)
    image = x_test[idx]

    for inc, out_dir in [(True, str(incremental_dir)), (False, str(baseline_dir))]:
        verix = VeriXMinimal(
            dataset=dataset,
            image=image,
            model_path_no_suffix=model_path_no_suffix,
            out_dir=out_dir,
            incremental=inc,
        )
        verix.new_traversal(epsilon=epsilon * 0.1)
        verix.compute_explanation(epsilon=epsilon)
        if inc and smoke_or_full_or_subset == "smoke":
            smoke_dir = Path("experiments") / "results" / "explainability" / smoke_or_full_or_subset
            smoke_dir.mkdir(parents=True, exist_ok=True)
            with open(smoke_dir / "success.json", "w") as f:
                json.dump({"success": True}, f)        


    # print(f"[done] wrote explainability outputs under: {root}")


if __name__ == "__main__":
    main()
