#!/usr/bin/env python3
"""
experiments/scripts/run_robustness.py

Run robustness-radius (epsilon) search for one MNIST test point,
in two modes:
  - baseline (non-incremental)
  - incremental (conflict reuse)

Outputs (per point, per mode):
experiments/results/robustness/<smoke|full>/point_<db_idx>/<baseline|incremental>/summary.json

Fixed configuration is hardcoded below (per your release plan).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------
# sys.path: make maraboupy importable from iv/Marabou, regardless of cwd
# File location: iv/Marabou/experiments/scripts/run_robustness.py
# parents: [0]=scripts, [1]=experiments, [2]=Marabou
# ---------------------------------------------------------------------
_MARABOU_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_MARABOU_ROOT))

from maraboupy import Marabou, MarabouCore, MarabouUtils  # noqa: E402
from maraboupy.MarabouCore import StatisticsUnsignedAttribute  # noqa: E402

# ---------------------------------------------------------------------
# Fixed config (release)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class RobustnessConfig:
    onnx_path: str = "experiments/data/mnist.onnx"
    eps_min: float = 0.0
    eps_max: float = 0.5
    precision: float = 1e-3
    timeout_per_query_sec: float = 120.0
    all_timeout_sec: float = 1200.0
    margin: float = 0.0
    input_min: float = 0.0
    input_max: float = 1.0

    # smoke override
    smoke_timeout_per_query_sec: float = 120.0
    smoke_all_timeout_sec: float = 120.0
    smoke_precision: float = 5e-2


# ---------------------------------------------------------------------
# MNIST loading 
# ---------------------------------------------------------------------

def load_mnist_test_openml() -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      x_test: uint8 (10000, 28, 28)
      y_test: uint8 (10000,)
    """
    from sklearn.datasets import fetch_openml  # local import on purpose

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.uint8).reshape(-1, 28, 28)
    y = mnist.target.astype(np.uint8)

    x_test = X[60000:]
    y_test = y[60000:]
    return x_test, y_test

def get_point_by_db_index(db_idx: int) -> Tuple[np.ndarray, int]:
    """
    db_idx is the MNIST *test-set* index in [0, 9999].
    Returns:
      point_flat: float32 shape (784,)
      true_label: int
    """
    x_test_u8, y_test_u8 = load_mnist_test_openml()
    if not (0 <= db_idx < x_test_u8.shape[0]):
        raise ValueError(f"db_idx must be in [0, {x_test_u8.shape[0]-1}]")

    img = x_test_u8[db_idx].astype(np.float32) / 255.0  # (28,28)
    point_flat = img.reshape(-1).astype(np.float32)     # (784,)
    true_label = int(y_test_u8[db_idx])
    return point_flat, true_label


# ---------------------------------------------------------------------
# ORT helper
# ---------------------------------------------------------------------

def ort_predict_label(onnx_path: str, point_flat: np.ndarray) -> int:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    # Old code used (1, 784, 1)
    x = point_flat.reshape(1, -1, 1).astype(np.float32)
    logits = sess.run(None, {input_name: x})[0].reshape(-1)
    return int(np.argmax(logits))


# ---------------------------------------------------------------------
# Marabou robustness query building (disjunction: misclassification)
# ---------------------------------------------------------------------

def add_output_constraints_for_misclassification(network, target_label: int, margin: float = 0.0):
    if margin < 0.0:
        raise ValueError("margin must be non-negative")

    # Flatten output vars
    flat_outputs = [int(v) for arr in network.outputVars for v in arr.flatten()]
    if not flat_outputs:
        raise RuntimeError("No output variables found")

    if not (0 <= int(target_label) < len(flat_outputs)):
        raise ValueError(f"target_label {target_label} out of range (num outputs={len(flat_outputs)})")

    y_t = flat_outputs[int(target_label)]
    disjuncts = []
    for k, y_k in enumerate(flat_outputs):
        if k == int(target_label):
            continue
        eq = MarabouUtils.Equation(MarabouCore.Equation.LE)
        eq.addAddend(1.0, y_t)
        eq.addAddend(-1.0, y_k)
        eq.setScalar(float(-margin))  # y_t - y_k <= -margin
        disjuncts.append([eq])

    network.addDisjunctionConstraint(disjuncts)


def build_options(timeout_per_query_sec: float, incremental: bool):
    options = Marabou.createOptions(timeoutInSeconds=int(timeout_per_query_sec))
    options._verbosity = 0
    options._incremental = bool(incremental)
    # Ensure DnC / SNC is off (important for incremental)
    if hasattr(options, "_snc"):
        options._snc = False
    return options


def compute_input_box_bounds(point_flat: np.ndarray, *, epsilon: float, input_min: float, input_max: float):
    p = np.asarray(point_flat, dtype=np.float32).reshape(-1)
    lb = np.maximum(p - float(epsilon), float(input_min))
    ub = np.minimum(p + float(epsilon), float(input_max))
    if np.any(lb > ub + 1e-12):
        bad = int(np.argmax(lb > ub + 1e-12))
        raise ValueError(f"Invalid box at dim {bad}: lb={lb[bad]} > ub={ub[bad]}")
    return lb, ub


def apply_input_bounds_to_ipq(ipq, lb: np.ndarray, ub: np.ndarray):
    n = ipq.getNumInputVariables()
    flat_input_vars = [ipq.inputVariableByIndex(i) for i in range(n)]
    if lb.shape != (n,) or ub.shape != (n,):
        raise ValueError(f"lb/ub must be shape ({n},), got {lb.shape} / {ub.shape}")
    for i, var in enumerate(flat_input_vars):
        ipq.setLowerBound(int(var), float(lb[i]))
        ipq.setUpperBound(int(var), float(ub[i]))


def _get_unsigned_stat(stats, attr) -> Optional[int]:
    """
    Safe extractor for Marabou unsigned stats.
    Returns None if missing/unavailable.
    """
    if stats is None:
        return None
    try:
        return int(stats.getUnsignedAttribute(attr))
    except Exception:
        return None


# ---------------------------------------------------------------------
# One epsilon solve
# ---------------------------------------------------------------------

def solve_at_epsilon(
    *,
    ipq,
    point_flat: np.ndarray,
    epsilon: float,
    marabou_options,
    incremental: bool,
    ica=None,
) -> Dict[str, Any]:
    """
    Returns a history entry:
      result: SAT/UNSAT/TIMEOUT
      runtime_ours
      num_incremental_tightenings (0 if baseline)
      conflicts_recorded (only meaningful for incremental)
      min_sat_epsilon (for SAT; else None)
    """
    lb, ub = compute_input_box_bounds(point_flat, epsilon=epsilon, input_min=0.0, input_max=1.0)

    # incremental epsilon hook if available
    if incremental and ica is not None:
        try:
            ica.setNewEpsilon(float(epsilon))
        except Exception:
            pass

    ipq.push()
    apply_input_bounds_to_ipq(ipq, lb, ub)

    t0 = time.perf_counter()
    exit_code, vals, stats = MarabouCore.solve(ipq, marabou_options)
    t1 = time.perf_counter()

    ipq.pop()

    if exit_code == "sat":
        result = "SAT"
    elif exit_code == "unsat":
        result = "UNSAT"
    elif exit_code == "TIMEOUT":
        result = "TIMEOUT"
    else:
        # treat unexpected as TIMEOUT to keep the search stable
        result = "TIMEOUT"

    num_tight = 0
    if incremental:
        v = _get_unsigned_stat(stats, StatisticsUnsignedAttribute.NUM_INCREMENTAL_TIGHTENINGS)
        num_tight = int(v or 0)

    # --- conflicts counters ---
    conflicts_recorded = 0
    if incremental and ica is not None:
        try:
            conflicts_recorded = int(ica.getRecordedConflictCount())
        except Exception:
            conflicts_recorded = 0

    # SAT min epsilon extraction
    min_sat_epsilon = None
    if result == "SAT":
        try:
            n = ipq.getNumInputVariables()
            in_vars = [ipq.inputVariableByIndex(i) for i in range(n)]
            x_star = np.array([vals[v] for v in in_vars], dtype=np.float64)
            diffs = np.abs(x_star - point_flat.astype(np.float64))
            min_sat_epsilon = min(float(np.max(diffs)), float(epsilon))
        except Exception:
            min_sat_epsilon = None

    return {
        "epsilon": float(epsilon),
        "result": result,
        "runtime_ours": float(t1 - t0),
        "incremental": bool(incremental),
        "num_incremental_tightenings": int(num_tight),
        "exit_code": str(exit_code),
        "conflicts_recorded": int(conflicts_recorded),
        "min_sat_epsilon": min_sat_epsilon,
    }


# ---------------------------------------------------------------------
# Binary robustness search (bracketed)
# ---------------------------------------------------------------------

def find_robustness_binary(
    *,
    solve_with_epsilon,
    eps_min: float,
    eps_max: float,
    precision: float,
    all_timeout_sec: float,
) -> Tuple[List[Dict[str, Any]], Optional[float], Optional[float]]:
    """
    Returns:
      history
      min_unsat (largest UNSAT observed)
      max_sat  (smallest SAT observed)
    TIMEOUT does not move the bracket.
    """
    t_start = time.perf_counter()

    def time_left() -> float:
        return float(all_timeout_sec) - (time.perf_counter() - t_start)

    history: List[Dict[str, Any]] = []
    min_unsat = None
    max_sat = None

    # Endpoint probes
    if time_left() <= 0:
        return history, min_unsat, max_sat

    e_max = solve_with_epsilon(eps_max)
    history.append(e_max)
    if e_max["result"] == "SAT":
        max_sat = float(e_max.get("min_sat_epsilon") or eps_max)
    elif e_max["result"] == "UNSAT":
        # cannot bracket SAT side
        return history, float(eps_max), float(eps_max)

    if time_left() <= 0:
        return history, min_unsat, max_sat

    e_min = solve_with_epsilon(eps_min)
    history.append(e_min)
    if e_min["result"] == "UNSAT":
        min_unsat = float(eps_min)
    elif e_min["result"] == "SAT":
        # non-robust already at eps_min
        return history, float(eps_min), float(eps_min)

    # Main loop
    P = 0.5
    direction_down_from_max = True  # True: probe from max downward; False: probe from min upward

    # If one side is unknown due to TIMEOUT, we still proceed, but precision check needs both.
    while time_left() > 0:
        if min_unsat is None or max_sat is None:
            # we don't have a bracket yet; probe mid of [eps_min, eps_max] but conservative
            lo = eps_min if min_unsat is None else min_unsat
            hi = eps_max if max_sat is None else max_sat
        else:
            lo, hi = min_unsat, max_sat
            if (hi - lo) <= precision:
                break

        width = hi - lo
        step = width * P
        eps = (hi - step) if direction_down_from_max else (lo + step)

        # clamp
        eps = max(lo, min(hi, eps))

        entry = solve_with_epsilon(eps)
        history.append(entry)

        if entry["result"] == "SAT":
            max_sat = float(entry.get("min_sat_epsilon") or eps)
            P = 0.5
            direction_down_from_max = True
        elif entry["result"] == "UNSAT":
            min_unsat = float(eps)
            P = 0.5
            direction_down_from_max = True
        else:
            # TIMEOUT backoff
            P *= 0.5
            direction_down_from_max = not direction_down_from_max
            if P < 2 ** -30:
                break

    return history, min_unsat, max_sat


# ---------------------------------------------------------------------
# Running + output
# ---------------------------------------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Dict[str, Any]):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def summarize_history(history: List[Dict[str, Any]], *, incremental: bool) -> Dict[str, Any]:
    num_sat = sum(1 for e in history if e.get("result") == "SAT")
    num_unsat = sum(1 for e in history if e.get("result") == "UNSAT")
    num_timeout = sum(1 for e in history if e.get("result") == "TIMEOUT")
    total_tight = sum(int(e.get("num_incremental_tightenings", 0) or 0) for e in history)

    # recorded conflicts: take last entry's conflicts_recorded (monotone counter)
    total_conflicts_recorded = 0
    if incremental:
        for e in reversed(history):
            if "conflicts_recorded" in e:
                total_conflicts_recorded = int(e["conflicts_recorded"])
                break

    return {
        "num_sat": int(num_sat),
        "num_unsat": int(num_unsat),
        "num_timeout": int(num_timeout),
        "total_incremental_tightenings": int(total_tight),
        "total_conflicts_recorded": int(total_conflicts_recorded),
        "total_solver_calls": int(len(history)),
        "total_runtime_ours": float(sum(float(e.get("runtime_ours", 0.0) or 0.0) for e in history)),
    }


def run_mode(
    *,
    cfg: RobustnessConfig,
    point_flat: np.ndarray,
    label: int,
    incremental: bool,
) -> Dict[str, Any]:
    options = build_options(cfg.timeout_per_query_sec, incremental=incremental)

    network = Marabou.read_onnx(cfg.onnx_path)
    add_output_constraints_for_misclassification(network, target_label=label, margin=cfg.margin)

    ipq = network.getInputQuery()
    ipq.push()

    ica = None
    if incremental:
        ica = MarabouCore.buildIncrementalConflictAnalyser()
        ipq.setIncrementalConflictAnalyser(ica)

    def solve_with_epsilon(eps: float):
        return solve_at_epsilon(
            ipq=ipq,
            point_flat=point_flat,
            epsilon=eps,
            marabou_options=options,
            incremental=incremental,
            ica=ica,
        )

    t0 = time.perf_counter()
    history, min_unsat, max_sat = find_robustness_binary(
        solve_with_epsilon=solve_with_epsilon,
        eps_min=cfg.eps_min,
        eps_max=cfg.eps_max,
        precision=cfg.precision,
        all_timeout_sec=cfg.all_timeout_sec,
    )
    t1 = time.perf_counter()

    precision_width = None
    if (min_unsat is not None) and (max_sat is not None):
        precision_width = float(max_sat - min_unsat)

    status = "Solved"
    if len(history) == 0:
        status = "Error"
    elif any(e.get("result") == "TIMEOUT" for e in history) and (precision_width is None or precision_width > cfg.precision):
        # conservative: if we didn't reach the precision guarantee, call it timed out
        status = "Timedout"

    summary = summarize_history(history, incremental=incremental)
    return {
        "status": status,
        "time_sec": float(t1 - t0),
        "precision_width": precision_width,
        "min_unsat_epsilon": min_unsat,
        "max_sat_epsilon": max_sat,
        **summary,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, required=True, help="MNIST test-set index in [0, 9999]")
    p.add_argument("--smoke", action="store_true", help="Use smoke timeouts/precision and write under results/.../smoke/")
    p.add_argument("--subset", action="store_true", help="Write under results/.../subset/")
    args = p.parse_args()

    cfg = RobustnessConfig()
    if args.smoke:
        cfg = RobustnessConfig(
            onnx_path=cfg.onnx_path,
            eps_min=cfg.eps_min,
            eps_max=cfg.eps_max,
            precision=cfg.smoke_precision,
            timeout_per_query_sec=cfg.smoke_timeout_per_query_sec,
            all_timeout_sec=cfg.smoke_all_timeout_sec,
            margin=cfg.margin,
            input_min=cfg.input_min,
            input_max=cfg.input_max,
        )

    db_idx = int(args.index)
    point_flat, true_label = get_point_by_db_index(db_idx)

    pred_label = ort_predict_label(cfg.onnx_path, point_flat)
    is_correct = (pred_label == true_label)

    tier = "smoke" if args.smoke else "full" if not args.subset else "subset"
    out_root =  Path("experiments") / "results" / "robustness" / tier / f"point_{db_idx}"

    # If misclassified: write a baseline+incremental summary with skipped status (easy downstream)
    if not is_correct:
        for mode in ["baseline", "incremental"]:
            mode_dir = out_root / mode
            ensure_dir(mode_dir)
            write_json(mode_dir / "summary.json", {
                "point_index_in_database": db_idx,
                "mode": mode,
                "incremental": (mode == "incremental"),
                "is_correctly_classified": False,
                "true_label": int(true_label),
                "predicted_label": int(pred_label),
                "status": "SkippedMisclassified",
            })
        # print(f"[OK] point_{db_idx} skipped (misclassified). Wrote summaries under {out_root}")
        return

    # Run baseline then incremental
    baseline_dir = out_root / "baseline"
    inc_dir = out_root / "incremental"
    ensure_dir(baseline_dir)
    ensure_dir(inc_dir)

    r_base = run_mode(cfg=cfg, point_flat=point_flat, label=true_label, incremental=False)
    r_inc = run_mode(cfg=cfg, point_flat=point_flat, label=true_label, incremental=True)

    write_json(baseline_dir / "summary.json", {
        "point_index_in_database": db_idx,
        "mode": "baseline",
        "incremental": False,
        "is_correctly_classified": True,
        **r_base,
    })
    write_json(inc_dir / "summary.json", {
        "point_index_in_database": db_idx,
        "mode": "incremental",
        "incremental": True,
        "is_correctly_classified": True,
        **r_inc,
    })

    print(f"[done] wrote robustness outputs to: {out_root}")


if __name__ == "__main__":
    main()