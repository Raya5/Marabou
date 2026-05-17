#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import warnings
import time
from pathlib import Path

# Suppress warnings caused by tensorflow/etc if any
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

_MARABOU_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_MARABOU_ROOT))

from maraboupy import Marabou, MarabouCore

def get_widest_input_variable(query):
    """
    Finds the input variable with the widest valid range (upper - lower).
    """
    num_inputs = query.getNumInputVariables()
    widest_var = None
    max_width = -1.0

    for i in range(num_inputs):
        var = query.inputVariableByIndex(i)
        lb = query.getLowerBound(var)
        ub = query.getUpperBound(var)
        
        # Check validity and finiteness
        if lb == float('-inf') or ub == float('inf'):
            continue
            
        width = ub - lb
        if width > max_width:
            max_width = width
            widest_var = var
            
    return widest_var

def verify_partition(input_query_path, timeout, timeout_factor, depth, max_depth, partition_path="ROOT", bounds_constraints=None):
    """
    Verifies a specific partition by creating a fresh query with bounds constraints.
    If timeout and depth < max_depth, splits the widest input variable.
    
    Args:
        input_query_path: Path to the original input query file
        timeout: Timeout in seconds for this partition
        timeout_factor: Factor to multiply timeout by at each depth
        depth: Current recursion depth
        max_depth: Maximum recursion depth
        partition_path: String representing the path through the search tree
        bounds_constraints: List of (var_index, lower_bound, upper_bound) tuples to apply
    """
    # Load a fresh query from file
    query = Marabou.load_query(input_query_path)
    
    # Apply bounds constraints if any
    if bounds_constraints:
        for var_idx, lb, ub in bounds_constraints:
            var = query.inputVariableByIndex(var_idx)
            query.setLowerBound(var, lb)
            query.setUpperBound(var, ub)
    
    # Create options with specified timeout
    timeout_int = int(timeout)
    options = Marabou.createOptions(timeoutInSeconds=timeout_int, verbosity=0)
    
    # Attempt to solve
    start_time = time.time()
    exit_code, vals, stats = Marabou.solve_query(query, options=options, verbose=False)
    elapsed_time = time.time() - start_time
    
    # Report actual result with timing
    if exit_code == "sat" or exit_code == "unsat":
        print(f"Depth {depth} Partition {partition_path}: {exit_code} (time: {elapsed_time:.3f}s)")
        return

    # If TIMEOUT, check depth
    if depth >= max_depth:
        print(f"Depth {depth} Partition {partition_path}: TIMEOUT (time: {elapsed_time:.3f}s)")
        return
    else:
        print(f"Depth {depth} Partition {partition_path}: Timeout (time: {elapsed_time:.3f}s), splitting...")
        
        # Identify split variable
        split_var = get_widest_input_variable(query)
        
        if split_var is None:
             print(f"Depth {depth} Partition {partition_path}: Cannot split (no finite bounded inputs).")
             return

        # Get the variable index
        num_inputs = query.getNumInputVariables()
        split_var_idx = None
        for i in range(num_inputs):
            if query.inputVariableByIndex(i) == split_var:
                split_var_idx = i
                break
        
        lb = query.getLowerBound(split_var)
        ub = query.getUpperBound(split_var)
        mid = (lb + ub) / 2.0
        
        # Create new bounds constraints for both branches
        new_bounds = bounds_constraints.copy() if bounds_constraints else []
        
        # Branch 1: Lower half
        lower_bounds = new_bounds.copy()
        lower_bounds.append((split_var_idx, lb, mid))
        verify_partition(input_query_path, timeout * timeout_factor, timeout_factor, depth + 1, max_depth, partition_path + "L", lower_bounds)
        
        # Branch 2: Upper half
        upper_bounds = new_bounds.copy()
        upper_bounds.append((split_var_idx, mid, ub))
        verify_partition(input_query_path, timeout * timeout_factor, timeout_factor, depth + 1, max_depth, partition_path + "R", upper_bounds)

def main():
    parser = argparse.ArgumentParser(description="Baseline verification with input splitting (no incremental).")
    parser.add_argument("input_query", help="Path to the input query file")
    parser.add_argument("--timeout", type=float, default=5.0, help="Initial timeout in seconds for each partition (default: 5.0)")
    parser.add_argument("--timeout-factor", type=float, default=1.0, help="Factor to multiply timeout by at each depth (default: 1.0)")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum recursion depth for timeouts (default: 3)")
    
    args = parser.parse_args()
    
    print(f"Loading query from {args.input_query}...")
    # Just verify the file exists
    try:
        query = Marabou.load_query(args.input_query)
    except:
        print(f"Error: Could not load query from {args.input_query}")
        return
    
    print(f"Starting baseline verification with input splitting (no incremental)")
    print(f"timeout={args.timeout}s, factor={args.timeout_factor}, max_depth={args.max_depth}...")
    verify_partition(args.input_query, args.timeout, args.timeout_factor, 0, args.max_depth)

if __name__ == "__main__":
    main()
