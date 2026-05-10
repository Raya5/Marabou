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

class IDGen:
    def __init__(self):
        self.val = 1
    def next(self):
        v = self.val
        self.val += 1
        return v

def verify_partition(query, timeout, timeout_factor, depth, max_depth, ica, ancestors, id_gen, partition_path="ROOT"):
    """
    Verifies a specific partition defined by the current set of constraints.
    If timeout and depth < max_depth, splits the widest input variable.
    """
    
    # Create options with specified timeout.
    # Ensure timeout is an integer string for Marabou options if required, or int
    timeout_int = int(timeout)
    options = Marabou.createOptions(timeoutInSeconds=timeout_int, verbosity=0)
    options._incremental = True

    # Get a unique ID for this query
    my_id = id_gen.next()

    # Configure the shared ICA for this node:
    # 1. Set the current query ID
    # 2. Tell it which ancestors to inherit conflicts from
    ica.setID(my_id)
    ica.setAncestors(ancestors)

    # Attach the analyzer
    query.setIncrementalConflictAnalyser(ica)
    
    # Attempt to solve
    # verbose=False to control output format
    start_time = time.time()
    exit_code, vals, stats = Marabou.solve_query(query, options=options, verbose=False)
    elapsed_time = time.time() - start_time
    
    num_conflicts = 0
    num_tightenings = 0
    
    # Get conflicts from ICA
    if ica and hasattr(ica, "getRecordedConflictCount"):
        num_conflicts = ica.getRecordedConflictCount()
    
    # Get tightenings from stats
    if stats:
        num_tightenings = stats.getUnsignedAttribute(MarabouCore.StatisticsUnsignedAttribute.NUM_INCREMENTAL_TIGHTENINGS)

    # Report actual result with timing
    if exit_code == "sat" or exit_code == "unsat":
        print(f"Depth {depth} Partition {partition_path}: {exit_code} (time: {elapsed_time:.3f}s, conflicts: {num_conflicts}, tightenings: {num_tightenings})")
        return

    # If TIMEOUT, check depth
    if depth >= max_depth:
        print(f"Depth {depth} Partition {partition_path}: TIMEOUT (time: {elapsed_time:.3f}s, conflicts: {num_conflicts}, tightenings: {num_tightenings})")
        return
    else:
        print(f"Depth {depth} Partition {partition_path}: TIMEOUT (time: {elapsed_time:.3f}s, conflicts: {num_conflicts}, tightenings: {num_tightenings}), splitting...")
        
        # Identify split variable
        split_var = get_widest_input_variable(query)
        
        if split_var is None:
             print(f"Depth {depth} Partition {partition_path}: Cannot split (no finite bounded inputs).")
             return

        lb = query.getLowerBound(split_var)
        ub = query.getUpperBound(split_var)
        mid = (lb + ub) / 2.0
        
        # New ancestor list for children
        new_ancestors = ancestors + [my_id]

        # Branch 1: Lower half
        query.push()
        query.tightenUpperBound(split_var, mid)
        verify_partition(query, timeout * timeout_factor, timeout_factor, depth + 1, max_depth, ica, new_ancestors, id_gen, partition_path + "L")
        query.pop()
        
        # Branch 2: Upper half
        query.push()
        query.tightenLowerBound(split_var, mid)
        verify_partition(query, timeout * timeout_factor, timeout_factor, depth + 1, max_depth, ica, new_ancestors, id_gen, partition_path + "R")
        query.pop()

def main():
    parser = argparse.ArgumentParser(description="Incremental verification with binary tree splitting.")
    parser.add_argument("input_query", help="Path to the input query file")
    parser.add_argument("--timeout", type=float, default=5.0, help="Initial timeout in seconds for each partition (default: 5.0)")
    parser.add_argument("--timeout-factor", type=float, default=1.0, help="Factor to multiply timeout by at each depth (default: 1.0)")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum recursion depth for timeouts (default: 3)")
    
    args = parser.parse_args()
    
    print(f"Loading query from {args.input_query}...")
    query = Marabou.load_query(args.input_query)

    # Build the shared IncrementalConflictAnalyser
    # reuseAllConflicts=True (to store the conflict pool)
    # autoInheritance=False (we will manually set ancestors)
    ica = MarabouCore.buildIncrementalConflictAnalyser()

    # Initialize ID generator
    id_gen = IDGen()
    
    print(f"Starting verification with timeout={args.timeout}s, factor={args.timeout_factor}, max_depth={args.max_depth}...")
    verify_partition(query, args.timeout, args.timeout_factor, 0, args.max_depth, ica, [], id_gen)

if __name__ == "__main__":
    main()
