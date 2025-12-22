#!/usr/bin/env python3
"""Compute lowest running times for fermionic search across N and M values.

Usage:
    python scripts/run_running_times.py          # Serial execution
    mpirun -n 8 python scripts/run_running_times.py  # Parallel execution
"""

from quantumsearch.parallel import run_parallel_search
from quantumsearch.parallel.utils import is_master, print_master
import os
import pickle
import numpy as np


def main():
    # Create output directories
    os.makedirs('results/data', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

    # Define parameter sets for different simulations
    task_list = []
    thresholds = [i/10 for i in range(1, 10)]  # [0.1, 0.2, ..., 0.9]

    # Start small to avoid OOM - increase gradually once confirmed working
    for N in range(2, 4):  # Start with just N=2,3
        for M in range(1, N):
            task_list.append({
                'graph_config': {
                    'graph_type': 'complete',
                    'N': N
                },
                'time_config': {
                    'T': 50,
                    'number_of_time_steps': 100,  # Reduced to save memory
                    'simulation_time_adjustment': True
                },
                'simulation_config': {
                    'search_type': 'fermionic',
                    'M': M,
                    'hopping_rate': None,
                    'output': 'states'  # Need states for running time calculations
                },
                'task': {
                    'determine_lowest_running_times': True,
                    'thresholds': thresholds,
                    'stop_condition': 2
                }
            })

    print_master(f"Running {len(task_list)} simulations to compute lowest running times...")
    print_master(f"Total configurations: N=2..10, M=1..N for each N")

    results = run_parallel_search(task_list)

    # Compute lowest running times for each simulation
    if is_master() and results:
        print_master(f"\nCompleted {len(results)} simulations!")
        print_master("\nComputing lowest running times...")

        # Save raw results
        with open('results/data/running_times_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print_master("\nResults saved to results/data/running_times_results.pkl")

        # Create summary
        print_master("\nSummary of lowest running times:")
        for result in results:
          graph = result['graph']
          sim = result['simulation']
          print(graph.summary())
          print(sim.summary())



if __name__ == '__main__':
    main()
