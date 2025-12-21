#!/usr/bin/env python3
"""Run parameter sweep across multiple dimensions.

Usage:
    python scripts/run_batch.py          # Serial execution
    mpirun -n 8 python scripts/run_batch.py  # Parallel execution
"""

from quantumsearch.parallel import run_parallel_search
from quantumsearch.parallel.utils import is_master, print_master
import os
import pickle


def main():
    # Create output directories
    os.makedirs('results/data', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

    # Define parameter sets for different simulations
    task_list = []

    # Vary graph size
    for N in [5, 10, 15, 20]:
        task_list.append({
            'graph_config': {
                'graph_type': 'line',
                'N': N
            },
            'time_config': {
                'T': 50,
                'number_of_time_steps': 200,
                'simulation_time_adjustment': False
            },
            'simulation_config': {
                'search_type': 'fermionic',
                'M': 3,
                'hopping_rate': None,
                'output': 'occupations'
            },
        })

    print_master(f"Running {len(task_list)} simulations in parallel...")

    results = run_parallel_search(task_list)

    # Save results
    if is_master() and results:
        print_master(f"\nCompleted {len(results)} simulations!")

        # Save raw results
        with open('results/data/sweep_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print_master("Results saved to results/data/sweep_results.pkl")

        # Create summary
        print_master("\nSummary:")
        for result in results:
            M = result['task']['simulation_config']['M']
            N = result['task']['graph_config']['N']
            calc_time = result['simulation'].params['simulation calculation time']
            print(f"  M={M}, N={N}: {calc_time:.2f}s")


if __name__ == '__main__':
    main()
