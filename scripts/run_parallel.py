#!/usr/bin/env python3
"""Run quantum search in parallel using MPI.

Usage:
    mpirun -n 4 python scripts/run_parallel.py

This will distribute simulations across 4 processes.
"""

from quantumsearch.parallel import run_parallel_search
from quantumsearch.parallel.utils import is_master, print_master
from quantumsearch.plotting import plot_site_occupations
import os


def main():
    # Create output directory
    os.makedirs('results/data', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

    # Define parameter sets for different simulations
    task_list = []

    # Vary graph size
    for N in [6, 8, 9]:
        task_list.append({
            'graph_config': {
                'graph_type': 'complete',
                'N': N
            },
            'time_config': {
                'T': 50,
                'number_of_time_steps': 200,
                'simulation_time_adjustment': False
            },
            'simulation_config': {
                'search_type': 'fermionic',
                'M': 2,
                'hopping_rate': None,
                'output': 'occupations'
            },
        })

    print_master(f"Running {len(task_list)} simulations in parallel...")

    # Run in parallel
    results = run_parallel_search(task_list)

    # Process results (only on master)
    if is_master() and results:
        print_master(f"\nCompleted {len(results)} simulations!")

        # Save or plot results
        for result in results:
            N = result['task']['graph_config']['N']
            sim = result['simulation']

            # Plot each result
            plot_site_occupations(sim, filename=f'results/plots/N_{N}.png')
            print(f"Saved plot for N={N}")


if __name__ == '__main__':
    main()
