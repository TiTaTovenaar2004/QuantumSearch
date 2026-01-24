#!/usr/bin/env python
"""
Parallel quantum search simulations using MPI.

This script demonstrates running multiple quantum search simulations in parallel
using MPI. Each task runs a simulation, estimates success probabilities, and
stores the results.

Usage:
    mpirun -n <num_processes> /home/aron/Tijmen/QuantumSearch/.venv/bin/python /home/aron/Tijmen/QuantumSearch/scripts/run_parallel_simulations.py

Example:
    mpirun -n 8 /home/aron/Tijmen/QuantumSearch/.venv/bin/python /home/aron/Tijmen/QuantumSearch/scripts/run_parallel_simulations.py
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from quantumsearch.parallel.mpi_runner import run_parallel_simulations, save_results


def main():
    """
    Configure and run parallel quantum search simulations.
    """

    # Define time points for simulations
    times = np.linspace(0, 40, 400)

    # Define task configurations
    task_configs = []
    for N in range(5, 6):
        for M in range(2, 3):
            p_values = np.random.uniform(0.4, 1.0, 64).tolist()
            for p in p_values:
                config = {
                    'graph_config': {
                        'graph_type': 'erdos-renyi',
                        'N': N,
                        'marked_vertex': 0,
                        'p': p
                    },
                    'simulation_config': {
                        'search_type': 'fermionic',
                        'M': M,
                        'hopping_rate': None
                    },
                    'times': times,
                    'estimation_config': {
                        'number_of_rounds': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        'threshold': 0.8,
                        'precision': 0.01,
                        'confidence': 0.9999,
                        'fast_mode': False
                    }
                }
                task_configs.append(config)

    print("=" * 60)
    print("PARALLEL QUANTUM SEARCH SIMULATIONS")
    print("=" * 60)
    print(f"Total tasks: {len(task_configs)}")
    print()

    # Run parallel simulations
    results = run_parallel_simulations(task_configs)

    # Process and display results (only on master rank)
    if results is not None:
        print()
        print("=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        for i, result in enumerate(results):
            print(f"\nTask {i + 1}:")
            print(f"  Graph: {result['graph_type']}, N={result['N']}")
            print(f"  Search: {result['search_type']}, M={result['M']}")
            print(f"  Hopping rate: {result['hopping_rate']:.4f}")
            print(f"  Simulation time: {result['simulation_time']:.2f} s")
            print(f"  Estimation time: {result['estimation_time']:.2f} s")

            if 'lower_running_times' in result and 'upper_running_times' in result:
                # Get number_of_rounds from task config
                number_of_rounds = result['task_config']['estimation_config']['number_of_rounds']
                if isinstance(number_of_rounds, int):
                    number_of_rounds = [number_of_rounds]

                print(f"  Running times for {len(number_of_rounds)} different rounds:")
                for idx, rounds in enumerate(number_of_rounds):
                    lower_rt = result['lower_running_times'][idx]
                    upper_rt = result['upper_running_times'][idx]

                    if not np.isinf(lower_rt) and not np.isinf(upper_rt):
                        print(f"    Rounds={rounds}: [{lower_rt:.4f}, {upper_rt:.4f}]")
                    else:
                        print(f"    Rounds={rounds}: Threshold never reached")

        print()
        print("=" * 60)
        total_sim_time = sum(r['simulation_time'] for r in results)
        total_est_time = sum(r['estimation_time'] for r in results)
        print(f"Total simulation time: {total_sim_time:.2f} s")
        print(f"Total estimation time: {total_est_time:.2f} s")
        print(f"Total time: {total_sim_time + total_est_time:.2f} s")
        print("=" * 60)

        # Save results to disk (use absolute path relative to script location)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, '..')
        output_dir = os.path.join(project_root, 'results', 'data')
        save_results(results, output_dir=output_dir)


if __name__ == '__main__':
    main()
