#!/usr/bin/env python
"""
Parallel quantum search simulations using MPI.

This script demonstrates running multiple quantum search simulations in parallel
using MPI. Each task runs a simulation, estimates success probabilities, and
stores the results.

Usage:
    mpirun -n <num_processes> python run_parallel_simulations.py

Example:
    mpirun -n 4 python run_parallel_simulations.py
"""

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
    times = np.linspace(0, 10, 50)

    # Define task configurations
    task_configs = [
        # Task 1: Bosonic search on complete graph N=4, M=2
        {
            'graph_config': {
                'graph_type': 'complete',
                'N': 4,
            },
            'simulation_config': {
                'search_type': 'bosonic',
                'M': 2,
            },
            'times': times,
            'estimation_config': {
                'number_of_rounds': [1, 3, 5, 7],  # Multiple rounds
                'threshold': 0.8,
                'precision': 0.02,
                'confidence': 0.95,
            }
        },

        # Task 2: Fermionic search on complete graph N=5, M=2
        {
            'graph_config': {
                'graph_type': 'complete',
                'N': 5,
            },
            'simulation_config': {
                'search_type': 'fermionic',
                'M': 2,
            },
            'times': times,
            'estimation_config': {
                'number_of_rounds': [1, 3, 5, 7],  # Multiple rounds
                'threshold': 0.8,
                'precision': 0.02,
                'confidence': 0.95,
            }
        },

        # Task 3: Bosonic search on cycle graph N=6, M=2
        {
            'graph_config': {
                'graph_type': 'cycle',
                'N': 6,
            },
            'simulation_config': {
                'search_type': 'bosonic',
                'M': 2,
            },
            'times': times,
            'estimation_config': {
                'number_of_rounds': [1, 3, 5, 7],  # Multiple rounds
                'threshold': 0.8,
                'precision': 0.02,
                'confidence': 0.95,
            }
        },

        # Task 4: Bosonic search on Erdos-Renyi graph N=6, M=2
        {
            'graph_config': {
                'graph_type': 'erdos-renyi',
                'N': 6,
                'p': 0.5,
            },
            'simulation_config': {
                'search_type': 'bosonic',
                'M': 2,
            },
            'times': times,
            'estimation_config': {
                'number_of_rounds': [1, 3, 5, 7],  # Multiple rounds
                'threshold': 0.8,
                'precision': 0.02,
                'confidence': 0.95,
            }
        },
    ]

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

            if result['estimated_success_probabilities']:
                print(f"  Estimated success probabilities for {len(result['estimated_success_probabilities'])} different rounds:")
                for est_result in result['estimated_success_probabilities']:
                    max_prob = np.max(est_result['probabilities'])
                    print(f"    Rounds={est_result['rounds']}: Max prob={max_prob:.4f}, "
                          f"Precision={est_result['precision']}, "
                          f"Confidence={est_result['confidence']}")
                    if 'lower_running_time' in est_result and not np.isinf(est_result['lower_running_time']):
                        print(f"      Running time: [{est_result['lower_running_time']:.4f}, "
                              f"{est_result['upper_running_time']:.4f}]")

        print()
        print("=" * 60)
        total_sim_time = sum(r['simulation_time'] for r in results)
        total_est_time = sum(r['estimation_time'] for r in results)
        print(f"Total simulation time: {total_sim_time:.2f} s")
        print(f"Total estimation time: {total_est_time:.2f} s")
        print(f"Total time: {total_sim_time + total_est_time:.2f} s")
        print("=" * 60)

        # Save results to disk
        save_results(results, output_dir='results/data')


if __name__ == '__main__':
    main()
