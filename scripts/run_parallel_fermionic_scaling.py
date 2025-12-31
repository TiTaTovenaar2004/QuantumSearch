#!/usr/bin/env python
"""
Parallel quantum search simulations using MPI.

This script demonstrates running multiple quantum search simulations in parallel
using MPI. Each task runs a simulation, estimates success probabilities, and
stores the results.

Usage:
    mpirun -n <num_processes> /home/aron/Tijmen/QuantumSearch/.venv/bin/python /home/aron/Tijmen/QuantumSearch/scripts/run_parallel_fermionic_scaling.py

Example:
    mpirun -n 8 /home/aron/Tijmen/QuantumSearch/.venv/bin/python /home/aron/Tijmen/QuantumSearch/scripts/run_parallel_fermionic_scaling.py
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
    times = np.linspace(0, 20, 100)

    # Define task configurations
    task_configs = []
    for N in range(3, 6):
        for M in range(2, N):
            config = {
                'graph_config': {
                    'graph_type': 'complete',
                    'N': N,
                    'marked_vertex': 0
                },
                'simulation_config': {
                    'search_type': 'fermionic',
                    'M': M,
                    'hopping_rate': None
                },
                'times': times,
                'estimation_config': {
                    'number_of_rounds': [1, 2, 3, 4, 5],
                    'threshold': 0.8,
                    'precision': 0.01,
                    'confidence': 0.99,
                    'fast_mode': True
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

            if result['estimated_success_probabilities']:
                print(f"  Estimated success probabilities for {len(result['estimated_success_probabilities'])} different rounds:")
                for est_result in result['estimated_success_probabilities']:
                    mode = 'fast' if 'estimated_locations' in est_result else 'slow'
                    print(f"    Rounds={est_result['rounds']} ({mode} mode): "
                          f"Precision={est_result['precision']}, "
                          f"Confidence={est_result['confidence']}")
                    if 'lower_running_time' in est_result and not np.isinf(est_result['lower_running_time']):
                        print(f"      Running time: [{est_result['lower_running_time']:.4f}, "
                              f"{est_result['upper_running_time']:.4f}]")
                    elif 'lower_running_time' in est_result:
                        print(f"      Threshold never reached")

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
