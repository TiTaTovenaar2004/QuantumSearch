"""Script to load and analyze saved simulation results.

This script loads the data from the most recent simulation run and displays
key information about the results.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path to import quantumsearch modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantumsearch.parallel.mpi_runner import load_results
from quantumsearch.plotting import plot_estimated_success_probabilities


def display_summary(results, summary):
    """Display a summary of the loaded results."""

    print("\n" + "="*70)
    print("SIMULATION RESULTS SUMMARY")
    print("="*70)

    print(f"\nTimestamp: {summary['timestamp']}")
    print(f"Total tasks: {summary['total_tasks']}")
    print(f"Total simulation time: {summary['total_simulation_time']:.4f} seconds")
    print(f"Total estimation time: {summary['total_estimation_time']:.4f} seconds")

    print("\n" + "-"*70)
    print("INDIVIDUAL TASK RESULTS")
    print("-"*70)

    for result in results:
        print(f"\nTask {result['task_id']}: {result['graph_type']} graph (N={result['N']})")
        print(f"  Search type: {result['search_type']}, M={result['M']}")
        print(f"  Hopping rate: {result['hopping_rate']:.6f}")
        print(f"  Time points: {len(result['times'])}")
        print(f"  Simulation time: {result['simulation_time']:.4f}s")
        print(f"  Estimation time: {result['estimation_time']:.4f}s")

        if result['estimated_success_probabilities']:
            for i, est in enumerate(result['estimated_success_probabilities']):
                max_prob = np.max(est['probabilities'])
                max_time_idx = np.argmax(est['probabilities'])
                max_time = result['times'][max_time_idx]

                print(f"  Estimation {i+1}:")
                print(f"    Rounds: {est['rounds']}, Precision: {est['precision']}, Confidence: {est['confidence']}")
                print(f"    Max probability: {max_prob:.6f} at t={max_time:.6f}")

                # Display running time bounds if available
                if 'threshold' in est and 'lower_running_time' in est:
                    threshold = est['threshold']
                    lower_rt = est['lower_running_time']
                    upper_rt = est['upper_running_time']

                    if np.isinf(lower_rt):
                        print(f"    Threshold {threshold:.3f} never reached")
                    else:
                        print(f"    Threshold {threshold:.3f}: Running time ∈ [{lower_rt:.6f}, {upper_rt:.6f}]")


def main():
    """Main function to load and analyze results."""

    # Set data directory
    data_dir = 'results/data'

    # Load results using the load_results function from mpi_runner
    print("Loading simulation results...")
    results, summary = load_results(input_dir=data_dir)
    print(f"Successfully loaded {len(results)} simulation results\n")

    # Display summary
    display_summary(results, summary)

    # Generate plots
    print("\nGenerating plots...")
    plot_estimated_success_probabilities(
        results,
        output_dir='results/plots',
        timestamp=summary['timestamp']
    )

    print("\n" + "="*70)
    print("Data loaded successfully!")
    print(f"Access the data via: results (list of {len(results)} dictionaries)")
    print("="*70 + "\n")

    return results, summary


if __name__ == '__main__':
    results, summary = main()

