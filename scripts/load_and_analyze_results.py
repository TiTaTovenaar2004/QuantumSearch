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

        # Display running times from new array format
        if 'lower_running_times' in result and 'upper_running_times' in result:
            lower_rts = result['lower_running_times']
            upper_rts = result['upper_running_times']

            # Get number_of_rounds from task_config if available, otherwise infer from array length
            if 'task_config' in result and 'estimation_config' in result['task_config']:
                number_of_rounds = result['task_config']['estimation_config']['number_of_rounds']
                threshold = result['task_config']['estimation_config']['threshold']
                if isinstance(number_of_rounds, int):
                    number_of_rounds = [number_of_rounds]
            else:
                # Fallback: create placeholder round numbers
                number_of_rounds = list(range(1, len(lower_rts) + 1))
                threshold = None

            print(f"  Running times:")
            for idx, rounds in enumerate(number_of_rounds):
                lower_rt = lower_rts[idx]
                upper_rt = upper_rts[idx]

                if np.isinf(lower_rt) or np.isinf(upper_rt):
                    print(f"    Rounds={rounds}: Threshold never reached")
                else:
                    if threshold is not None:
                        print(f"    Rounds={rounds}: [{lower_rt:.6f}, {upper_rt:.6f}] (threshold={threshold:.3f})")
                    else:
                        print(f"    Rounds={rounds}: [{lower_rt:.6f}, {upper_rt:.6f}]")


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

