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


def filter_results(results, graph_type=None, N=None, M=None, search_type=None, hopping_rate=None, fast_mode=None):
    """
    Filter results based on specified criteria.

    Parameters:
    -----------
    results : list
        List of result dictionaries
    graph_type : str, optional
        Filter by graph type (e.g., 'complete', 'cycle', 'line')
    N : int, optional
        Filter by number of vertices
    M : int, optional
        Filter by number of particles
    search_type : str, optional
        Filter by search type ('bosonic' or 'fermionic')
    hopping_rate : None or float, optional
        Filter by hopping rate. Use None to find results where hopping_rate was None in task_config
    fast_mode : bool, optional
        Filter by fast_mode (True or False)

    Returns:
    --------
    filtered_results : list
        List of results matching all specified criteria
    """
    filtered = results

    if graph_type is not None:
        filtered = [r for r in filtered if r['graph_type'] == graph_type]

    if N is not None:
        filtered = [r for r in filtered if r['N'] == N]

    if M is not None:
        filtered = [r for r in filtered if r['M'] == M]

    if search_type is not None:
        filtered = [r for r in filtered if r['search_type'] == search_type]

    if hopping_rate is not None or hopping_rate == 'None':
        # Filter by hopping_rate from task_config
        if hopping_rate == 'None' or hopping_rate is None:
            # Find results where hopping_rate was None in config
            filtered = [r for r in filtered
                       if 'task_config' in r
                       and r['task_config']['simulation_config'].get('hopping_rate', None) is None]
        else:
            # Find results with specific hopping_rate value
            filtered = [r for r in filtered
                       if 'task_config' in r
                       and r['task_config']['simulation_config'].get('hopping_rate', None) == hopping_rate]

    if fast_mode is not None:
        # Filter by fast_mode from task_config
        filtered = [r for r in filtered
                   if 'task_config' in r
                   and r['task_config']['estimation_config'].get('fast_mode', False) == fast_mode]

    return filtered


def main(timestamp=None, graph_type=None, N=None, M=None, search_type=None, hopping_rate=None, fast_mode=None):
    """
    Main function to load and analyze results.

    Parameters:
    -----------
    timestamp : str, optional
        Specific timestamp to load (format: YYYYMMDD_HHMMSS). If None, loads most recent.
    graph_type : str, optional
        Filter by graph type (e.g., 'complete', 'cycle', 'line')
    N : int, optional
        Filter by number of vertices
    M : int, optional
        Filter by number of particles
    search_type : str, optional
        Filter by search type ('bosonic' or 'fermionic')
    hopping_rate : None or float or 'None', optional
        Filter by hopping rate. Use None or 'None' to find results where hopping_rate was None
    fast_mode : bool, optional
        Filter by fast_mode (True or False)
    """

    # Set data directory
    data_dir = 'results/data'

    # Load results using the load_results function from mpi_runner
    print("Loading simulation results...")
    results, summary = load_results(input_dir=data_dir, timestamp=timestamp)
    print(f"Successfully loaded {len(results)} simulation results")

    # Apply filters if any are specified
    filter_applied = any([graph_type is not None, N is not None, M is not None,
                          search_type is not None, hopping_rate is not None, fast_mode is not None])

    if filter_applied:
        print("\nApplying filters:")
        if graph_type is not None:
            print(f"  - graph_type = '{graph_type}'")
        if N is not None:
            print(f"  - N = {N}")
        if M is not None:
            print(f"  - M = {M}")
        if search_type is not None:
            print(f"  - search_type = '{search_type}'")
        if hopping_rate is not None or hopping_rate == 'None':
            print(f"  - hopping_rate = None (from config)")
        if fast_mode is not None:
            print(f"  - fast_mode = {fast_mode}")

        results = filter_results(results, graph_type=graph_type, N=N, M=M,
                                search_type=search_type, hopping_rate=hopping_rate, fast_mode=fast_mode)
        print(f"\nFiltered to {len(results)} results\n")

    # Display summary
    display_summary(results, summary)

    # Generate plots
    if len(results) > 0:
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
    # Example usage:
    # To load most recent data:
    #   results, summary = main()
    #
    # To filter by specific attributes:
    #   results, summary = main(graph_type='cycle')
    #   results, summary = main(N=8, M=4)
    #   results, summary = main(search_type='fermionic', hopping_rate=None)
    #   results, summary = main(fast_mode=True)  # Only fast mode results
    #   results, summary = main(fast_mode=False)  # Only slow mode results
    #   results, summary = main(N=8, fast_mode=True)  # Combined filters
    #   results, summary = main(timestamp='20260103_145228')

    results, summary = main(graph_type='erdos-renyi')

