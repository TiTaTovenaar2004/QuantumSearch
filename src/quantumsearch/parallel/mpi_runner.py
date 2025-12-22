"""MPI wrapper for parallel quantum search execution."""

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    MPI = None

import numpy as np
from quantumsearch.core.bosonic_search import bosonic_search
from quantumsearch.core.fermionic_search import fermionic_search
from quantumsearch.core.graph import Graph


def run_parallel_search(task_list, output_dir='results/data'):
    """
    Run quantum search in parallel across different parameter sets.

    Parameters:
    -----------
    task_list : list of dict
        Each dict contains parameters for one simulation
        --> task = {
            'graph_config': {
              'graph_type': str, # 'complete', 'cycle', 'line', 'erdos-renyi', 'barabasi-albert'
              'N': int,
              'p': float,
              'm': int,
              'marked_vertex': int
            },
            'time_config': {
              'T': float,
              'number_of_time_steps': int,
              'simulation_time_adjustment': bool
            },
            'simulation_config': {
              'search_type': str, # 'bosonic' or 'fermionic'
              'M': int,
              'hopping_rate': float,
              'output': str # 'occupations' or 'states'
            },
            'task': {
                'determine_lowest_running_times': bool,
                'thresholds': list of float,
                'stop_condition': int
            }
          }
    output_dir : str
        Directory to save results

    Returns:
    --------
    results : list
        Results from this process's assigned tasks
    """
    if not MPI_AVAILABLE:
        raise ImportError("mpi4py is not installed. Install with: pip install mpi4py")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Distribute work: each rank gets subset of task_list
    my_tasks = task_list[rank::size]  # Distribute round-robin

    results = []
    for i, task in enumerate(my_tasks):
        # --- Task setup ---
        task_index = rank + i * size
        print(f"Rank {rank}: Running simulation {task_index+1}/{len(task_list)}")

        graph = Graph(**task['graph_config'])

        # --- Determine lowest running times if True ---
        if 'determine_lowest_running_times' in task['task'] and task['task']['determine_lowest_running_times']:
            # Validate required parameters
            if 'thresholds' not in task['task'] or 'stop_condition' not in task['task']:
                raise ValueError("When 'determine_lowest_running_times' is True, 'thresholds' and 'stop_condition' must be provided in 'task'.")

            # Calculate hopping rate if not provided
            if 'hopping_rate' not in task['simulation_config'] or task['simulation_config']['hopping_rate'] is None:
                graph.calculate_hopping_rate()

            # Run search
            search_func = bosonic_search if task['simulation_config']['search_type'] == 'bosonic' else fermionic_search
            simulation = search_func(
                M=task['simulation_config']['M'],
                graph=graph,
                output=task['simulation_config']['output'],
                T=task['time_config']['T'],
                number_of_time_steps=task['time_config']['number_of_time_steps'],
                simulation_time_adjustment=task['time_config']['simulation_time_adjustment']
            )

            # Determine lowest running times
            simulation.determine_lowest_running_times(
                thresholds=task['task']['thresholds'],
                stop_condition=task['task']['stop_condition']
            )

            # Free memory by deleting large state arrays
            simulation.states = None
            simulation.occupations = None

            # Store results
            results.append({
                'task': task,
                'graph': graph,
                'simulation': simulation,
                'rank': rank,
                'task_idx': task_index
            })
        else:
            raise ValueError("No task specified.")

    # Gather all results to rank 0
    all_results = comm.gather(results, root=0)

    if rank == 0:
        # Flatten list of lists
        all_results = [item for sublist in all_results for item in sublist]
        return all_results

    return results


# Optional: Additional utility functions for parameter sweeps can be added here
