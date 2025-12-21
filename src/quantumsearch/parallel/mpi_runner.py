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
        task_index = rank + i * size
        print(f"Rank {rank}: Running simulation {task_index+1}/{len(task_list)}")

        # Create graph
        graph = Graph(**task['graph_config'])
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

        results.append({
            'task': task,
            'graph': graph,
            'simulation': simulation,
            'rank': rank,
            'task_idx': task_index
        })

    # Gather all results to rank 0
    all_results = comm.gather(results, root=0)

    if rank == 0:
        # Flatten list of lists
        all_results = [item for sublist in all_results for item in sublist]
        return all_results

    return results


# Optional: Additional utility functions for parameter sweeps can be added here
