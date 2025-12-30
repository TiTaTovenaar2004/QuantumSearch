"""MPI runner for parallel quantum search simulations.

This module provides a broad framework for running quantum search simulations
in parallel using MPI (Message Passing Interface).
"""

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    MPI = None

import numpy as np
from quantumsearch.core.simulation import Simulation
from quantumsearch.core.graph import Graph


def run_parallel_simulations(task_configs):
    """
    Execute quantum search simulations in parallel across multiple MPI processes.

    This function distributes simulation tasks across available MPI processes,
    runs them in parallel, and gathers results back to the master process.

    Parameters:
    -----------
    task_configs : list of dict
        List of configuration dictionaries, each defining one simulation task.
        Each dict must contain:
        
        'graph_config': dict
            Parameters for Graph constructor:
            - 'graph_type': str ('complete', 'cycle', 'line', 'erdos-renyi', 'barabasi-albert')
            - 'N': int (number of vertices)
            - 'p': float (optional, for 'erdos-renyi')
            - 'm': int (optional, for 'barabasi-albert')
            - 'marked_vertex': int (optional)
        
        'simulation_config': dict
            Parameters for Simulation:
            - 'search_type': str ('bosonic' or 'fermionic')
            - 'M': int (number of particles)
            - 'hopping_rate': float (optional)
        
        'times': array-like
            Time points at which to evaluate the quantum state
        
        'estimation_config': dict (optional)
            Parameters for success probability estimation:
            - 'number_of_rounds': int
            - 'precision': float
            - 'confidence': float

    Returns:
    --------
    results : list or None
        List of result dictionaries from all simulations (only on rank 0).
        Each result contains: graph parameters, simulation parameters, times,
        estimated_success_probabilities, simulation_time, estimation_time, status.
        Returns None on non-master ranks.

    Structure:
    ----------
    1. Initialize MPI communicator and get rank/size
    2. Distribute tasks across processes (round-robin)
    3. Each process runs its assigned simulations
    4. Gather results back to master process (rank 0)
    5. Master process aggregates and returns results
    """

    if not MPI_AVAILABLE:
        raise ImportError("mpi4py is not installed. Cannot run parallel simulations.")

    # --- 1. MPI Setup ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Process ID
    size = comm.Get_size()  # Total number of processes

    # --- 2. Task Distribution ---
    # Distribute tasks to processes (round-robin: task i goes to rank i % size)
    my_tasks = [task for i, task in enumerate(task_configs) if i % size == rank]

    if rank == 0:
        print(f"Distributing {len(task_configs)} tasks across {size} processes")

    # --- 3. Execute Assigned Tasks ---
    my_results = []

    for task_idx, task_config in enumerate(my_tasks):
        print(f"Rank {rank}: Processing task {task_idx + 1}/{len(my_tasks)}")

        # - Create Graph from config
        graph_config = task_config['graph_config']
        graph = Graph(**graph_config)
        
        # - Create Simulation object
        sim_config = task_config['simulation_config']
        simulation = Simulation(
            search_type=sim_config['search_type'],
            M=sim_config['M'],
            graph=graph,
            hopping_rate=sim_config.get('hopping_rate', None)
        )
        
        # - Run simulation with simulate(times)
        times = task_config['times']
        simulation.simulate(times)
        
        # - Calculate/estimate success probabilities
        if 'estimation_config' in task_config:
            est_config = task_config['estimation_config']
            simulation.estimate_success_probabilities(
                number_of_rounds=est_config['number_of_rounds'],
                precision=est_config['precision'],
                confidence=est_config['confidence']
            )
        
        # - Extract and store relevant results
        result = {
            'rank': rank,
            'task_config': task_config,
            'graph_type': graph_config['graph_type'],
            'N': graph.N,
            'search_type': sim_config['search_type'],
            'M': sim_config['M'],
            'hopping_rate': simulation.hopping_rate,
            'times': simulation.times,
            'simulation_time': simulation.simulation_time,
            'estimation_time': simulation.estimation_time,
            'estimated_success_probabilities': simulation.estimated_success_probabilities,
            'status': 'completed'
        }

        my_results.append(result)

    # --- 4. Gather Results to Master ---
    all_results = comm.gather(my_results, root=0)

    # --- 5. Master Process Aggregates Results ---
    if rank == 0:
        # Flatten list of lists into single list
        results = [item for sublist in all_results for item in sublist]
        print(f"All {len(results)} simulations completed")
        return results
    else:
        return None


def save_results(results, output_dir='results/data'):
    """
    Save simulation results to disk.

    Parameters:
    -----------
    results : list
        List of result dictionaries from run_parallel_simulations
    output_dir : str
        Directory to save results

    Structure:
    ----------
    User should implement saving logic based on what data needs to be preserved:
    - Success probabilities
    - Running times
    - Quantum states (if needed)
    - Metadata (graph parameters, timing info, etc.)
    """

    import os
    import json
    from datetime import datetime
    
    # - Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # - Save individual simulation results
    for i, result in enumerate(results):
        # Create filename based on task parameters
        filename = f"task_{i}_{result['graph_type']}_N{result['N']}_{result['search_type']}_M{result['M']}_{timestamp}.npz"
        filepath = os.path.join(output_dir, filename)
        
        # Prepare data for saving (convert to serializable format)
        save_data = {
            'graph_type': result['graph_type'],
            'N': result['N'],
            'search_type': result['search_type'],
            'M': result['M'],
            'hopping_rate': result['hopping_rate'],
            'times': result['times'],
            'simulation_time': result['simulation_time'],
            'estimation_time': result['estimation_time'],
        }
        
        # Add estimated success probabilities
        if result['estimated_success_probabilities']:
            for j, est_result in enumerate(result['estimated_success_probabilities']):
                save_data[f'rounds_{j}'] = est_result['rounds']
                save_data[f'precision_{j}'] = est_result['precision']
                save_data[f'confidence_{j}'] = est_result['confidence']
                save_data[f'probabilities_{j}'] = est_result['probabilities']
        
        # Save to .npz file
        np.savez(filepath, **save_data)
    
    # - Save summary/aggregate results
    summary_filepath = os.path.join(output_dir, f'summary_{timestamp}.json')
    
    summary = {
        'timestamp': timestamp,
        'total_tasks': len(results),
        'total_simulation_time': sum(r['simulation_time'] for r in results),
        'total_estimation_time': sum(r['estimation_time'] for r in results),
        'tasks': [
            {
                'task_id': i,
                'graph_type': r['graph_type'],
                'N': r['N'],
                'search_type': r['search_type'],
                'M': r['M'],
                'hopping_rate': float(r['hopping_rate']),
                'simulation_time': float(r['simulation_time']),
                'estimation_time': float(r['estimation_time']),
                'num_time_points': len(r['times']),
                'estimated_success_probabilities': [
                    {
                        'rounds': int(est['rounds']),
                        'precision': float(est['precision']),
                        'confidence': float(est['confidence']),
                        'max_probability': float(np.max(est['probabilities']))
                    }
                    for est in r['estimated_success_probabilities']
                ] if r['estimated_success_probabilities'] else []
            }
            for i, r in enumerate(results)
        ]
    }
    
    with open(summary_filepath, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    print(f"  - {len(results)} individual result files (.npz)")
    print(f"  - 1 summary file: summary_{timestamp}.json")


def load_results(input_dir='results/data'):
    """
    Load previously saved simulation results.

    Parameters:
    -----------
    input_dir : str
        Directory containing saved results

    Returns:
    --------
    results : list
        List of result dictionaries

    Structure:
    ----------
    User should implement loading logic matching save_results format
    """

    # TODO: Implement result loading logic

    pass
