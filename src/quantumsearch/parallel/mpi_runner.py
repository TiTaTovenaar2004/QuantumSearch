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
import networkx as nx
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
            - 'number_of_rounds': list of int
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
        est_config = task_config['estimation_config']
        estimation_result = simulation.simulate_estimate(
            times=times,
            number_of_rounds=est_config['number_of_rounds'],
            threshold=est_config['threshold'],
            precision=est_config['precision'],
            confidence=est_config['confidence'],
            fast_mode = est_config['fast_mode']
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
            'times': times,
            'simulation_time': simulation.simulation_time,
            'estimation_time': simulation.estimation_time,
            'estimated_success_probabilities': estimation_result['success_probabilities'],
            'lower_running_times': estimation_result['lower_running_times'],
            'upper_running_times': estimation_result['upper_running_times'],
            'graph': graph.graph,  # Store NetworkX graph object
            'adjacency_matrix': nx.to_numpy_array(graph.graph),  # Store adjacency matrix for reconstruction
            'number_of_edges': graph.graph.number_of_edges(),  # Store edge count for plotting
            'status': 'completed'
        }

        my_results.append(result)

        # Explicitly delete large objects to free memory
        del simulation.states  # Delete stored quantum states (largest memory consumer)
        del simulation
        del graph
        import gc
        gc.collect()  # Force garbage collection

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
            'adjacency_matrix': result.get('adjacency_matrix'),  # Save adjacency matrix instead of graph
            'number_of_edges': result.get('number_of_edges'),
        }

        # Save task_config as JSON string if present
        if 'task_config' in result:
            # Convert numpy arrays to lists for JSON serialization
            task_config_serializable = {}
            for key, value in result['task_config'].items():
                if isinstance(value, dict):
                    task_config_serializable[key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in value.items()
                    }
                elif isinstance(value, np.ndarray):
                    task_config_serializable[key] = value.tolist()
                else:
                    task_config_serializable[key] = value
            save_data['task_config_json'] = json.dumps(task_config_serializable)

        # Add running times and success probabilities as arrays
        if 'lower_running_times' in result:
            save_data['lower_running_times'] = result['lower_running_times']
        if 'upper_running_times' in result:
            save_data['upper_running_times'] = result['upper_running_times']
        if 'estimated_success_probabilities' in result:
            save_data['estimated_success_probabilities'] = result['estimated_success_probabilities']

        # Save to .npz file (allow_pickle for NetworkX graph object)
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
                'p': r.get('task_config', {}).get('graph_config', {}).get('p') if 'task_config' in r else None,
                'simulation_time': float(r['simulation_time']),
                'estimation_time': float(r['estimation_time']),
                'num_time_points': len(r['times']),
                'lower_running_times': r['lower_running_times'].tolist() if 'lower_running_times' in r else [],
                'upper_running_times': r['upper_running_times'].tolist() if 'upper_running_times' in r else []
            }
            for i, r in enumerate(results)
        ]
    }

    with open(summary_filepath, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"  - {len(results)} individual result files (.npz)")
    print(f"  - 1 summary file: summary_{timestamp}.json")


def load_results(input_dir='results/data', timestamp=None):
    """
    Load previously saved simulation results.

    Parameters:
    -----------
    input_dir : str
        Directory containing saved results
    timestamp : str, optional
        Specific timestamp to load (format: YYYYMMDD_HHMMSS).
        If None, loads the most recent run.

    Returns:
    --------
    results : list
        List of result dictionaries, each containing:
        - task_id, graph_type, N, search_type, M, hopping_rate
        - times, probabilities (from estimation)
        - simulation_time, estimation_time
    summary : dict
        Summary information about the run
    """

    import os
    import json

    # Find the run to load
    if timestamp is None:
        # Find most recent run based on timestamp
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Data directory not found: {input_dir}")

        # Find all summary files
        summary_files = [f for f in os.listdir(input_dir)
                        if f.startswith('summary_') and f.endswith('.json')]

        if not summary_files:
            raise FileNotFoundError(f"No summary files found in {input_dir}")

        # Sort by timestamp (embedded in filename) - most recent first
        summary_files.sort(reverse=True)
        latest_summary = summary_files[0]

        # Extract timestamp
        timestamp = latest_summary.replace('summary_', '').replace('.json', '')
        summary_path = os.path.join(input_dir, latest_summary)
    else:
        summary_path = os.path.join(input_dir, f'summary_{timestamp}.json')

    # Load summary
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    # Load individual result files
    results = []

    for task_info in summary['tasks']:
        task_id = task_info['task_id']
        graph_type = task_info['graph_type']
        N = task_info['N']
        search_type = task_info['search_type']
        M = task_info['M']
        p = task_info.get('p')  # Get p parameter from summary if available

        # Construct filename
        filename = f"task_{task_id}_{graph_type}_N{N}_{search_type}_M{M}_{timestamp}.npz"
        filepath = os.path.join(input_dir, filename)

        # Load data
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filename}")
            continue

        data = np.load(filepath, allow_pickle=True)

        # Extract data
        result = {
            'task_id': task_id,
            'graph_type': str(data['graph_type']),
            'N': int(data['N']),
            'search_type': str(data['search_type']),
            'M': int(data['M']),
            'hopping_rate': float(data['hopping_rate']),
            'times': data['times'],
            'simulation_time': float(data['simulation_time']),
            'estimation_time': float(data['estimation_time'])
        }

        # Load number of edges if present
        if 'number_of_edges' in data.keys():
            result['number_of_edges'] = int(data['number_of_edges'])

        # Reconstruct NetworkX graph from adjacency matrix
        if 'adjacency_matrix' in data.keys():
            adjacency_matrix = data['adjacency_matrix']
            result['graph'] = nx.from_numpy_array(adjacency_matrix)
        # Fallback: try to load graph object if present (for old data files)
        elif 'graph' in data.keys():
            # Handle different storage formats for graph object
            graph_data = data['graph']
            # Numpy stores pickled objects as 0-dimensional arrays
            if isinstance(graph_data, np.ndarray):
                # Use [()] indexing to extract object from 0-d array, or .item() for 1-element array
                if graph_data.ndim == 0:
                    result['graph'] = graph_data.item()  # or graph_data[()]
                elif graph_data.size == 1:
                    result['graph'] = graph_data.item()
                else:
                    result['graph'] = graph_data
            else:
                result['graph'] = graph_data  # Already in the correct format

        # Load task_config if present in npz file
        if 'task_config_json' in data.keys():
            result['task_config'] = json.loads(str(data['task_config_json']))
        # Otherwise reconstruct minimal task_config from available data
        elif p is not None:
            result['task_config'] = {
                'graph_config': {
                    'graph_type': graph_type,
                    'N': N,
                    'p': p
                }
            }

        # Load running times and success probabilities as arrays
        if 'lower_running_times' in data.keys():
            result['lower_running_times'] = data['lower_running_times']
        if 'upper_running_times' in data.keys():
            result['upper_running_times'] = data['upper_running_times']
        if 'estimated_success_probabilities' in data.keys():
            result['estimated_success_probabilities'] = data['estimated_success_probabilities']

        results.append(result)

    return results, summary
