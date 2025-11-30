import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation
import networkx as nx

from majority_vote_operator import majority_vote_operator
from simulation import Simulation

def fermionic_search(
    N, # Number of sites in the graph
    M, # Number of fermions
    graph_type, # 'complete', 'cycle', 'line', 'erdos_renyi', 'barabasi_albert'
    output, # 'states', 'occupations' or 'success probabilities'
    marked_vertex = 0, # Vertex to be marked
    p = 0.5, # Parameter for Erdős-Rényi graph
    m = 2, # Parameter for Barabási-Albert graph
    hopping_rate = None, # Hopping rate of the model (if None, set to critical hopping rate for complete graph
    T = 200, # Total time for the simulation
    number_of_time_steps = 200, # Number of time steps in the simulation
    R = [1] # List of number of rounds of the majority vote for which to calculate the success probabilities (in ascending order!)
):
    dim_per_site = 2 # Dimension of the Hilbert space per site
    if hopping_rate is None:
        hopping_rate = 1 / N # Critical hopping rate for complete graph   

    # --- Create dictionary to hold parameters ---
    params = {
        'N' : N,
        'M' : M,
        'graph type' : graph_type,
        'output' : output,
        'marked vertex' : marked_vertex,
        'p' : p,
        'm' : m,
        'dim per site' : dim_per_site,
        'hopping rate' : hopping_rate,
        'T' : T,
        'number of time steps' : number_of_time_steps,
        'R' : R
    }

    # --- Define creation and annihilation operators ---
    def annihilation_operator(site, N): # N is the number of sites

        return fdestroy(N, site)

    def creation_operator(site, N):

        return fcreate(N, site)

    def number_operator(site, N):

        return fcreate(N, site) * fdestroy(N, site)

    def uniform_creation_operator(N): # Creates fermion in uniform superposition over all sites
        ops = 0
        for i in range(N):
            ops += fcreate(N, i)

        return ops / np.sqrt(N)

    # --- Define initial state ---
    # Vacuum state
    vacuum = tensor([basis(2, 0) for _ in range(N)])

    # Build M orthonormal orbitals
    orbitals = []
    for k in range(M):
        phi = np.exp(2j * np.pi * k * np.arange(N) / N)
        phi = phi / np.linalg.norm(phi)
        orbitals.append(phi)

    # Turn each orbital into a creation operator
    creators = []
    for phi in orbitals:
        op = 0
        for j in range(N):
            op = op + phi[j] * fcreate(N, j)
        creators.append(op)

    # Multiply creation operators onto vacuum
    init_state = vacuum
    for op in creators:
        init_state = op * init_state

    # --- Construct the graph ---
    if graph_type == 'complete':
        graph = nx.complete_graph(N)
    elif graph_type == 'cycle':
        graph = nx.cycle_graph(N)
    elif graph_type == 'line':
        graph = nx.path_graph(N)
    elif graph_type == 'erdos_renyi':
        graph = nx.erdos_renyi_graph(N, p)
        while not nx.is_connected(graph): # Ensure the graph is connected
            graph = nx.erdos_renyi_graph(N, p)
    elif graph_type == 'barabasi_albert':
        graph = nx.barabasi_albert_graph(N, m)
        while not nx.is_connected(graph): # Ensure the graph is connected
            graph = nx.barabasi_albert_graph(N, m)        
    else:
        raise ValueError("Graph must be 'complete', 'cycle', 'line', 'erdos_renyi' or 'barabasi_albert'")

    adjacency_matrix = nx.to_numpy_array(graph)

    # --- Construct the Hamiltonian ---
    H = 0

    for j in range(N):
        for l in range(N):
            H += -hopping_rate * adjacency_matrix[j, l] * creation_operator(j, N) * annihilation_operator(l, N)

    H += -creation_operator(marked_vertex, N) * annihilation_operator(marked_vertex, N)

    # --- Time evolution ---
    times = np.linspace(0, T, number_of_time_steps)

    if output == 'states':
        result = sesolve(H, init_state, times)
    elif output == 'occupations':
        number_operators = [number_operator(i, N) for i in range(N)]
        result = sesolve(H, init_state, times, e_ops = number_operators)
    elif output == 'success probabilities':
        result = sesolve(H, init_state, times)
        states = result.states
        success_probabilities = []

        # Tensoring each state R[0] times with itself, so that we can apply the R[0] rounds majority vote operator
        total_states = [tensor([state for _ in range(R[0])]) for state in states]

        for idx, r in enumerate(R):
            op = majority_vote_operator(N, M, r, marked_vertex, dim_per_site)
            probs = [expect(op, total_state) for total_state in total_states]
            success_probabilities.append(probs)

            # Tensoring each state R[idx + 1] - r times with itself, so that we can apply the R[idx + 1] rounds majority vote operator
            if idx < len(R) - 1:
                total_states = [tensor([total_state] + [state for _ in range(R[idx + 1] - r)]) for total_state, state in zip(total_states, states)]

        result = np.array(success_probabilities)
    else:
        raise ValueError("Output must be 'states', 'occupations' or 'success probabilities'")

    return Simulation(result, times, graph, params)