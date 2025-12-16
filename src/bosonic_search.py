import numpy as np
import math
from qutip import *

from simulation import Simulation

def bosonic_search(
    M, # Number of bosons
    graph, # Graph object
    output, # 'states' or 'occupations'
    hopping_rate = None, # Hopping rate of the model (if None, set to critical hopping rate)
    T = 200, # Total time for the simulation
    number_of_time_steps = 200, # Number of time steps in the simulation
):
    N = graph.N # Number of sites in the graph
    marked_vertex = graph.marked_vertex
    dim_per_site = M + 1 # Dimension of the Hilbert space per site
    if hopping_rate is None:
        if graph.eigenvalues is None:
            raise ValueError("The graph's eigenvalues have not been calculated yet. Please run the 'calculate_eig'-method of the Graph class before using the bosonic_search function.")
        hopping_rate = graph.hopping_rate

    # --- Create dictionary to hold parameters ---
    params = {
        'N' : N,
        'M' : M,
        'graph' : graph,
        'output' : output,
        'marked vertex' : marked_vertex,
        'dim per site' : dim_per_site,
        'hopping rate' : hopping_rate,
        'T' : T,
        'number of time steps' : number_of_time_steps,
    }

    # --- Define creation and annihilation operators ---
    def creation_operator(site, N): # N is the number of sites
        ops = []
        for i in range(N):
            if i != site:
                ops.append(qeye(dim_per_site))
            else:
                ops.append(create(dim_per_site))

        return tensor(ops)

    def annihilation_operator(site, N): # N is the number of sites
        ops = []
        for i in range(N):
            if i != site:
                ops.append(qeye(dim_per_site))
            else:
                ops.append(destroy(dim_per_site))

        return tensor(ops)

    def number_operator(site, N): # Counts the number of bosons at a given site
        ops = []
        for i in range(N):
            if i != site:
                ops.append(qeye(dim_per_site))
            else:
                ops.append(num(dim_per_site))

        return tensor(ops)

    def uniform_creation_operator(N): # Creates boson in uniform superposition over all sites
        ops = 0
        for i in range(N):
            ops += creation_operator(i, N)

        return ops / np.sqrt(N)

    # --- Define initial state ---
    vacuum = tensor([basis(dim_per_site, 0) for i in range(N)])
    init_state = ((uniform_creation_operator(N) ** M) / np.sqrt(math.factorial(M))) * vacuum

    # --- Construct the Hamiltonian ---
    adjacency_matrix = graph.adjacency
    H = 0

    for j in range(N):
        for l in range(N):
            H += -hopping_rate * adjacency_matrix[j, l] * creation_operator(j, N) * annihilation_operator(l, N)

    H += -creation_operator(marked_vertex, N) * annihilation_operator(marked_vertex, N)

    # --- Time evolution ---
    times = np.linspace(0, T, number_of_time_steps)

    if output == 'states': # Only calculate states
        result = sesolve(H, init_state, times)
        states = result.states
        occupations = None
    elif output == 'occupations': # Only calculate occupations
        number_operators = [number_operator(i, N) for i in range(N)]
        result = sesolve(H, init_state, times, e_ops = number_operators)
        states = None
        occupations = result.expect
    else:
        raise ValueError("The 'output'-parameter must be either 'states' or 'occupations'.")

    return Simulation(states, occupations, times, graph, params)
