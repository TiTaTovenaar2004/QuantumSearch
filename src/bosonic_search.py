import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation
import networkx as nx

from majority_vote_operator import majority_vote_operator
from simulation import Simulation
from utils import critical_hopping_rate

def bosonic_search(
    M, # Number of bosons
    graph, # Graph object
    hopping_rate = None, # Hopping rate of the model (if None, set to critical hopping rate for complete graph
    calculate_occupations = False, # Whether to calculate each (average) site occupation (as a function of time) as well
    marked_vertex = 0, # Vertex to be marked
    T = 200, # Total time for the simulation
    number_of_time_steps = 200, # Number of time steps in the simulation
):
    N = graph.number_of_nodes() # Number of sites in the graph
    dim_per_site = M + 1 # Dimension of the Hilbert space per site
    if hopping_rate is None:
        hopping_rate = critical_hopping_rate(graph)

    # --- Create dictionary to hold parameters ---
    params = {
        'N' : N,
        'M' : M,
        'graph' : graph,
        'calculate occupations' : calculate_occupations,
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
    adjacency_matrix = nx.to_numpy_array(graph)
    H = 0

    for j in range(N):
        for l in range(N):
            H += -hopping_rate * adjacency_matrix[j, l] * creation_operator(j, N) * annihilation_operator(l, N)

    H += -creation_operator(marked_vertex, N) * annihilation_operator(marked_vertex, N)

    # --- Time evolution ---
    times = np.linspace(0, T, number_of_time_steps)

    if not calculate_occupations: # Only calculate states
        result = sesolve(H, init_state, times)
        states = result.states
        occupations = None
    else: # Also calculate occupations
        number_operators = [number_operator(i, N) for i in range(N)]
        result = sesolve(H, init_state, times, e_ops = number_operators)
        states = result.states
        occupations = result.expect

    return Simulation(states, occupations, times, graph, params)
