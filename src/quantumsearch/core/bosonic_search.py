import numpy as np
import math
from qutip import *
import time

from quantumsearch.core.simulation import Simulation
from quantumsearch.core.utils import number_of_extrema

def bosonic_search(
    M, # Number of bosons
    graph, # Graph object
    hopping_rate, # Hopping rate of the model
    times, # Array of time points for the simulation
):
    N = graph.N # Number of sites in the graph
    marked_vertex = graph.marked_vertex
    dim_per_site = M + 1 # Dimension of the Hilbert space per site

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
    result = sesolve(H, init_state, times)
    states = result.states

    return states