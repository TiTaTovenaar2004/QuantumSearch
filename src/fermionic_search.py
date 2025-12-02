import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation
import networkx as nx

from majority_vote_operator import majority_vote_operator
from simulation import Simulation
from utils import critical_hopping_rate

def fermionic_search(
    M, # Number of fermions
    graph, # Graph object
    output, # 'states' or 'occupations'
    hopping_rate = None, # Hopping rate of the model (if None, set to critical hopping rate for complete graph)
    marked_vertex = 0, # Vertex to be marked
    T = 200, # Total time for the simulation
    number_of_time_steps = 200, # Number of time steps in the simulation
):
    N = graph.number_of_nodes() # Number of sites in the graph
    dim_per_site = 2 # Dimension of the Hilbert space per site
    if hopping_rate is None:
        hopping_rate = critical_hopping_rate(graph)

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

    # --- Construct the Hamiltonian ---
    adjacency_matrix = nx.to_numpy_array(graph)
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