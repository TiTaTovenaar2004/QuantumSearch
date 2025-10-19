import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation
import networkx as nx

def fermionic_search(
    N, # Number of sites in the graph
    M, # Number of fermions
    output = 'occupations', # 'state' or 'occupations'
    T = 200, # Total time for the simulation
    number_of_time_steps = 200, # Number of time steps in the simulation
    graph = 'complete', # 'complete', 'cycle', 'line', 'erdos_renyi', 'barabasi_albert'
    marked_vertex = 0, # Vertex to be marked
    p = 0.5, # Parameter for Erdős-Rényi graph
    m = 2 # Parameter for Barabási-Albert graph
):
    dim_per_site = 2 # Dimension of the Hilbert space per site
    hopping_rate = 1 / N # Critical hopping rate for complete graph   

    # --- Create dictionary to hold parameters ---
    params = {}
    params['output'] = output
    params['N'] = N
    params['M'] = M
    params['T'] = T
    params['number_of_time_steps'] = number_of_time_steps
    params['graph'] = graph
    params['marked_vertex'] = marked_vertex
    params['p'] = p
    params['m'] = m
    params['dim_per_site'] = dim_per_site
    params['hopping_rate'] = hopping_rate

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
    if graph == 'complete':
        G = nx.complete_graph(N)
    elif graph == 'cycle':
        G = nx.cycle_graph(N)
    elif graph == 'line':
        G = nx.path_graph(N)
    elif graph == 'erdos_renyi':
        G = nx.erdos_renyi_graph(N, p)
        while not nx.is_connected(G): # Ensure the graph is connected
            G = nx.erdos_renyi_graph(N, p)
    elif graph == 'barabasi_albert':
        G = nx.barabasi_albert_graph(N, m)
        while not nx.is_connected(G): # Ensure the graph is connected
            G = nx.barabasi_albert_graph(N, m)        
    else:
        raise ValueError("Graph must be 'complete', 'cycle', 'line', 'erdos_renyi' or 'barabasi_albert'")

    adjacency_matrix = nx.to_numpy_array(G)

    # --- Construct the Hamiltonian ---
    H = 0

    for j in range(N):
        for l in range(N):
            H += -hopping_rate * adjacency_matrix[j, l] * creation_operator(j, N) * annihilation_operator(l, N)

    H += -creation_operator(marked_vertex, N) * annihilation_operator(marked_vertex, N)

    # --- Time evolution ---
    times = np.linspace(0, T, number_of_time_steps)

    if output == 'state':
        result = sesolve(H, init_state, times)
    elif output == 'occupations':
        number_operators = [number_operator(i, N) for i in range(N)]
        result = sesolve(H, init_state, times, e_ops = number_operators)
    else:
        raise ValueError("Output must be 'state' or 'occupations'")

    return result, times, G, params