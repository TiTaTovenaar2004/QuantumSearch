import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation
import networkx as nx

# --- Choose parameter values ---
output = 'occupations' # 'state' or 'occupations'
N = 6 # Number of sites in the graph
marked_vertex = 0
M = 2 # Number of fermions
dim_per_site = 2 # Dimension of the Hilbert space per site
hopping_rate = 1 / N # Critical hopping rate for complete graph
T = 80 # Total time for the simulation
number_of_time_steps = 160 # Number of time steps in the simulation
graph = 'cycle' # 'complete', 'cycle', 'line', 'erdos_renyi', 'barabasi_albert'
p = 0.5 # Parameter for Erdős-Rényi graph
m = 2 # Parameter for Barabási-Albert graph

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

# --- Plot site populations ---
def plot_site_populations(result):

    plt.figure(figsize=(8, 5))
    plt.tight_layout()

    # Plot the color map of site populations
    plt.imshow(result.expect, aspect='auto', cmap='jet', interpolation='nearest')

    # Labels
    plt.xlabel(r'Time, $\gamma t$', fontsize=12)
    plt.ylabel(r'Vertex, $i$', fontsize=12)

    # Colorbar
    plt.colorbar(label='Population')

    # Add yticks on both sides
    plt.tick_params(axis='y', which='both', right=True, labelright=True)

    # --- Rescale x-axis ticks ---
    # Get current x-ticks (these correspond to index positions, not actual times)
    ax = plt.gca()
    num_steps = number_of_time_steps  # from your global variable
    scale_factor = (num_steps / T) * N

    # Compute new tick positions corresponding to multiples of pi
    # Let’s cover the full time range
    max_pi = int(np.floor(T * math.pi / (T / N)))  # a rough upper bound
    max_tick = number_of_time_steps
    pi_ticks = np.arange(0, max_tick, int(scale_factor * math.pi))

    # Simpler, consistent with your request: 0, pi, 2pi, 3pi...
    xtick_labels = [rf'{i}$\pi$' if i > 0 else '0' for i in range(len(pi_ticks))]
    plt.xticks(pi_ticks, xtick_labels)

    plt.title('Site Populations over Time', fontsize=14)
    plt.show()


plot_site_populations(result)
