import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation
import networkx as nx

from bosonic_search import bosonic_search
from fermionic_search import fermionic_search
from plotting import plot_site_populations, plot_marked_vertex_occupation_distribution, animate_marked_vertex_distribution

result, times, G, params = bosonic_search(N=4, M=4, output='states', graph='erdos_renyi', p=0.7)

# plot_marked_vertex_occupation_distribution(result.states[-1], params)

# animate_marked_vertex_distribution(result.states, times, params)
