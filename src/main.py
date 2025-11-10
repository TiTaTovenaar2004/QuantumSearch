import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation
import networkx as nx

from bosonic_search import bosonic_search
from fermionic_search import fermionic_search
from majority_vote_operator import majority_vote_operator
from plotting import plot_site_populations, plot_marked_vertex_occupation_distribution, animate_marked_vertex_distribution

result, times, G, params = fermionic_search(N=5, M=1, output='occupations', graph='barabasi_albert', m=2)

plot_site_populations(result, params)

# plot_marked_vertex_occupation_distribution(result.states[-1], params)

# animate_marked_vertex_distribution(result.states, times, params)
