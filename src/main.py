import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation
import networkx as nx

from bosonic_search import bosonic_search
from fermionic_search import fermionic_search
from majority_vote_operator import majority_vote_operator
from plotting import plot_site_populations, plot_marked_vertex_occupation_distribution, animate_marked_vertex_distribution, plot_success_probabilities
from utils import show_superposition

result1, times, G, params = fermionic_search(N=5, M=2, output='success probabilities', R = [3], graph='barabasi_albert', m=2, T=30)
result2, times, G, params = bosonic_search(N=5, M=2, output='success probabilities', R = [3], graph='barabasi_albert', m=2, T=30)

# show_superposition(result.states[-1])

plot_success_probabilities(result1, times, R = [3])  
plot_success_probabilities(result2, times, R = [3])

# plot_marked_vertex_occupation_distribution(result.states[-1], params)

# animate_marked_vertex_distribution(result.states, times, params)
