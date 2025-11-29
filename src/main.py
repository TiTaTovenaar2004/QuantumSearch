import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation
import networkx as nx
import time

from bosonic_search import bosonic_search
from fermionic_search import fermionic_search
from majority_vote_operator import majority_vote_operator
from plotting import plot_site_populations, plot_marked_vertex_occupation_distribution, animate_marked_vertex_distribution, plot_success_probabilities
from utils import show_superposition

result, times, G, params = bosonic_search(N = 4, M = 2, output = 'success probabilities', graph = 'complete', R = [1, 2, 3, 4], T = 20, hopping_rate = 0.25)
plot_success_probabilities(result, times, R = [1, 2, 3, 4])


