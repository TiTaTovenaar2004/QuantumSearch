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

simulation = fermionic_search(
    N = 3,
    M = 1,
    graph_type = 'complete',
    output = 'success probabilities',
)

plot_success_probabilities(simulation.result, simulation.times, R=[1])