import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation
import networkx as nx
import time

from bosonic_search import bosonic_search
from fermionic_search import fermionic_search
from utils import create_graph

simulation = bosonic_search(
    M = 2,
    graph = create_graph(4, 'complete'),
    output = 'states',
    T = 40,
)

# simulation.plot_marked_vertex_occupation_distribution()
# simulation.animate_marked_vertex_distribution()
simulation.calculate_success_probabilities([1, 2])
simulation.determine_running_times([0.5, 0.75, 0.9])
print(simulation.running_times)
simulation.plot_success_probabilities()

