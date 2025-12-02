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

simulation = fermionic_search(
    M = 2,
    graph = create_graph(4, 'complete'),
    output = 'states',
    T = 40,
)

simulation.determine_lowest_running_times([0.2, 0.4, 0.6], stop_condition=1)
print(simulation.lowest_running_times)
print(simulation.rounds_of_lowest_running_times)
simulation.calculate_success_probabilities([1, 2, 3, 4])
simulation.determine_running_times([0.2, 0.4, 0.6])
print(simulation.running_times)