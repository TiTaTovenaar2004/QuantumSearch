import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation
import networkx as nx
import time

from bosonic_search import bosonic_search
from fermionic_search import fermionic_search
from graph import Graph

import numpy as np
import matplotlib.pyplot as plt

graph = Graph(graph_type='complete', N=3)
graph.calculate_eig()

simulation = bosonic_search(
    M=2,
    graph=graph,
    output='states',
    T=40
)
simulation.calculate_success_probabilities(rounds=[1, 2, 3])
simulation.plot_success_probabilities()