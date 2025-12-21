import matplotlib
matplotlib.use("Agg")

import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation, FFMpegWriter
import networkx as nx
import time

from quantumsearch.core.bosonic_search import bosonic_search
from quantumsearch.core.fermionic_search import fermionic_search
from quantumsearch.core.graph import Graph
from quantumsearch.plotting import plot_site_occupations

graph = Graph(graph_type='cycle', N=3)
graph.calculate_hopping_rate()

simulation = fermionic_search(
    M=2,
    graph=graph,
    output='occupations',
    T=50,
    number_of_time_steps = 200,
    simulation_time_adjustment=False
)

plot_site_occupations(simulation)
