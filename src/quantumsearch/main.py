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
from quantumsearch.plotting import plot_site_occupations, plot_marked_vertex_occupation_distribution, animate_marked_vertex_distribution, plot_success_probabilities

graph = Graph(graph_type='line', N=3)
graph.calculate_hopping_rate()

simulation = fermionic_search(
    M=2,
    graph=graph,
    output='states',
    T=50,
    number_of_time_steps = 5000,
    simulation_time_adjustment=True
)

simulation.determine_lowest_running_times(thresholds=[0.5])
print("T: ", simulation.params['T'])
print("Lowest running times: ", simulation.lowest_running_times)
print("Rounds of lowest running times: ", simulation.rounds_of_lowest_running_times)
print("total calculation time: ", simulation.simulation_calculation_time + simulation.hopping_rate_calculation_time + simulation.running_time_calculation_time)
