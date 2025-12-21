import matplotlib
matplotlib.use("Agg")

import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation, FFMpegWriter
import networkx as nx
import time

from bosonic_search import bosonic_search
from fermionic_search import fermionic_search
from graph import Graph

graph = Graph(graph_type='cycle', N=3)
graph.calculate_hopping_rate()

simulation = fermionic_search(
    M=2,
    graph=graph,
    output='states',
    T=50,
    number_of_time_steps = 200,
    simulation_time_adjustment=False
)
simulation.determine_lowest_running_times(thresholds=[0.2, 0.4])
print("Lowest running times:", simulation.lowest_running_times)
print("Rounds of lowest running times:", simulation.rounds_of_lowest_running_times)
print("Simulation time: ", simulation.simulation_calculation_time)
print("Hopping rate calculation time: ", simulation.hopping_rate_calculation_time)
print("Runtime calculation time: ", simulation.running_time_calculation_time)
print("T: ", simulation.params['T'])
