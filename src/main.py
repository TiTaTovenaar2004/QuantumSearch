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

graph = Graph(graph_type='complete', N=3)
graph.calculate_eig()

simulation = fermionic_search(
    M=2,
    graph=graph,
    output='occupations',
    T=20
)
simulation.plot_site_occupations()
