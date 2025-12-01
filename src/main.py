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
    graph = create_graph(4, 'erdos-renyi', p=0.8),
    hopping_rate = 1/4,
    calculate_occupations=True,
    T = 100,
)

simulation.plot_site_occupations()