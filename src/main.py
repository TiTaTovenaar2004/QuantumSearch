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
    M = 3,
    graph = create_graph(4, 'erdos-renyi', p=0.8),
    calculate_occupations = False,
    T = 110,
)

simulation.animate_marked_vertex_distribution()