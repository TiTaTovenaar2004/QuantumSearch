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
    N = 3,
    M = 2,
    graph = create_graph(3, 'complete'),
    hopping_rate = 1/3,
    output = 'occupations'
)

simulation.plot_site_populations()