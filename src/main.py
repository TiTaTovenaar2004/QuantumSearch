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

graph1 = Graph('erdos-renyi', 200)
graph1.calculate_eig()
print("Eigenvalues:", graph1.eigenvalues)
print("Eigenvectors:", graph1.eigenvectors)
print("Spectral Radius:", graph1.spectral_radius)
print("Eigenvalues Normalized:", graph1.eigenvalues_normalized)
print("Spectral Gap:", graph1.spectral_gap)
print("Marked Vertex Projection:", graph1.marked_vertex_projection)
print("sqrt(epsilon):", graph1.sqrt_epsilon)
print("S1:", graph1.S1)
print("S2:", graph1.S2)
print("S3:", graph1.S3)
print("Hopping Rate:", graph1.hopping_rate)
print("c:", graph1.c)