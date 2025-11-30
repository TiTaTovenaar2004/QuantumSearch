import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation
import networkx as nx

from majority_vote_operator import majority_vote_operator
from plotting import plot_site_populations, plot_marked_vertex_occupation_distribution, animate_marked_vertex_distribution, plot_success_probabilities

class Simulation:
    def __init__(self, result, times, graph, params):
        self.result = result
        self.times = times
        self.graph = graph
        self.params = params

    def plot_site_populations(self):
        if self.params['output'] != 'occupations':
            raise ValueError("Site populations can only be plotted if output is 'occupations'")
        else:
            plot_site_populations(self.result, self.params)
    
    def plot_marked_vertex_occupation_distribution(self):
        if self.params['output'] != 'occupations':
            raise ValueError("Marked vertex occupation distribution can only be plotted if output is 'occupations'")
        else:
            plot_marked_vertex_occupation_distribution(self.result, self.times, self.params)