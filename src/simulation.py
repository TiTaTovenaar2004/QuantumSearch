import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation
import networkx as nx

from majority_vote_operator import majority_vote_operator
from plotting import plot_site_occupations, plot_marked_vertex_occupation_distribution, animate_marked_vertex_distribution, plot_success_probabilities

class Simulation:
    def __init__(self, states, occupations, times, graph, params):
        self.states = states
        self.occupations = occupations
        self.times = times
        self.graph = graph
        self.params = params

        self.success_probabilities = None
        self.rounds = None

    # --- Plotting states/occupations methods ---
    def plot_site_occupations(self):
        if self.occupations is None:
            raise ValueError("Site populations can only be plotted if the occupations were calculated during the simulation.")
        else:
            plot_site_occupations(self.occupations, self.params)
    
    def plot_marked_vertex_occupation_distribution(self):
        plot_marked_vertex_occupation_distribution(self.states[-1], self.params)

    def animate_marked_vertex_distribution(self):
        animate_marked_vertex_distribution(self.states, self.times, self.params)

    # --- Majority vote method ---
    def calculate_success_probabilities(self, rounds):
        # Check if rounds is a list of strictly positive integers and sort it in ascending order
        if not all(isinstance(r, int) and r > 0 for r in rounds):
            raise ValueError("The input of the 'calculate_success_probabilities'-method must be a list of strictly positive integers.")
        else:
            rounds = sorted(rounds)
        
        success_probabilities = []

        # Tensoring each state in self.states rounds[0] times with itself, so that we can apply the rounds[0] rounds majority vote operator
        total_states = [tensor([state for _ in range(rounds[0])]) for state in self.states]

        for idx, r in enumerate(rounds):
            op = majority_vote_operator(self.params['N'], self.params['M'], r, self.params['marked vertex'], self.params['dim per site'])
            probs = [expect(op, total_state) for total_state in total_states]
            success_probabilities.append(probs)

            # Tensoring each state rounds[idx + 1] - r times with itself, so that we can apply the rounds[idx + 1] rounds majority vote operator
            if idx < len(rounds) - 1:
                total_states = [tensor([total_state] + [state for _ in range(rounds[idx + 1] - r)]) for total_state, state in zip(total_states, self.states)]

        self.success_probabilities = np.array(success_probabilities)
        self.rounds = rounds
    
    # --- Plotting success probabilities method ---
    def plot_success_probabilities(self):
        if self.success_probabilities is None or self.rounds is None:
            raise ValueError("Success probabilities have not been calculated yet. Please run the 'calculate_success_probabilities'-method first.")
        else:
            plot_success_probabilities(self.success_probabilities, self.times, self.rounds)
