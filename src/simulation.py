import numpy as np
from qutip import *
import time

from majority_vote_operator import majority_vote_operator
from plotting import plot_site_occupations, plot_marked_vertex_occupation_distribution, animate_marked_vertex_distribution, plot_success_probabilities
from utils import determine_success_time

class Simulation:
    def __init__(self, states, occupations, times, graph, params):
        self.states = states
        self.occupations = occupations
        self.times = times
        self.graph = graph
        self.params = params

        self.success_probabilities = None # [[success probabilities for MV of rounds[0] rounds], [success probabilities for MV of rounds[1] rounds], ...]
        self.rounds = None # List of the number of rounds used in the majority vote
        self.running_times = None # [[running times for MV of rounds[0] rounds for different thresholds], [running times for MV of rounds[1] rounds for different thresholds], ...]
        self.lowest_running_times = None # [lowest running times for different thresholds (over all MV's)]
        self.rounds_of_lowest_running_times = None # [number of rounds of MV that gives the lowest running time for each threshold]

        self.simulation_calculation_time = self.params['simulation calculation time']
        self.hopping_rate_calculation_time = graph.hopping_rate_calculation_time
        self.running_time_calculation_time = None

    # --- Plotting states/occupations methods ---
    def plot_site_occupations(self, filename="plot_site_occupations.png"):
        if self.occupations is None:
            raise ValueError("Site populations can only be plotted if the occupations were calculated during the simulation.")
        else:
            plot_site_occupations(self.occupations, self.params, filename)
    
    def plot_marked_vertex_occupation_distribution(self, filename="plot_marked_vertex_occupation_distribution.png"):
        if self.states is None:
            raise ValueError("Marked vertex occupation distribution can only be plotted if the states were calculated during the simulation.")
        else:
            plot_marked_vertex_occupation_distribution(self.states[-1], self.params, filename)

    def animate_marked_vertex_distribution(self, filename="animate_marked_vertex_distribution.mp4"):
        if self.states is None:
            raise ValueError("Marked vertex occupation distribution can only be animated if the states were calculated during the simulation.")
        else:
            animate_marked_vertex_distribution(self.states, self.times, self.params, filename)

    # --- Majority vote method ---
    def calculate_success_probabilities(self, rounds):
        # Check if rounds is a list of strictly positive integers and sort it in ascending order
        if not all(isinstance(r, int) and r > 0 for r in rounds):
            raise ValueError("The input of the 'calculate_success_probabilities'-method must be a list of strictly positive integers.")
        elif self.states is None:
            raise ValueError("Success probabilities can only be calculated if the states were calculated during the simulation.")
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
    def plot_success_probabilities(self, filename="plot_success_probabilities.png"):
        if self.success_probabilities is None or self.rounds is None:
            raise ValueError("Success probabilities have not been calculated yet. Please run the 'calculate_success_probabilities'-method first.")
        else:
            plot_success_probabilities(self.success_probabilities, self.times, self.rounds, filename)
    
    # --- Method for determining the running times for each MV (so MV of rounds[0] rounds, MV of rounds[1] rounds, etc..) ---
    def determine_running_times(self, thresholds):
        if self.success_probabilities is None or self.rounds is None:
            raise ValueError("Success probabilities have not been calculated yet. Please run the 'calculate_success_probabilities'-method first.")
        else:
            running_times = []
            for idx, r in enumerate(self.rounds):
                success_times = determine_success_time(thresholds, self.success_probabilities[idx], self.times) # Determine the time to reach each success probability threshold
                running_times.append(r * success_times) # The running time is the number of rounds times the time to reach the success probability

            self.running_times = np.array(running_times) # Returns a 2D array with shape (len(rounds), len(thresholds))

    # --- Method for determining the lowest running time (over all MV's) for each threshold ---
    def determine_lowest_running_times(self, thresholds, stop_condition = 2):
        start_time = time.time()

        if self.states is None:
            raise ValueError("Running times can only be determined if the states were calculated during the simulation.")
        
        # Calculate running times for increasing number of rounds majority vote, until increasing the number of rounds no longer decreases the running time (for any threshold)
        running_times = np.empty((0, len(thresholds))) # running_times[i][j] is the running time for threshold j using rounds[i] rounds majority vote
        stop_conditions = np.zeros(len(thresholds)) # stop_conditions[i] is incremented whenever increasing the number of rounds (for threshold i) does not decrease the running time. If all elements reach 'stop_condition', we stop increasing the number of rounds, and return the lowest running times found so far.
        current_number_of_rounds = 1 # Start with majority vote of 1 round
        total_states = self.states # Start with the original states (no tensoring yet)

        while True:
            # Determine success probabilities for current number of rounds
            op = majority_vote_operator(self.params['N'], self.params['M'], current_number_of_rounds, self.params['marked vertex'], self.params['dim per site'])
            success_probabilities = [expect(op, total_state) for total_state in total_states]

            # Determine success times for current number of rounds
            success_times = determine_success_time(thresholds, success_probabilities, self.times)
            running_times = np.vstack([running_times, current_number_of_rounds * success_times])

            # Update stop conditions
            if current_number_of_rounds > 1: # If we have done only 1 round so far, we cannot compare yet
                for i in range(len(thresholds)):
                    if running_times[current_number_of_rounds - 1, i] >= running_times[current_number_of_rounds - 2, i]:
                        stop_conditions[i] += 1
            
            # Setting up for next iteration (if stop condition not yet met)
            if np.any(stop_conditions < stop_condition):
                current_number_of_rounds += 1
                total_states = [tensor([total_state, state]) for total_state, state in zip(total_states, self.states)]
            else:
                break

        # Determine lowest running times for each threshold
        self.lowest_running_times = np.min(running_times, axis = 0) 
        self.rounds_of_lowest_running_times = np.argmin(running_times, axis = 0) + 1    

        end_time = time.time()
        self.running_time_calculation_time = end_time - start_time      
