import numpy as np
from qutip import *
import time

from quantumsearch.core.bosonic_search import bosonic_search
from quantumsearch.core.fermionic_search import fermionic_search
from quantumsearch.core.majority_vote_operator import majority_vote_operator
from quantumsearch.core.utils import determine_success_time, bosonic_number_operator, fermionic_number_operator, number_of_samples

class Simulation:
    def __init__(self, search_type, M, graph, hopping_rate = None):
        # --- General attributes ---
        self.states = np.array([])
        self.times = np.array([])
        self.occupations = np.array([])
        self.occupation_times = np.array([])
        self.estimated_success_probabilities = []  # List of dictionaries
        self.status = 'Initialized'

        # --- Timing attributes ---
        self.simulation_time = 0.0  # Total time spent in simulate()
        self.estimation_time = 0.0  # Total time spent in estimate_success_probabilities()

        # --- Parameters ---
        if search_type not in ['bosonic', 'fermionic']:
            raise ValueError("The 'search_type' parameter must be either 'bosonic' or 'fermionic'.")
        self.search_type = search_type
        self.M = M
        if search_type == 'bosonic':
            self.dim_per_site = M + 1
        else:
            self.dim_per_site = 2
        self.graph = graph
        if hopping_rate == None:
            try :
                self.hopping_rate = graph.hopping_rate
            except AttributeError:
                graph.calculate_hopping_rate()
                self.hopping_rate = graph.hopping_rate
        else:
            self.hopping_rate = hopping_rate

    # --- Action that runs the simulation for some times ---
    def simulate(self, times):
        start_time = time.time()

        # Filter out times that are already in self.times
        new_times = np.array([t for t in times if t not in self.times])

        # If all times already exist, skip simulation
        if len(new_times) == 0:
            return

        if self.search_type == 'bosonic':
            states = bosonic_search(self.M, self.graph, self.hopping_rate, new_times)
        else:
            states = fermionic_search(self.M, self.graph, self.hopping_rate, new_times)

        # Add new states and times to existing arrays
        self.states = np.append(self.states, states)
        self.times = np.append(self.times, new_times)

        # Sort chronologically while maintaining correspondence
        sorted_indices = np.argsort(self.times)
        self.times = self.times[sorted_indices]
        self.states = self.states[sorted_indices]

        # Update simulation time
        end_time = time.time()
        self.simulation_time += end_time - start_time

        return self

    # --- Action that calculates vertex occupations for some times ---
    def calculate_occupations(self, times, vertex=None):
        if vertex is None:
            vertex = self.graph.marked_vertex

        # Check if all given times are in self.times
        if not all(t in self.times for t in times):
            missing_times = [t for t in times if t not in self.times]
            raise ValueError(f"The following times are not in self.times: {missing_times}")

        # Get the appropriate number operator
        if self.search_type == 'bosonic':
            num_op = bosonic_number_operator(vertex, self.graph.N, self.M)
        else:
            num_op = fermionic_number_operator(vertex, self.graph.N)

        # Calculate occupations for the requested times
        occupations = []
        for t in times:
            # Find the index of this time in self.times
            idx = np.where(self.times == t)[0][0]
            # Get the corresponding state
            state = self.states[idx]
            # Calculate expectation value of number operator
            occupation = expect(num_op, state)
            occupations.append(occupation)

        # Add new occupations and times to existing arrays
        self.occupations = np.append(self.occupations, occupations)
        self.occupation_times = np.append(self.occupation_times, times)

        # Sort chronologically while maintaining correspondence
        sorted_indices = np.argsort(self.occupation_times)
        self.occupation_times = self.occupation_times[sorted_indices]
        self.occupations = self.occupations[sorted_indices]

        return self

    # --- Action that checks the number of extrema in the occupations ---
    def number_of_extrema(self):
        if len(self.occupations) < 3:
            return 0

        # Calculate differences between consecutive points
        diff = np.diff(self.occupations)

        # Count local maxima: points where slope changes from positive to negative
        # (diff[i-1] > 0 and diff[i] < 0)
        maxima = np.sum((diff[:-1] > 0) & (diff[1:] < 0))

        # Count local minima: points where slope changes from negative to positive
        # (diff[i-1] < 0 and diff[i] > 0)
        minima = np.sum((diff[:-1] < 0) & (diff[1:] > 0))

        return maxima + minima

    # --- Action that estimates success probabilities via sampling ---
    def estimate_success_probabilities(self, number_of_rounds, precision, confidence):
        start_time = time.time()

        if len(self.states) == 0:
            raise ValueError("States are required to estimate success probabilities.")

        num_samples = number_of_samples(precision, confidence)
        estimated_probs = []

        for state in self.states:
            # Measure the state 'num_samples' times
            data = state.full().flatten()
            probs = np.abs(data)**2

            # Sample according to the probabilities
            dims = tuple([self.dim_per_site] * self.graph.N)
            indices = np.random.choice(len(probs), size=num_samples*number_of_rounds, p=probs)

            # Convert indices to configurations
            configs = np.array([np.unravel_index(idx, dims) for idx in indices])
            configs = configs.reshape((num_samples, number_of_rounds, self.graph.N))

            # Apply a majority vote over each sample
            total_particles_per_vertex = configs.sum(axis=1)  # (num_samples, N)

            marked_counts = total_particles_per_vertex[:, self.graph.marked_vertex]
            max_other_counts = np.array([np.max(np.delete(sample, self.graph.marked_vertex)) for sample in total_particles_per_vertex])

            success_count = np.sum(marked_counts > max_other_counts)

            # Estimate success probability
            estimated_prob = success_count / num_samples
            estimated_probs.append(estimated_prob)

        # Store results as a dictionary
        result = {
            'rounds': number_of_rounds,
            'precision': precision,
            'confidence': confidence,
            'probabilities': np.array(estimated_probs)
        }

        # Append to list if storing multiple estimates
        if not hasattr(self, 'estimated_success_probabilities') or len(self.estimated_success_probabilities) == 0:
            self.estimated_success_probabilities = [result]
        else:
            self.estimated_success_probabilities.append(result)

        end_time = time.time()
        self.estimation_time += end_time - start_time

        return self

    # ------ Old methods ------
    # --- Majority vote method ---
    def calculate_success_probabilities(self, rounds):
        # Check if rounds is a list of strictly positive integers and sort it in ascending order
        if not all(isinstance(r, int) and r > 0 for r in rounds):
            raise ValueError("The input of the 'calculate_success_probabilities'-method must be a list of strictly positive integers.")
        elif len(self.states) == 0:
            raise ValueError("Success probabilities can only be calculated if the states were calculated during the simulation.")
        else:
            rounds = sorted(rounds)

        success_probabilities = []

        # Get parameters from simulation object
        N = self.graph.N
        marked_vertex = self.graph.marked_vertex
        M = self.M

        # Determine dimension per site based on search type
        if self.search_type == 'bosonic':
            dim_per_site = M + 1
        else:  # fermionic
            dim_per_site = 2

        # Tensoring each state in self.states rounds[0] times with itself, so that we can apply the rounds[0] rounds majority vote operator
        total_states = [tensor([state for _ in range(rounds[0])]) for state in self.states]

        for idx, r in enumerate(rounds):
            op = majority_vote_operator(N, M, r, marked_vertex, dim_per_site)
            probs = [expect(op, total_state) for total_state in total_states]
            success_probabilities.append(probs)

            # Tensoring each state rounds[idx + 1] - r times with itself, so that we can apply the rounds[idx + 1] rounds majority vote operator
            if idx < len(rounds) - 1:
                total_states = [tensor([total_state] + [state for _ in range(rounds[idx + 1] - r)]) for total_state, state in zip(total_states, self.states)]

        self.success_probabilities = np.array(success_probabilities)
        self.rounds = rounds

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


