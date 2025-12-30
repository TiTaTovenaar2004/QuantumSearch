"""
Script to plot success probabilities for a small simulation with majority vote.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aron/Tijmen/QuantumSearch/src')

from quantumsearch.core.graph import Graph
from quantumsearch.core.simulation import Simulation
from quantumsearch.plotting import plot_success_probabilities

# Create a small complete graph
print("Creating graph...")
graph = Graph('line', N=5, marked_vertex=0)

# Create simulation
print("Creating simulation...")
sim = Simulation(search_type='fermionic', M=2, graph=graph)
# Run simulation over a time range
print("Running simulation...")
times = np.linspace(0, 20, 100)
sim.simulate(times)

print(f"Simulation complete: {len(sim.times)} time points")
print(f"Time range: [{sim.times[0]:.2f}, {sim.times[-1]:.2f}]")

# Calculate success probabilities for different rounds
print("Calculating success probabilities...")
rounds = [1, 2, 3]
sim.calculate_success_probabilities(rounds)

print(f"Success probabilities calculated for rounds: {rounds}")

# Plot success probabilities
print("Creating plot...")
plot_success_probabilities(sim, filename='results/plots/success_probabilities.png')

print("Plot saved to results/plots/success_probabilities.png")
