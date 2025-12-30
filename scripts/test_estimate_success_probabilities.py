"""
Script to test estimate_success_probabilities method and plot results.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/aron/Tijmen/QuantumSearch/src')

from quantumsearch.core.graph import Graph
from quantumsearch.core.simulation import Simulation

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

# Estimate success probabilities for different numbers of rounds
print("\nEstimating success probabilities...")
rounds_list = [1, 2, 3]
precision = 0.01
confidence = 0.99

for rounds in rounds_list:
    print(f"  Rounds = {rounds}...")
    sim.estimate_success_probabilities(rounds, precision, confidence)

print(f"\nEstimated {len(sim.estimated_success_probabilities)} sets of success probabilities")

# Plot results
print("\nCreating plot...")
plt.figure(figsize=(10, 6))

for idx, result in enumerate(sim.estimated_success_probabilities):
    rounds = result['rounds']
    probs = result['probabilities']
    precision = result['precision']
    confidence = result['confidence']

    plt.plot(sim.times, probs, 'o-', label=f"R = {rounds} (±{precision*100:.0f}%, {confidence*100:.0f}% conf.)")

plt.xlabel("Time t")
plt.ylabel("Estimated Success Probability")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.05, 1.05)
plt.tight_layout()

# Save plot
plt.savefig('results/plots/estimated_success_probabilities.png')
print("Plot saved to results/plots/estimated_success_probabilities.png")

# Also print some statistics
print("\n" + "="*60)
print("STATISTICS")
print("="*60)
for result in sim.estimated_success_probabilities:
    rounds = result['rounds']
    probs = result['probabilities']
    max_prob = np.max(probs)
    max_time = sim.times[np.argmax(probs)]
    print(f"\nRounds = {rounds}:")
    print(f"  Max probability: {max_prob:.4f} at t = {max_time:.2f}")
    print(f"  Mean probability: {np.mean(probs):.4f}")
    print(f"  Final probability: {probs[-1]:.4f}")

plt.show()
