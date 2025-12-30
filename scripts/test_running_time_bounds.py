"""Test script to verify running time bounds calculation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from quantumsearch.core.graph import Graph
from quantumsearch.core.simulation import Simulation

# Create a simple test case
print("Testing running time bounds calculation...")
print("-" * 60)

# Setup
graph = Graph(graph_type='complete', N=4)
sim = Simulation(search_type='bosonic', M=2, graph=graph)

# Run simulation
times = np.linspace(0, 10, 500)
sim.simulate(times)

# Estimate success probabilities with threshold
threshold = 0.8
print(f"\nEstimating success probabilities with threshold={threshold}...")
sim.estimate_success_probabilities(
    number_of_rounds=5,
    threshold=threshold,
    precision=0.02,
    confidence=0.95
)

# Display results
print("\nResults:")
for i, est in enumerate(sim.estimated_success_probabilities):
    print(f"\nEstimation {i+1}:")
    print(f"  Rounds: {est['rounds']}")
    print(f"  Threshold: {est['threshold']}")
    print(f"  Max probability: {np.max(est['probabilities']):.6f}")

    if 'lower_running_time' in est:
        lower_rt = est['lower_running_time']
        upper_rt = est['upper_running_time']

        if np.isinf(lower_rt):
            print(f"  Running time: Threshold never reached")
        else:
            print(f"  Lower running time: {lower_rt:.6f}")
            print(f"  Upper running time: {upper_rt:.6f}")
            print(f"  Running time range: [{lower_rt:.6f}, {upper_rt:.6f}]")
    else:
        print(f"  Running time bounds: Not calculated")

print("\n" + "-" * 60)
print("Test completed successfully!")
