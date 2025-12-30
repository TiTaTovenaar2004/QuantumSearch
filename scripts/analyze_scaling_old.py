"""
Analysis of memory usage and computation time for quantum search simulations.
Tests scaling with N (vertices) and M (particles).
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import psutil
import os
sys.path.insert(0, '/home/aron/Tijmen/QuantumSearch/src')

from quantumsearch.core.graph import Graph
from quantumsearch.core.simulation import Simulation

def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def estimate_state_size_mb(N, M, search_type='bosonic'):
    """Estimate memory size of quantum state in MB."""
    if search_type == 'bosonic':
        dim_per_site = M + 1
    else:
        dim_per_site = 2

    hilbert_space_dim = dim_per_site ** N
    # Complex128: 16 bytes per element
    size_bytes = hilbert_space_dim * 16
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

# ============================================================
# Test 1: Bosonic - Scaling with N (number of vertices)
# ============================================================
print("="*70)
print("ANALYSIS 1: Bosonic - Scaling with N (number of vertices)")
print("="*70)
print()

M_fixed = 2
N_values = [3, 4, 5, 6, 7]
num_time_points = 15
precision = 0.01
confidence = 0.999

results_N_bosonic = {
    'N': [],
    'hilbert_dim': [],
    'state_size_mb': [],
    'simulate_time': [],
    'estimate_time': [],
    'total_memory_mb': []
}

for N in N_values:
    print(f"Testing BOSONIC N={N}, M={M_fixed}...")

    # Calculate Hilbert space dimension
    dim_per_site = M_fixed + 1
    hilbert_dim = dim_per_site ** N
    state_size = estimate_state_size_mb(N, M_fixed)

    print(f"  Hilbert space dimension: {hilbert_dim:,}")
    print(f"  Estimated state size: {state_size:.2f} MB")

    # Skip if Hilbert space is too large
    if hilbert_dim > 5e6:
        print(f"  Skipping: Hilbert space too large ({hilbert_dim:,} > 5,000,000)")
        continue

    # Create graph and simulation
    graph = Graph('complete', N=N, marked_vertex=0)
    sim = Simulation(search_type='bosonic', M=M_fixed, graph=graph)

    # Measure memory before
    mem_before = get_memory_usage_mb()

    # Simulate
    times = np.linspace(0, 20, num_time_points)
    start_time = time.time()
    sim.simulate(times)
    simulate_time = time.time() - start_time

    # Estimate success probabilities
    start_time = time.time()
    sim.estimate_success_probabilities(number_of_rounds=3, precision=precision, confidence=confidence)
    estimate_time = time.time() - start_time

    # Measure memory after
    mem_after = get_memory_usage_mb()

    print(f"  Simulation time: {simulate_time:.3f} s")
    print(f"  Estimation time: {estimate_time:.3f} s")
    print(f"  Memory increase: {mem_after - mem_before:.2f} MB")
    print()

    results_N_bosonic['N'].append(N)
    results_N_bosonic['hilbert_dim'].append(hilbert_dim)
    results_N_bosonic['state_size_mb'].append(state_size)
    results_N_bosonic['simulate_time'].append(simulate_time)
    results_N_bosonic['estimate_time'].append(estimate_time)
    results_N_bosonic['total_memory_mb'].append(mem_after - mem_before)

# ============================================================
# Test 2: Fermionic - Scaling with N (number of vertices)
# ============================================================
print("="*70)
print("ANALYSIS 2: Fermionic - Scaling with N (number of vertices)")
print("="*70)
print()

M_fermionic = 1  # Number of fermions
N_values_fermionic = [3, 4, 5, 6, 7, 8]

results_N_fermionic = {
    'N': [],
    'hilbert_dim': [],
    'state_size_mb': [],
    'simulate_time': [],
    'estimate_time': [],
    'total_memory_mb': []
}

for N in N_values_fermionic:
    print(f"Testing FERMIONIC N={N}, M={M_fermionic}...")

    # Calculate Hilbert space dimension
    dim_per_site = 2  # Fermionic
    hilbert_dim = dim_per_site ** N
    state_size = estimate_state_size_mb(N, M_fermionic, search_type='fermionic')

    print(f"  Hilbert space dimension: {hilbert_dim:,}")
    print(f"  Estimated state size: {state_size:.2f} MB")

    # Skip if Hilbert space is too large
    if hilbert_dim > 5e6:
        print(f"  Skipping: Hilbert space too large ({hilbert_dim:,} > 5,000,000)")
        continue

    # Create graph and simulation
    graph = Graph('complete', N=N, marked_vertex=0)
    sim = Simulation(search_type='fermionic', M=M_fermionic, graph=graph)

    # Measure memory before
    mem_before = get_memory_usage_mb()

    # Simulate
    times = np.linspace(0, 20, num_time_points)
    start_time = time.time()
    sim.simulate(times)
    simulate_time = time.time() - start_time

    # Estimate success probabilities
    start_time = time.time()
    sim.estimate_success_probabilities(number_of_rounds=3, precision=precision, confidence=confidence)
    estimate_time = time.time() - start_time

    # Measure memory after
    mem_after = get_memory_usage_mb()

    print(f"  Simulation time: {simulate_time:.3f} s")
    print(f"  Estimation time: {estimate_time:.3f} s")
    print(f"  Memory increase: {mem_after - mem_before:.2f} MB")
    print()

    results_N_fermionic['N'].append(N)
    results_N_fermionic['hilbert_dim'].append(hilbert_dim)
    results_N_fermionic['state_size_mb'].append(state_size)
    results_N_fermionic['simulate_time'].append(simulate_time)
    results_N_fermionic['estimate_time'].append(estimate_time)
    results_N_fermionic['total_memory_mb'].append(mem_after - mem_before)
    'simulate_time': [],
    'estimate_time': [],
    'total_memory_mb': []
}

for N in N_values:
    print(f"Testing N={N}, M={M_fixed}...")

    # Calculate Hilbert space dimension
    dim_per_site = M_fixed + 1
    hilbert_dim = dim_per_site ** N
    state_size = estimate_state_size_mb(N, M_fixed)

    print(f"  Hilbert space dimension: {hilbert_dim:,}")
    print(f"  Estimated state size: {state_size:.2f} MB")

    # Skip if Hilbert space is too large
    if hilbert_dim > 5e6:
        print(f"  Skipping: Hilbert space too large ({hilbert_dim:,} > 5,000,000)")
        continue

    # Create graph and simulation
    graph = Graph('complete', N=N, marked_vertex=0)
    sim = Simulation(search_type='bosonic', M=M_fixed, graph=graph)

    # Measure memory before
    mem_before = get_memory_usage_mb()

    # Simulate
    times = np.linspace(0, 20, num_time_points)
    start_time = time.time()
    sim.simulate(times)
    simulate_time = time.time() - start_time

    # Measure memory after simulation
    mem_after_sim = get_memory_usage_mb()

    # Estimate success probabilities
    start_time = time.time()
    sim.estimate_success_probabilities(number_of_rounds=3, precision=0.1, confidence=0.9)
    estimate_time = time.time() - start_time

    # Measure memory after estimation
    mem_after = get_memory_usage_mb()

    print(f"  Simulation time: {simulate_time:.3f} s")
    print(f"  Estimation time: {estimate_time:.3f} s")
    print(f"  Memory increase: {mem_after - mem_before:.2f} MB")
    print()

    results_N['N'].append(N)
    results_N['hilbert_dim'].append(hilbert_dim)
    results_N['state_size_mb'].append(state_size)
    results_N['simulate_time'].append(simulate_time)
    results_N['estimate_time'].append(estimate_time)
    results_N['total_memory_mb'].append(mem_after - mem_before)

# ============================================================
# Test 2: Scaling with M (number of particles)
# ============================================================
print("="*70)
print("ANALYSIS 2: Scaling with M (number of particles)")
print("="*70)
print()

N_fixed = 4
M_values = [1, 2, 3, 4, 5, 6, 7, 8]

results_M = {
    'M': [],
    'hilbert_dim': [],
    'state_size_mb': [],
    'simulate_time': [],
    'estimate_time': [],
    'total_memory_mb': []
}

for M in M_values:
    print(f"Testing N={N_fixed}, M={M}...")

    # Calculate Hilbert space dimension
    dim_per_site = M + 1
    hilbert_dim = dim_per_site ** N_fixed
    state_size = estimate_state_size_mb(N_fixed, M)

    print(f"  Hilbert space dimension: {hilbert_dim:,}")
    print(f"  Estimated state size: {state_size:.2f} MB")

    # Skip if Hilbert space is too large
    if hilbert_dim > 5e6:
        print(f"  Skipping: Hilbert space too large ({hilbert_dim:,} > 5,000,000)")
        continue

    # Create graph and simulation
    graph = Graph('complete', N=N_fixed, marked_vertex=0)
    sim = Simulation(search_type='bosonic', M=M, graph=graph)

    # Measure memory before
    mem_before = get_memory_usage_mb()

    # Simulate
    times = np.linspace(0, 20, num_time_points)
    start_time = time.time()
    sim.simulate(times)
    simulate_time = time.time() - start_time

    # Estimate success probabilities
    start_time = time.time()
    sim.estimate_success_probabilities(number_of_rounds=3, precision=0.1, confidence=0.9)
    estimate_time = time.time() - start_time

    # Measure memory after
    mem_after = get_memory_usage_mb()

    print(f"  Simulation time: {simulate_time:.3f} s")
    print(f"  Estimation time: {estimate_time:.3f} s")
    print(f"  Memory increase: {mem_after - mem_before:.2f} MB")
    print()

    results_M['M'].append(M)
    results_M['hilbert_dim'].append(hilbert_dim)
    results_M['state_size_mb'].append(state_size)
    results_M['simulate_time'].append(simulate_time)
    results_M['estimate_time'].append(estimate_time)
    results_M['total_memory_mb'].append(mem_after - mem_before)

# ============================================================
# Test 3: Scaling with precision and confidence
# ============================================================
print("="*70)
print("ANALYSIS 3: Scaling with precision (confidence fixed)")
print("="*70)
print()

# Use small system to isolate precision/confidence effects
N_small = 4
M_small = 2
confidence_fixed = 0.95

# Test different precision values (lower precision = more samples)
precision_values = [0.15, 0.10, 0.07, 0.05, 0.03]

results_precision = {
    'precision': [],
    'num_samples': [],
    'estimate_time': [],
    'memory_mb': []
}

# Pre-simulate once so we can reuse the states
graph_small = Graph('complete', N=N_small, marked_vertex=0)
sim_small = Simulation(search_type='bosonic', M=M_small, graph=graph_small)
times_small = np.linspace(0, 20, 10)  # Fewer time points for speed
sim_small.simulate(times_small)

from quantumsearch.core.utils import number_of_samples

for prec in precision_values:
    num_samp = number_of_samples(prec, confidence_fixed)
    print(f"Testing precision={prec:.2f} (confidence={confidence_fixed:.2f})...")
    print(f"  Number of samples required: {num_samp:,}")

    # Measure memory before
    mem_before = get_memory_usage_mb()

    # Estimate success probabilities
    start_time = time.time()
    sim_small.estimate_success_probabilities(number_of_rounds=3, precision=prec, confidence=confidence_fixed)
    estimate_time = time.time() - start_time

    # Measure memory after
    mem_after = get_memory_usage_mb()

    print(f"  Estimation time: {estimate_time:.3f} s")
    print(f"  Memory increase: {mem_after - mem_before:.2f} MB")
    print()

    results_precision['precision'].append(prec)
    results_precision['num_samples'].append(num_samp)
    results_precision['estimate_time'].append(estimate_time)
    results_precision['memory_mb'].append(mem_after - mem_before)

# ============================================================
print("="*70)
print("ANALYSIS 4: Scaling with confidence (precision fixed)")
print("="*70)
print()

precision_fixed = 0.10
confidence_values = [0.80, 0.90, 0.95, 0.99]

results_confidence = {
    'confidence': [],
    'num_samples': [],
    'estimate_time': [],
    'memory_mb': []
}

for conf in confidence_values:
    num_samp = number_of_samples(precision_fixed, conf)
    print(f"Testing confidence={conf:.2f} (precision={precision_fixed:.2f})...")
    print(f"  Number of samples required: {num_samp:,}")

    # Measure memory before
    mem_before = get_memory_usage_mb()

    # Estimate success probabilities
    start_time = time.time()
    sim_small.estimate_success_probabilities(number_of_rounds=3, precision=precision_fixed, confidence=conf)
    estimate_time = time.time() - start_time

    # Measure memory after
    mem_after = get_memory_usage_mb()

    print(f"  Estimation time: {estimate_time:.3f} s")
    print(f"  Memory increase: {mem_after - mem_before:.2f} MB")
    print()

    results_confidence['confidence'].append(conf)
    results_confidence['num_samples'].append(num_samp)
    results_confidence['estimate_time'].append(estimate_time)
    results_confidence['memory_mb'].append(mem_after - mem_before)

# ============================================================
# Create plots
# ============================================================
print("="*70)
print("CREATING PLOTS")
print("="*70)

fig, axes = plt.subplots(3, 3, figsize=(15, 14))

# Plot 1: Hilbert space dimension vs N
axes[0, 0].semilogy(results_N['N'], results_N['hilbert_dim'], 'o-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Number of vertices N')
axes[0, 0].set_ylabel('Hilbert space dimension')
axes[0, 0].set_title(f'Hilbert Space Scaling (M={M_fixed})')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Memory vs N
axes[0, 1].plot(results_N['N'], results_N['state_size_mb'], 'o-', label='Estimated (per state)', linewidth=2, markersize=8)
axes[0, 1].plot(results_N['N'], results_N['total_memory_mb'], 's-', label='Measured (total)', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Number of vertices N')
axes[0, 1].set_ylabel('Memory (MB)')
axes[0, 1].set_title(f'Memory Usage (M={M_fixed})')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Computation time vs N
axes[0, 2].plot(results_N['N'], results_N['simulate_time'], 'o-', label='Simulate', linewidth=2, markersize=8)
axes[0, 2].plot(results_N['N'], results_N['estimate_time'], 's-', label='Estimate', linewidth=2, markersize=8)
axes[0, 2].set_xlabel('Number of vertices N')
axes[0, 2].set_ylabel('Time (seconds)')
axes[0, 2].set_title(f'Computation Time (M={M_fixed})')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Hilbert space dimension vs M
axes[1, 0].semilogy(results_M['M'], results_M['hilbert_dim'], 'o-', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Number of particles M')
axes[1, 0].set_ylabel('Hilbert space dimension')
axes[1, 0].set_title(f'Hilbert Space Scaling (N={N_fixed})')
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Memory vs M
axes[1, 1].plot(results_M['M'], results_M['state_size_mb'], 'o-', label='Estimated (per state)', linewidth=2, markersize=8)
axes[1, 1].plot(results_M['M'], results_M['total_memory_mb'], 's-', label='Measured (total)', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Number of particles M')
axes[1, 1].set_ylabel('Memory (MB)')
axes[1, 1].set_title(f'Memory Usage (N={N_fixed})')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Computation time vs M
axes[1, 2].plot(results_M['M'], results_M['simulate_time'], 'o-', label='Simulate', linewidth=2, markersize=8)
axes[1, 2].plot(results_M['M'], results_M['estimate_time'], 's-', label='Estimate', linewidth=2, markersize=8)
axes[1, 2].set_xlabel('Number of particles M')
axes[1, 2].set_ylabel('Time (seconds)')
axes[1, 2].set_title(f'Computation Time (N={N_fixed})')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

# Plot 7: Number of samples vs precision
axes[2, 0].plot(results_precision['precision'], results_precision['num_samples'], 'o-', linewidth=2, markersize=8, color='purple')
axes[2, 0].set_xlabel('Precision')
axes[2, 0].set_ylabel('Number of samples required')
axes[2, 0].set_title(f'Samples vs Precision (conf={confidence_fixed:.2f})')
axes[2, 0].invert_xaxis()  # Lower precision on the right
axes[2, 0].grid(True, alpha=0.3)

# Plot 8: Estimation time vs precision
axes[2, 1].plot(results_precision['precision'], results_precision['estimate_time'], 'o-', linewidth=2, markersize=8, color='purple')
axes[2, 1].set_xlabel('Precision')
axes[2, 1].set_ylabel('Estimation time (seconds)')
axes[2, 1].set_title(f'Time vs Precision (conf={confidence_fixed:.2f}, N={N_small}, M={M_small})')
axes[2, 1].invert_xaxis()
axes[2, 1].grid(True, alpha=0.3)

# Plot 9: Estimation time vs confidence
axes[2, 2].plot(results_confidence['confidence'], results_confidence['estimate_time'], 'o-', linewidth=2, markersize=8, color='green')
axes[2, 2].set_xlabel('Confidence')
axes[2, 2].set_ylabel('Estimation time (seconds)')
axes[2, 2].set_title(f'Time vs Confidence (prec={precision_fixed:.2f}, N={N_small}, M={M_small})')
axes[2, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/scaling_analysis.png', dpi=150)
print("\nPlots saved to results/plots/scaling_analysis.png")

# ============================================================
# Summary statistics
# ============================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nScaling with N (M={M_fixed} fixed):")
print(f"  Hilbert dimension grows as: (M+1)^N = {M_fixed+1}^N")
for i, N in enumerate(results_N['N']):
    ratio_sim = results_N['simulate_time'][i] / results_N['simulate_time'][0] if i > 0 else 1.0
    ratio_est = results_N['estimate_time'][i] / results_N['estimate_time'][0] if i > 0 else 1.0
    print(f"  N={N}: dim={results_N['hilbert_dim'][i]:,}, sim×{ratio_sim:.1f}, est×{ratio_est:.1f}")

print(f"\nScaling with M (N={N_fixed} fixed):")
print(f"  Hilbert dimension grows as: (M+1)^N = (M+1)^{N_fixed}")
for i, M in enumerate(results_M['M']):
    ratio_sim = results_M['simulate_time'][i] / results_M['simulate_time'][0] if i > 0 else 1.0
    ratio_est = results_M['estimate_time'][i] / results_M['estimate_time'][0] if i > 0 else 1.0
    print(f"  M={M}: dim={results_M['hilbert_dim'][i]:,}, sim×{ratio_sim:.1f}, est×{ratio_est:.1f}")

print(f"\nScaling with precision (confidence={confidence_fixed:.2f} fixed, N={N_small}, M={M_small}):")
print(f"  Number of samples grows as: n ≈ ln(2/(1-conf)) / (2*prec²)")
for i, prec in enumerate(results_precision['precision']):
    ratio_time = results_precision['estimate_time'][i] / results_precision['estimate_time'][0] if i > 0 else 1.0
    print(f"  prec={prec:.2f}: samples={results_precision['num_samples'][i]:,}, time×{ratio_time:.1f}")

print(f"\nScaling with confidence (precision={precision_fixed:.2f} fixed, N={N_small}, M={M_small}):")
for i, conf in enumerate(results_confidence['confidence']):
    ratio_time = results_confidence['estimate_time'][i] / results_confidence['estimate_time'][0] if i > 0 else 1.0
    print(f"  conf={conf:.2f}: samples={results_confidence['num_samples'][i]:,}, time×{ratio_time:.1f}")

plt.show()
