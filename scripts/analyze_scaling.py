"""
Analysis of memory usage and computation time for quantum search simulations.
Tests scaling with N (vertices) and M (particles) for both bosonic and fermionic search.
Uses precision=0.01, confidence=0.999 for success probability estimation.
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

# Estimation parameters
precision = 0.01
confidence = 0.999

# ============================================================
# Test 1: Bosonic - Scaling with N (number of vertices)
# ============================================================
print("="*70)
print("ANALYSIS 1: Bosonic - Scaling with N (number of vertices)")
print(f"Precision={precision}, Confidence={confidence}")
print("="*70)
print()

M_fixed = 2
N_values_bosonic = [3, 4, 5, 6]
num_time_points = 15

results_N_bosonic = {
    'N': [],
    'hilbert_dim': [],
    'state_size_mb': [],
    'simulate_time': [],
    'estimate_time': [],
    'total_memory_mb': []
}

for N in N_values_bosonic:
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
print(f"Precision={precision}, Confidence={confidence}")
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

# ============================================================
# Test 3: Bosonic - Scaling with M (number of particles)
# ============================================================
print("="*70)
print("ANALYSIS 3: Bosonic - Scaling with M (number of particles)")
print(f"Precision={precision}, Confidence={confidence}")
print("="*70)
print()

N_fixed = 4
M_values_bosonic = [1, 2, 3, 4]

results_M_bosonic = {
    'M': [],
    'hilbert_dim': [],
    'state_size_mb': [],
    'simulate_time': [],
    'estimate_time': [],
    'total_memory_mb': []
}

for M in M_values_bosonic:
    print(f"Testing BOSONIC N={N_fixed}, M={M}...")

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
    sim.estimate_success_probabilities(number_of_rounds=3, precision=precision, confidence=confidence)
    estimate_time = time.time() - start_time

    # Measure memory after
    mem_after = get_memory_usage_mb()

    print(f"  Simulation time: {simulate_time:.3f} s")
    print(f"  Estimation time: {estimate_time:.3f} s")
    print(f"  Memory increase: {mem_after - mem_before:.2f} MB")
    print()

    results_M_bosonic['M'].append(M)
    results_M_bosonic['hilbert_dim'].append(hilbert_dim)
    results_M_bosonic['state_size_mb'].append(state_size)
    results_M_bosonic['simulate_time'].append(simulate_time)
    results_M_bosonic['estimate_time'].append(estimate_time)
    results_M_bosonic['total_memory_mb'].append(mem_after - mem_before)

# ============================================================
# Create plots
# ============================================================
print("="*70)
print("CREATING PLOTS")
print("="*70)

fig, axes = plt.subplots(3, 3, figsize=(15, 14))

# Plot 1: Hilbert space dimension vs N (Bosonic vs Fermionic)
axes[0, 0].semilogy(results_N_bosonic['N'], results_N_bosonic['hilbert_dim'], 'o-', label=f'Bosonic (M={M_fixed})', linewidth=2, markersize=8)
axes[0, 0].semilogy(results_N_fermionic['N'], results_N_fermionic['hilbert_dim'], 's-', label=f'Fermionic (M={M_fermionic})', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Number of vertices N')
axes[0, 0].set_ylabel('Hilbert space dimension')
axes[0, 0].set_title('Hilbert Space Scaling with N')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Simulation time vs N
axes[0, 1].plot(results_N_bosonic['N'], results_N_bosonic['simulate_time'], 'o-', label=f'Bosonic (M={M_fixed})', linewidth=2, markersize=8)
axes[0, 1].plot(results_N_fermionic['N'], results_N_fermionic['simulate_time'], 's-', label=f'Fermionic (M={M_fermionic})', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Number of vertices N')
axes[0, 1].set_ylabel('Simulation time (seconds)')
axes[0, 1].set_title('Simulation Time vs N')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Estimation time vs N
axes[0, 2].plot(results_N_bosonic['N'], results_N_bosonic['estimate_time'], 'o-', label=f'Bosonic (M={M_fixed})', linewidth=2, markersize=8)
axes[0, 2].plot(results_N_fermionic['N'], results_N_fermionic['estimate_time'], 's-', label=f'Fermionic (M={M_fermionic})', linewidth=2, markersize=8)
axes[0, 2].set_xlabel('Number of vertices N')
axes[0, 2].set_ylabel('Estimation time (seconds)')
axes[0, 2].set_title(f'Estimation Time vs N (prec={precision}, conf={confidence})')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Hilbert space dimension vs M (Bosonic only)
axes[1, 0].semilogy(results_M_bosonic['M'], results_M_bosonic['hilbert_dim'], 'o-', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Number of particles M')
axes[1, 0].set_ylabel('Hilbert space dimension')
axes[1, 0].set_title(f'Hilbert Space Scaling with M (N={N_fixed}, Bosonic)')
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Simulation time vs M
axes[1, 1].plot(results_M_bosonic['M'], results_M_bosonic['simulate_time'], 'o-', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Number of particles M')
axes[1, 1].set_ylabel('Simulation time (seconds)')
axes[1, 1].set_title(f'Simulation Time vs M (N={N_fixed}, Bosonic)')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Estimation time vs M
axes[1, 2].plot(results_M_bosonic['M'], results_M_bosonic['estimate_time'], 'o-', linewidth=2, markersize=8)
axes[1, 2].set_xlabel('Number of particles M')
axes[1, 2].set_ylabel('Estimation time (seconds)')
axes[1, 2].set_title(f'Estimation Time vs M (N={N_fixed}, prec={precision}, conf={confidence})')
axes[1, 2].grid(True, alpha=0.3)

# Plot 7: Memory usage vs N (Bosonic vs Fermionic)
axes[2, 0].plot(results_N_bosonic['N'], results_N_bosonic['state_size_mb'], 'o--', label=f'Bosonic estimated (per state)', linewidth=2, markersize=8, alpha=0.7)
axes[2, 0].plot(results_N_bosonic['N'], results_N_bosonic['total_memory_mb'], 'o-', label=f'Bosonic measured (total)', linewidth=2, markersize=8)
axes[2, 0].plot(results_N_fermionic['N'], results_N_fermionic['state_size_mb'], 's--', label=f'Fermionic estimated (per state)', linewidth=2, markersize=8, alpha=0.7)
axes[2, 0].plot(results_N_fermionic['N'], results_N_fermionic['total_memory_mb'], 's-', label=f'Fermionic measured (total)', linewidth=2, markersize=8)
axes[2, 0].set_xlabel('Number of vertices N')
axes[2, 0].set_ylabel('Memory (MB)')
axes[2, 0].set_title('Memory Usage vs N')
axes[2, 0].legend(fontsize=8)
axes[2, 0].grid(True, alpha=0.3)

# Plot 8: Memory usage vs M (Bosonic)
axes[2, 1].plot(results_M_bosonic['M'], results_M_bosonic['state_size_mb'], 'o--', label='Estimated (per state)', linewidth=2, markersize=8, alpha=0.7)
axes[2, 1].plot(results_M_bosonic['M'], results_M_bosonic['total_memory_mb'], 'o-', label='Measured (total)', linewidth=2, markersize=8)
axes[2, 1].set_xlabel('Number of particles M')
axes[2, 1].set_ylabel('Memory (MB)')
axes[2, 1].set_title(f'Memory Usage vs M (N={N_fixed}, Bosonic)')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

# Plot 9: Time breakdown comparison
time_labels = ['Simulation', 'Estimation']
bosonic_times = [np.mean(results_N_bosonic['simulate_time']), np.mean(results_N_bosonic['estimate_time'])]
fermionic_times = [np.mean(results_N_fermionic['simulate_time']), np.mean(results_N_fermionic['estimate_time'])]

x = np.arange(len(time_labels))
width = 0.35

axes[2, 2].bar(x - width/2, bosonic_times, width, label=f'Bosonic (N avg)', color='steelblue')
axes[2, 2].bar(x + width/2, fermionic_times, width, label=f'Fermionic (N avg)', color='orange')
axes[2, 2].set_ylabel('Time (seconds)')
axes[2, 2].set_title('Average Time Breakdown')
axes[2, 2].set_xticks(x)
axes[2, 2].set_xticklabels(time_labels)
axes[2, 2].legend()
axes[2, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/plots/scaling_analysis.png', dpi=150)
print("\nPlots saved to results/plots/scaling_analysis.png")

# ============================================================
# Summary statistics
# ============================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nBosonic - Scaling with N (M={M_fixed} fixed):")
print(f"  Hilbert dimension grows as: (M+1)^N = {M_fixed+1}^N")
for i, N in enumerate(results_N_bosonic['N']):
    ratio_sim = results_N_bosonic['simulate_time'][i] / results_N_bosonic['simulate_time'][0] if i > 0 and results_N_bosonic['simulate_time'][0] > 0 else 1.0
    ratio_est = results_N_bosonic['estimate_time'][i] / results_N_bosonic['estimate_time'][0] if i > 0 and results_N_bosonic['estimate_time'][0] > 0 else 1.0
    ratio_mem = results_N_bosonic['total_memory_mb'][i] / results_N_bosonic['total_memory_mb'][0] if i > 0 and results_N_bosonic['total_memory_mb'][0] > 0 else 1.0
    print(f"  N={N}: dim={results_N_bosonic['hilbert_dim'][i]:,}, sim×{ratio_sim:.1f}, est×{ratio_est:.1f}, mem×{ratio_mem:.1f}")

print(f"\nFermionic - Scaling with N (M={M_fermionic} fixed):")
print(f"  Hilbert dimension grows as: 2^N")
for i, N in enumerate(results_N_fermionic['N']):
    ratio_sim = results_N_fermionic['simulate_time'][i] / results_N_fermionic['simulate_time'][0] if i > 0 and results_N_fermionic['simulate_time'][0] > 0 else 1.0
    ratio_est = results_N_fermionic['estimate_time'][i] / results_N_fermionic['estimate_time'][0] if i > 0 and results_N_fermionic['estimate_time'][0] > 0 else 1.0
    ratio_mem = results_N_fermionic['total_memory_mb'][i] / results_N_fermionic['total_memory_mb'][0] if i > 0 and results_N_fermionic['total_memory_mb'][0] > 0 else 1.0
    print(f"  N={N}: dim={results_N_fermionic['hilbert_dim'][i]:,}, sim×{ratio_sim:.1f}, est×{ratio_est:.1f}, mem×{ratio_mem:.1f}")

print(f"\nBosonic - Scaling with M (N={N_fixed} fixed):")
print(f"  Hilbert dimension grows as: (M+1)^N = (M+1)^{N_fixed}")
for i, M in enumerate(results_M_bosonic['M']):
    ratio_sim = results_M_bosonic['simulate_time'][i] / results_M_bosonic['simulate_time'][0] if i > 0 and results_M_bosonic['simulate_time'][0] > 0 else 1.0
    ratio_est = results_M_bosonic['estimate_time'][i] / results_M_bosonic['estimate_time'][0] if i > 0 and results_M_bosonic['estimate_time'][0] > 0 else 1.0
    ratio_mem = results_M_bosonic['total_memory_mb'][i] / results_M_bosonic['total_memory_mb'][0] if i > 0 and results_M_bosonic['total_memory_mb'][0] > 0 else 1.0
    print(f"  M={M}: dim={results_M_bosonic['hilbert_dim'][i]:,}, sim×{ratio_sim:.1f}, est×{ratio_est:.1f}, mem×{ratio_mem:.1f}")

plt.show()
