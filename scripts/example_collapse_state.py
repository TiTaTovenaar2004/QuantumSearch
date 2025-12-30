"""
Example showing in detail how collapse_and_extract_configuration works.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aron/Tijmen/QuantumSearch/src')

from qutip import *

# ============================================================
# Create an example quantum state
# ============================================================
# Let's say we have N=3 vertices, M=2 bosons (so dim_per_site = 3)
# We'll create a simple superposition state

N = 3  # Number of vertices
M = 2  # Number of bosons
dim_per_site = M + 1  # = 3 (can have 0, 1, or 2 bosons per vertex)

# Create a simple superposition:
# 0.6 |2,0,0> + 0.8 |0,1,1>
# This means: 60% amplitude for (2 bosons on vertex 0, 0 on vertex 1, 0 on vertex 2)
#             80% amplitude for (0 bosons on vertex 0, 1 on vertex 1, 1 on vertex 2)

state1 = tensor(basis(dim_per_site, 2), basis(dim_per_site, 0), basis(dim_per_site, 0))
state2 = tensor(basis(dim_per_site, 0), basis(dim_per_site, 1), basis(dim_per_site, 1))

state = 0.6 * state1 + 0.8 * state2
state = state.unit()  # Normalize

print("="*60)
print("QUANTUM STATE")
print("="*60)
print(f"N = {N} vertices")
print(f"M = {M} bosons")
print(f"dim_per_site = {dim_per_site}")
print()
print("State in Fock basis:")
data = state.full().flatten()
dims = state.dims[0]
for idx, amp in enumerate(data):
    if abs(amp) > 1e-10:
        indices = np.unravel_index(idx, dims)
        print(f"  {amp:.4f} |{','.join(map(str, indices))}>")
print()

# ============================================================
# STEP 1: Get the full state vector
# ============================================================
print("="*60)
print("STEP 1: Extract full state vector")
print("="*60)

data = state.full().flatten()
print(f"data = state.full().flatten()")
print(f"Shape: {data.shape}")
print(f"Length: {len(data)}")
print()
print("This is a complex vector with one entry per basis state.")
print(f"Total number of basis states = {dim_per_site}^{N} = {dim_per_site**N}")
print()

# Show first few non-zero entries
print("Non-zero entries in state vector:")
for idx, amp in enumerate(data):
    if abs(amp) > 1e-10:
        print(f"  data[{idx}] = {amp:.4f}")
print()

# ============================================================
# STEP 2: Calculate probabilities
# ============================================================
print("="*60)
print("STEP 2: Calculate probabilities")
print("="*60)

probs = np.abs(data)**2
print(f"probs = np.abs(data)**2")
print()
print("This gives the probability of measuring each basis state:")
for idx, prob in enumerate(probs):
    if abs(prob) > 1e-10:
        indices = np.unravel_index(idx, dims)
        config = ','.join(map(str, indices))
        print(f"  P(|{config}>) = {prob:.4f} = {prob*100:.1f}%")
print()
print(f"Sum of probabilities = {np.sum(probs):.10f} (should be 1.0)")
print()

# ============================================================
# STEP 3: Sample according to probabilities (collapse!)
# ============================================================
print("="*60)
print("STEP 3: Sample (collapse the wavefunction)")
print("="*60)

np.random.seed(42)  # For reproducibility
idx = np.random.choice(len(probs), p=probs)

print(f"idx = np.random.choice(len(probs), p=probs)")
print(f"idx = {idx}")
print()
print("This randomly selects one of the basis states according to")
print("their probabilities. This is the 'collapse'!")
print()

# ============================================================
# STEP 4: Convert linear index to configuration
# ============================================================
print("="*60)
print("STEP 4: Convert index to configuration")
print("="*60)

dims = tuple([dim_per_site] * N)
print(f"dims = ({', '.join(map(str, dims))})")
print()

config = np.unravel_index(idx, dims)
print(f"config = np.unravel_index(idx={idx}, dims={dims})")
print(f"config = {config}")
print()

print("Interpretation:")
print(f"  Vertex 0: {config[0]} bosons")
print(f"  Vertex 1: {config[1]} bosons")
print(f"  Vertex 2: {config[2]} bosons")
print(f"  Total: {sum(config)} bosons")
print()

# ============================================================
# Show how the indexing works
# ============================================================
print("="*60)
print("HOW THE INDEXING WORKS")
print("="*60)
print()
print("The state vector is flattened in 'C' order (row-major).")
print("For dims = (3, 3, 3), the mapping is:")
print()
print("Linear idx → Configuration (vertex 0, vertex 1, vertex 2)")
print("-" * 50)
for i in range(27):  # 3^3 = 27 total states
    conf = np.unravel_index(i, (3, 3, 3))
    marker = " <-- sampled!" if i == idx else ""
    print(f"  {i:2d}        → {conf}{marker}")
print()

# ============================================================
# Do multiple collapses
# ============================================================
print("="*60)
print("MULTIPLE COLLAPSES")
print("="*60)

num_samples = 10
print(f"Collapsing the state {num_samples} times:")
print()

for i in range(num_samples):
    idx_sample = np.random.choice(len(probs), p=probs)
    config_sample = np.unravel_index(idx_sample, dims)
    print(f"  Sample {i+1}: {config_sample}")

print()
print("Each collapse gives a configuration according to the")
print("quantum probabilities we calculated.")
