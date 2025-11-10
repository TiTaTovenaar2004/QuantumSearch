import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation
import networkx as nx

# Visualize vector in as superposition of Fock basis states
def show_superposition(state):
    data = state.full().flatten()
    dims = state.dims[0]
    for idx, amp in enumerate(data):
        if abs(amp) > 1e-10:
            indices = np.unravel_index(idx, dims)
            print(f"{amp:.2f} |{','.join(map(str, indices))}>")

# Convert Fock basis representation into vector
def to_fock(fock_state, dim_per_site): # fock_state: [ [1, 0, 0], [0, 1, 0] ] => 3 vertices, 2 rounds
    temp = []
    for round in fock_state:
        for i in round:
            temp.append(i)
    state = tensor(basis(dim_per_site, i) for i in temp)
    
    return state

# Determines all combinations m_1, ..., m_r such that m1 + ... + mr = m_tot
def distribute(k, n): # Generates all ways to distribute k indistinguishable items into n distinguishable boxes
    if n == 1:
        yield (k,)
    else:
        for i in range(k + 1):
            for rest in distribute(k - i, n - 1):
                yield (i,) + rest

# Determines all combinations m_1, ..., m_r such that m1 + ... + mr = k
# and each m_i <= dim_per_site - 1
def distribute_with_cap(k, n, dim_per_site):
    if n == 1:
        # Only one site left: valid only if within capacity
        if k <= dim_per_site - 1:
            yield (k,)
    else:
        # Limit each site to at most dim_per_site - 1 particles
        for i in range(min(k, dim_per_site - 1) + 1):
            for rest in distribute_with_cap(k - i, n - 1, dim_per_site):
                yield (i,) + rest
