import numpy as np
import math
from qutip import *

# --- Sampling analysis ---
# Calculate the necessary number of samples based on Hoeffding's inequality
def number_of_samples(precision, confidence):
    """
    :param precision: accuracy for your guess of the mean (e.g., 0.05 for 5% precision)
    :param confidence: desired confidence level (e.g., 0.95 for 95% confidence)
    :return: number of samples needed to achieve the specified precision and confidence
    """

    if not (0 < confidence < 1):
        raise ValueError("Confidence level must be between 0 and 1.")
    if precision <= 0:
        raise ValueError("Precision must be a positive number.")

    return math.ceil(math.log((1 - confidence) / 2) / (-2*(precision ** 2)))

# --- Eigenvector manipulation ---
# Returns list of orthonormalized vectors
def orthonormalize_vectors(vecs):
    M = np.stack(vecs, axis=1)
    Q, R = np.linalg.qr(M, mode='reduced')
    return Q.T

# Orthonormalize eigenvectors corresponding to degenerate eigenvalues
def orthonormalize_eigenvectors(eigenvalues, eigenvectors):
    new_eigenvectors = np.copy(eigenvectors)
    eigenvalues_shifted = np.roll(eigenvalues, -1)
    degeneracy = (eigenvalues == eigenvalues_shifted).astype(int) # element i is 1 if eigenvalue i is degenerate with eigenvalue i+1
    current_idx = 0
    while current_idx < len(degeneracy):
        if degeneracy[current_idx] == 1:
            start_idx = current_idx
            while current_idx < len(degeneracy) and degeneracy[current_idx] == 1:
                current_idx += 1
            end_idx = current_idx + 1 # +1 to include the last degenerate eigenvalue
            degenerate_vecs = [eigenvectors[:, i] for i in range(start_idx, end_idx)]
            orthonormal_vecs = orthonormalize_vectors(degenerate_vecs)
            for i, vec in enumerate(orthonormal_vecs):
                new_eigenvectors[:, start_idx + i] = vec
        else:
            current_idx += 1

    return new_eigenvectors

# --- Quantum operators ---
# Bosonic number operator
def bosonic_number_operator(site, N, M): # Counts the number of bosons at a given site
        dim_per_site = M + 1
        ops = []
        for i in range(N):
            if i != site:
                ops.append(qeye(dim_per_site))
            else:
                ops.append(num(dim_per_site))

        return tensor(ops)

# Fermionic number operator
def fermionic_number_operator(site, N):

        return fcreate(N, site) * fdestroy(N, site)

# --- Functions for the majority vote operator ---
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

# --- Function for determining the success time of a quantum search
# Determine success time of quantum search, given a list of success probability thresholds and a list of success probabilities
def determine_success_time(thresholds, success_probabilities, times):
    if not all(0 < threshold < 1 for threshold in thresholds):
        raise ValueError("All thresholds must be strictly between 0 and 1.")
    thresholds = sorted(thresholds)

    success_times = []
    current_threshold_index = 0
    current_probability_index = 0
    while current_threshold_index < len(thresholds):
        if success_probabilities[current_probability_index] >= thresholds[current_threshold_index]:
            success_times.append(times[current_probability_index])
            current_threshold_index += 1
        else:
            current_probability_index += 1
            if current_probability_index >= len(success_probabilities):
                # If we reach the end of success probabilities without meeting all thresholds, append None
                for threshold in range(current_threshold_index, len(thresholds)):
                    success_times.append(math.inf)
                break

    return np.array(success_times)

# --- Testing functions ---
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