import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv  # Bessel function J_n


# ================================================================
# 1. CLASSICAL CONTINUOUS-TIME RANDOM WALK (CTRW)
# ================================================================

def simulate_ctrw(T, gamma=1.0):
    """
    Simulate a single classical continuous-time random walk on the infinite line.
    Starts at 0.
    Returns the final position after time T.
    """
    t = 0.0
    pos = 0
    
    while t < T:
        dt = np.random.exponential(1/(2*gamma))  # waiting time
        t += dt
        if t >= T:
            break
        pos += np.random.choice([-1, 1])
    
    return pos


def ctrw_distribution(T, Nsamples=20000, gamma=1.0):
    data = np.array([simulate_ctrw(T, gamma) for _ in range(Nsamples)])
    positions, counts = np.unique(data, return_counts=True)
    probs = counts / Nsamples
    return positions, probs, data


def plot_ctrw_distribution(T, Nsamples=20000, gamma=1.0):
    positions, probs, _ = ctrw_distribution(T, Nsamples, gamma)
    plt.figure(figsize=(8,4))
    plt.bar(positions, probs, width=0.8)
    plt.title(f"Classical CTRW distribution at time T={T}")
    plt.xlabel("Position")
    plt.ylabel("Probability")
    plt.grid(True, alpha=0.3)
    plt.show()


# ================================================================
# 2. QUANTUM CONTINUOUS-TIME WALK (CTQW)
# ================================================================

def ctqw_wavefunction(T, n_values, gamma=1.0):
    """
    psi(n,T) = i^n J_n(2 gamma T)
    """
    return (1j**n_values) * jv(n_values, 2 * gamma * T)


def ctqw_distribution(T, n_max=100, gamma=1.0):
    n_values = np.arange(-n_max, n_max+1)
    psi = ctqw_wavefunction(T, n_values, gamma)
    probs = np.abs(psi)**2
    return n_values, probs


def plot_ctqw_distribution(T, n_max=100, gamma=1.0):
    n_values, probs = ctqw_distribution(T, n_max, gamma)
    plt.figure(figsize=(8,4))
    plt.bar(n_values, probs, width=0.8)
    plt.title(f"Quantum CTQW distribution at time T={T}")
    plt.xlabel("Position")
    plt.ylabel("Probability")
    plt.grid(True, alpha=0.3)
    plt.show()


# ================================================================
# 3. VARIANCE COMPUTATION
# ================================================================

def classical_variance_vs_time(T_values, Nsamples=20000, gamma=1.0):
    variances = []
    for T in T_values:
        _, _, samples = ctrw_distribution(T, Nsamples, gamma)
        variances.append(np.var(samples))
    return np.array(variances)


def quantum_variance_vs_time(T_values, n_max=200, gamma=1.0):
    variances = []
    for T in T_values:
        n, p = ctqw_distribution(T, n_max, gamma)
        variance = np.sum((n**2) * p)
        variances.append(variance)
    return np.array(variances)


# ================================================================
# 4. PLOT VARIANCE COMPARISON
# ================================================================

def plot_variances(T_values, Nsamples=20000, n_max=200, gamma=1.0):
    classical_var = classical_variance_vs_time(T_values, Nsamples, gamma)
    quantum_var = quantum_variance_vs_time(T_values, n_max, gamma)

    plt.figure(figsize=(8,5))
    plt.plot(T_values, classical_var, label="Classical CTRW variance", linewidth=2)
    plt.plot(T_values, quantum_var, label="Quantum CTQW variance", linewidth=2)
    plt.xlabel("Time T")
    plt.ylabel("Variance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


# ================================================================
# 5. NEW: PLOT FINAL DISTRIBUTIONS (CLASSICAL vs QUANTUM)
# ================================================================

def plot_final_distributions(T, Nsamples=20000, n_max=50, gamma=1.0):
    """
    Side-by-side comparison of classical and quantum distributions at time T.
    """
    positions_c, probs_c, _ = ctrw_distribution(T, Nsamples, gamma)
    positions_q, probs_q = ctqw_distribution(T, n_max, gamma)

    plt.figure(figsize=(12,5))

    # Classical
    plt.subplot(1, 2, 1)
    plt.bar(positions_c, probs_c, width=0.8)
    plt.xlabel("Position")
    plt.ylabel("Probability")
    plt.grid(True, alpha=0.3)

    # Quantum
    plt.subplot(1, 2, 2)
    plt.bar(positions_q, probs_q, width=0.8)
    plt.xlabel("Position")
    plt.ylabel("Probability")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ================================================================
# Example usage
# ================================================================
if __name__ == "__main__":
    T_values = np.linspace(0, 10, 30)

    # Variance plot
    # plot_variances(T_values)

    # Final distribution comparison
    plot_final_distributions(T=20)
