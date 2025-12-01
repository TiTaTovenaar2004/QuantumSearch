import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation
import networkx as nx

# --- Plot site populations ---
def plot_site_occupations(occupations, params):
    T = params['T']
    number_of_time_steps = params['number of time steps']
    N = params['N']

    # --- Create time array from 0 to T ---
    times = np.linspace(0, T, number_of_time_steps)

    # --- Plot site populations ---
    plt.figure(figsize=(8, 5))
    plt.tight_layout()

    # Plot the color map of site populations
    plt.imshow(np.array(occupations, dtype=float),
               aspect='auto',
               cmap='jet',
               interpolation='nearest',
               extent=[0, T, len(occupations) + 0.5, 0.5]) # y from 0.5 to num_rows + 0.5
                 

    # Labels
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Vertex', fontsize=12)

    # Colorbar
    plt.colorbar(label='Population')

    # Set y-ticks: one tick per row
    plt.yticks(np.arange(1, len(occupations) + 1))
    plt.tick_params(axis='y', which='both', right=True, labelright=True)

    plt.show()


# --- Plot marked vertex occupation distribution ---
def plot_marked_vertex_occupation_distribution(state, params): # Plots the occupation distribution of the marked vertex at time T
    # Unpack parameters
    N = params['N']
    dim_per_site = params['dim per site']
    marked_vertex = params['marked vertex']
    
    probs = np.zeros(dim_per_site)

    # Loop over all possible occupation numbers for the marked vertex
    for k in range(dim_per_site):
        # Projector onto the |k> state at the marked vertex
        projector_ops = []
        for i in range(N):
            if i == marked_vertex:
                projector_ops.append(basis(dim_per_site, k) * basis(dim_per_site, k).dag())
            else:
                projector_ops.append(qeye(dim_per_site))
        P_k = tensor(projector_ops)

        # Probability of measuring k bosons at the marked vertex
        probs[k] = expect(P_k, state)

    # Plot histogram
    plt.figure(figsize=(6, 4))
    plt.bar(range(dim_per_site), probs, color='steelblue', edgecolor='black')
    plt.xlabel("Number of bosons k on marked vertex")
    plt.ylabel("Probability P(k)")
    plt.xticks(range(dim_per_site))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- Animate marked vertex occupation distribution ---
def animate_marked_vertex_distribution(states, times, params):
    """
    Create an animation showing the probability distribution of finding k bosons
    on the marked vertex as a function of time.
    """

    # Unpack parameters
    N = params['N']
    dim_per_site = params['dim per site']
    marked_vertex = params['marked vertex']

    # Precompute probabilities P_k(t) for all times
    probs_time = np.zeros((len(times), dim_per_site))
    for ti, state in enumerate(states):
        for k in range(dim_per_site):
            projector_ops = []
            for i in range(N):
                if i == marked_vertex:
                    projector_ops.append(basis(dim_per_site, k) * basis(dim_per_site, k).dag())
                else:
                    projector_ops.append(qeye(dim_per_site))
            P_k = tensor(projector_ops)
            probs_time[ti, k] = expect(P_k, state)

    # --- Setup figure ---
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(dim_per_site), probs_time[0], color='steelblue', edgecolor='black')
    ax.set_ylim(0, 1)
    ax.set_xlabel("Number of bosons k on marked vertex")
    ax.set_ylabel("Probability P(k)")
    ax.set_xticks(range(dim_per_site))
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # --- Update function for each frame ---
    def update(frame):
        for bar, h in zip(bars, probs_time[frame]):
            bar.set_height(h)
        time_text.set_text(f"t = {times[frame]:.2f}")
        # Must return a SEQUENCE of artists (bars + text)
        return list(bars) + [time_text]

    # --- Create animation ---
    ani = FuncAnimation(fig, update, frames=len(times), blit=True, interval=100)

    plt.tight_layout()
    plt.show()

    return ani

# --- Plot success probabilities ---
def plot_success_probabilities(success_probabilities, times, rounds):
    """
    Plots the success probabilities as a function of time.

    Parameters
    ----------
    result : qutip.solver.Result
        Output from `sesolve` when `output='success probabilities'`.
        It should contain one expectation value array per R value.
    times : array-like
        Array of time points corresponding to the simulation.
    R : list[int]
        List of the number of majority vote rounds (same as passed to `bosonic_search`).
    """
    plt.figure(figsize=(8, 5))
    
    for idx, r in enumerate(rounds):
        plt.plot(times, success_probabilities[idx], label=f"R = {r}")

    plt.xlabel("Time t")
    plt.ylabel("Success probability P(t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
