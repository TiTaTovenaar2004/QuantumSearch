import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation
import networkx as nx

# --- Plot site populations ---
def plot_site_populations(result, params):
    # Unpack parameters
    T = params['T']
    number_of_time_steps = params['number_of_time_steps']
    N = params['N']

    # --- Plot site populations ---
    plt.figure(figsize=(8, 5))
    plt.tight_layout()

    # Plot the color map of site populations
    plt.imshow(np.array(result.expect, dtype=float), aspect='auto', cmap='jet', interpolation='nearest')

    # Labels
    plt.xlabel(r'Time, $\gamma t$', fontsize=12)
    plt.ylabel(r'Vertex, $i$', fontsize=12)

    # Colorbar
    plt.colorbar(label='Population')

    # Add yticks on both sides
    plt.tick_params(axis='y', which='both', right=True, labelright=True)

    # --- Rescale x-axis ticks ---
    # Get current x-ticks (these correspond to index positions, not actual times)
    ax = plt.gca()
    num_steps = number_of_time_steps  # from your global variable
    scale_factor = (num_steps / T) * N

    # Compute new tick positions corresponding to multiples of pi
    # Let’s cover the full time range
    max_pi = int(np.floor(T * math.pi / (T / N)))  # a rough upper bound
    max_tick = number_of_time_steps
    pi_ticks = np.arange(0, max_tick, int(scale_factor * math.pi))

    # Simpler, consistent with your request: 0, pi, 2pi, 3pi...
    xtick_labels = [rf'{i}$\pi$' if i > 0 else '0' for i in range(len(pi_ticks))]
    plt.xticks(pi_ticks, xtick_labels)

    plt.title('Site Populations over Time', fontsize=14)
    plt.show()

# --- Plot marked vertex occupation distribution ---
def plot_marked_vertex_occupation_distribution(state): # Plots the occupation distribution of the marked vertex at time T
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
    plt.title(f"Occupation distribution on marked vertex {marked_vertex}")
    plt.xticks(range(dim_per_site))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- Animate marked vertex occupation distribution ---
def animate_marked_vertex_distribution(states, times):
    """
    Create an animation showing the probability distribution of finding k bosons
    on the marked vertex as a function of time.
    """

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
    ax.set_title(f"Occupation distribution on marked vertex {marked_vertex}")
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