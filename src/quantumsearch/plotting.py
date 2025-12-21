import numpy as np
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation, FFMpegWriter

# --- Plot site populations ---
def plot_site_occupations(simulation, filename='plot_site_occupations.png'):
    occupations = simulation.occupations

    if occupations is None:
            raise ValueError("Site populations can only be plotted if the occupations were calculated during the simulation.")

    T = simulation.params['T']

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

    plt.savefig(filename)
    plt.close()


# --- Plot marked vertex occupation distribution ---
def plot_marked_vertex_occupation_distribution(state, params, filename): # Plots the occupation distribution of the marked vertex at time T
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
    plt.xlabel("Number of particles k on marked vertex")
    plt.ylabel("Probability P(k)")
    plt.xticks(range(dim_per_site))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- Animate marked vertex occupation distribution ---
def animate_marked_vertex_distribution(states, times, params, filename):
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
    ax.set_xlabel("Number of particles k on marked vertex")
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
    writer = FFMpegWriter(fps=10, bitrate=1800)
    ani.save(filename, writer=writer)

    plt.close(fig)

    return ani

# --- Plot success probabilities ---
def plot_success_probabilities(success_probabilities, times, rounds, filename):

    plt.figure(figsize=(8, 5))
    
    for idx, r in enumerate(rounds):
        plt.plot(times, success_probabilities[idx], label=f"R = {r}")

    plt.xlabel("Time t")
    plt.ylabel("Success probability P(t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- Plot with error bars or shaded region ---
def plot_with_error(x, y, filename, shaded=True, percentiles=(0.5, 99.5)):
    """
    Plots the mean of measurements with error bars or shaded region containing 99% of values.

    Parameters:
    - x : 1D array-like, x-axis values
    - y : 2D array-like, shape (len(x), n_measurements)
          y[i] contains measurements corresponding to x[i]
    - shaded : bool, if True, uses shaded region instead of error bars
    - title : str, plot title
    """
    x = np.array(x)
    y = np.array(y)
    
    if y.shape[0] != len(x):
        raise ValueError("Number of rows in y must match length of x")
    
    # Compute mean for central line
    y_mean = np.mean(y, axis=1)
    
    # Compute percentiles for the interval
    lower = np.percentile(y, percentiles[0], axis=1)
    upper = np.percentile(y, percentiles[1], axis=1)
    
    if shaded:
        plt.plot(x, y_mean, 'o-', label='Mean')
        plt.fill_between(x, lower, upper, alpha=0.2, label=str(100 - (percentiles[0] + 100 - percentiles[1])) + '% interval')
    else:
        # Compute asymmetric error bars
        y_err_lower = y_mean - lower
        y_err_upper = upper - y_mean
        plt.errorbar(x, y_mean, yerr=[y_err_lower, y_err_upper], fmt='o', capsize=5, label='Mean ± ' + str(100 - (percentiles[0] + 100 - percentiles[1])) + '% interval')
    
    plt.xlabel('Number of vertices N')
    plt.ylabel('c value')
    plt.legend()
    plt.savefig(filename)
    plt.close()