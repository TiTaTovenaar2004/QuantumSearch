import numpy as np
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation, FFMpegWriter

# --- Plot marked vertex occupation distribution ---
def plot_marked_vertex_occupation_distribution(simulation, filename='results/plots/marked_vertex_occupation_distribution.png'): # Plots the occupation distribution of the marked vertex at time T
    if len(simulation.states) == 0:
        raise ValueError("States are required to plot the marked vertex occupation distribution.")

    state = simulation.states[-1]  # Final state

    # Get parameters from simulation object
    N = simulation.graph.N
    marked_vertex = simulation.graph.marked_vertex

    # Determine dimension per site based on search type
    if simulation.search_type == 'bosonic':
        dim_per_site = simulation.M + 1  # Bosonic: 0 to M particles per site
    else:  # fermionic
        dim_per_site = 2  # Fermionic: 0 or 1 particle per site

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

        # Probability of measuring k particles at the marked vertex
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
def animate_marked_vertex_distribution(simulation, filename='results/plots/marked_vertex_occupation_distribution_animation.mp4'):
    if len(simulation.states) == 0:
        raise ValueError("States are required to animate the marked vertex occupation distribution.")

    states = simulation.states
    times = simulation.times

    # Get parameters from simulation object
    N = simulation.graph.N
    marked_vertex = simulation.graph.marked_vertex

    # Determine dimension per site based on search type
    if simulation.search_type == 'bosonic':
        dim_per_site = simulation.M + 1  # Bosonic: 0 to M particles per site
    else:  # fermionic
        dim_per_site = 2  # Fermionic: 0 or 1 particle per site

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
def plot_success_probabilities(simulation, filename='results/plots/plot_success_probabilities.png'):
    if simulation.success_probabilities is None or simulation.rounds is None:
        raise ValueError("Success probabilities have not been calculated yet. Please run the 'calculate_success_probabilities'-method first.")

    success_probabilities = simulation.success_probabilities
    times = simulation.times
    rounds = simulation.rounds

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
def plot_with_error(x, y, filename='results/plots/plot_with_error.png', shaded=True, percentiles=(0.5, 99.5)):
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

# --- Plot estimated success probabilities for multiple tasks ---
def plot_estimated_success_probabilities(results, output_dir='results/plots', timestamp=None):
    """
    Plot estimated success probabilities for each task in the results.

    Creates individual plots for each task showing how the success probability
    evolves over time.

    Parameters:
    -----------
    results : list
        List of result dictionaries from load_results, each containing:
        - task_id, graph_type, N, search_type, M
        - times: array of time points
        - estimated_success_probabilities: list of estimation results
    output_dir : str
        Directory to save plots
    timestamp : str, optional
        Timestamp string to include in filenames

    Returns:
    --------
    None
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Determine grid layout
    n_tasks = len(results)
    n_cols = min(3, n_tasks)  # Max 3 columns
    n_rows = (n_tasks + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))

    # Handle case of single plot
    if n_tasks == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_tasks > 1 else axes

    for idx, result in enumerate(results):
        ax = axes[idx] if n_tasks > 1 else axes[0]

        times = result['times']
        task_id = result['task_id']
        graph_type = result['graph_type']
        N = result['N']
        search_type = result['search_type']
        M = result['M']

        # Plot each estimation (multiple rounds may be present for same task)
        if result['estimated_success_probabilities']:
            colors = plt.cm.viridis(np.linspace(0, 0.9, len(result['estimated_success_probabilities'])))

            for est_idx, est in enumerate(result['estimated_success_probabilities']):
                probs = est['probabilities']
                rounds = est['rounds']
                precision = est['precision']

                # Plot main line
                label = f"R={rounds}, ε={precision}"
                line, = ax.plot(times, probs, '-', linewidth=2, label=label, color=colors[est_idx])

                # Mark maximum
                max_idx = np.argmax(probs)
                ax.plot(times[max_idx], probs[max_idx], 'o', markersize=8,
                       color=colors[est_idx], zorder=10)

                # Add threshold line and running time bounds if available
                if 'threshold' in est and 'lower_running_time' in est:
                    threshold = est['threshold']
                    lower_rt = est['lower_running_time']
                    upper_rt = est['upper_running_time']

                    # Add horizontal threshold line (only once)
                    if est_idx == 0:
                        ax.axhline(threshold, color='gray', linestyle=':',
                                 alpha=0.5, linewidth=1, label=f'Threshold={threshold}')

                    # Add vertical lines for running time bounds if threshold is reached
                    if not np.isinf(lower_rt):
                        lower_t = lower_rt / rounds  # Convert back to single-run time
                        upper_t = upper_rt / rounds
                        ax.axvline(lower_t, color=colors[est_idx], linestyle='--',
                                 alpha=0.4, linewidth=1.5)
                        ax.axvline(upper_t, color=colors[est_idx], linestyle='--',
                                 alpha=0.4, linewidth=1.5)

        ax.set_xlabel('Time t', fontsize=10)
        ax.set_ylabel('Success Probability', fontsize=10)
        ax.set_title(f'Task {task_id}: {graph_type} (N={N}, {search_type}, M={M})',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim([0, 1.05])

    # Hide unused subplots
    for idx in range(n_tasks, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save figure
    if timestamp:
        filename = f'estimated_success_probabilities_{timestamp}.png'
    else:
        filename = 'estimated_success_probabilities.png'

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {filepath}")