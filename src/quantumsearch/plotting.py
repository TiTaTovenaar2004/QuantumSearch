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

    # Group results by N, then sort by M within each N
    from collections import defaultdict
    results_by_N = defaultdict(list)
    for result in results:
        results_by_N[result['N']].append(result)

    # Sort each group by M
    for N in results_by_N:
        results_by_N[N].sort(key=lambda r: r['M'])

    # Sort N values
    N_values = sorted(results_by_N.keys())

    # Split results into rows with max 3 columns
    # Each N can span multiple rows if it has more than 3 M values
    plot_rows = []
    for N in N_values:
        results_for_N = results_by_N[N]
        # Chunk into groups of 3
        for i in range(0, len(results_for_N), 3):
            plot_rows.append(results_for_N[i:i+3])

    # Determine grid dimensions
    n_rows = len(plot_rows)
    n_cols = 3  # Maximum 3 columns

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows), squeeze=False)

    # Place each result in the appropriate subplot
    for row_idx, row_results in enumerate(plot_rows):
        for col_idx, result in enumerate(row_results):
            ax = axes[row_idx, col_idx]

            times = result['times']
            task_id = result['task_id']
            graph_type = result['graph_type']
            N_val = result['N']
            search_type = result['search_type']
            M = result['M']

            # Plot each estimation (multiple rounds may be present for same task)
            # Check if we have the new array format or old list format
            if 'estimated_success_probabilities' in result and result['estimated_success_probabilities'] is not None:
                # New format: array of success probabilities
                if isinstance(result['estimated_success_probabilities'], np.ndarray):
                    success_probs = result['estimated_success_probabilities']

                    # Get running times and rounds from new format
                    if 'lower_running_times' in result and 'upper_running_times' in result:
                        lower_rts = result['lower_running_times']
                        upper_rts = result['upper_running_times']

                        # Try to get number_of_rounds and threshold from task_config
                        if 'task_config' in result and 'estimation_config' in result['task_config']:
                            number_of_rounds = result['task_config']['estimation_config']['number_of_rounds']
                            threshold_value = result['task_config']['estimation_config']['threshold']
                            if isinstance(number_of_rounds, int):
                                number_of_rounds = [number_of_rounds]
                        else:
                            # Fallback
                            number_of_rounds = list(range(1, len(lower_rts) + 1))
                            threshold_value = None

                        colors = plt.cm.viridis(np.linspace(0, 0.9, len(number_of_rounds)))

                        # Plot success probabilities for each round
                        for idx, rounds in enumerate(number_of_rounds):
                            if success_probs.ndim == 2:
                                probs = success_probs[idx, :]
                                # Plot probability curve
                                label = f"R={rounds}"
                                ax.plot(times, probs, '-', linewidth=2, label=label, color=colors[idx])

                            # Add vertical line for running time
                            lower_rt = lower_rts[idx]
                            upper_rt = upper_rts[idx]
                            if not np.isinf(lower_rt):
                                avg_rt = (lower_rt + upper_rt) / 2
                                avg_t = avg_rt / rounds
                                ax.axvline(avg_t, color=colors[idx], linestyle='--',
                                         alpha=0.4, linewidth=1.5)

                        # Add threshold line
                        if threshold_value is not None:
                            ax.axhline(threshold_value, color='gray', linestyle='--',
                                     alpha=0.7, linewidth=1.5, label=f'Threshold={threshold_value}')
                else:
                    # Old format: list of dictionaries
                    colors = plt.cm.viridis(np.linspace(0, 0.9, len(result['estimated_success_probabilities'])))
                    threshold_value = None

                    for est_idx, est in enumerate(result['estimated_success_probabilities']):
                        rounds = est['rounds']
                        precision = est['precision']

                        # Check if we have probabilities (slow mode) or estimated_locations (fast mode)
                        if 'probabilities' in est:
                            # Slow mode: plot the probability curve
                            probs = est['probabilities']

                            # Plot main line
                            label = f"R={rounds}"
                            line, = ax.plot(times, probs, '-', linewidth=2, label=label, color=colors[est_idx])
                        elif 'estimated_locations' in est:
                            # Fast mode: just show the label, no probability curve
                            label = f"R={rounds} (fast mode)"
                            # Create an empty plot just for the label
                            ax.plot([], [], '-', linewidth=2, label=label, color=colors[est_idx])

                        # Store threshold and add vertical line for average running time
                        if 'threshold' in est and 'lower_running_time' in est:
                            if threshold_value is None:
                                threshold_value = est['threshold']

                            lower_rt = est['lower_running_time']
                            upper_rt = est['upper_running_time']

                            # Add vertical line at average running time if threshold is reached
                            if not np.isinf(lower_rt):
                                avg_rt = (lower_rt + upper_rt) / 2
                                avg_t = avg_rt / rounds  # Convert back to single-run time
                                ax.axvline(avg_t, color=colors[est_idx], linestyle='--',
                                         alpha=0.4, linewidth=1.5)

                    # Add threshold line last (so it appears at bottom of legend)
                    if threshold_value is not None:
                        ax.axhline(threshold_value, color='gray', linestyle='--',
                                 alpha=0.7, linewidth=1.5, label=f'Threshold={threshold_value}')

            ax.set_xlabel('Time t', fontsize=10)
            ax.set_ylabel('Success Probability', fontsize=10)
            ax.set_title(f'{search_type.capitalize()} search (M={M}) on the {graph_type} graph (N={N_val})',
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='upper right')
            ax.set_ylim([0, 1.05])

        # Hide unused subplots in this row
        for col_idx in range(len(row_results), n_cols):
            axes[row_idx, col_idx].axis('off')

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