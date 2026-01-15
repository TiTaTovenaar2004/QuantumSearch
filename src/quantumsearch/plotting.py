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
def plot_estimated_success_probabilities(results, output_dir='results/plots', timestamp=None, plots_per_row=3):
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
    plots_per_row : int, optional
        Number of subplots to display per row

    Returns:
    --------
    None
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    import matplotlib as mpl

    BASE_FONT_SIZE = 13  # slightly larger than default

    mpl.rcParams.update({
        'font.size': BASE_FONT_SIZE,
        'axes.titlesize': BASE_FONT_SIZE + 3,
        'axes.labelsize': BASE_FONT_SIZE + 3,
        'xtick.labelsize': BASE_FONT_SIZE,
        'ytick.labelsize': BASE_FONT_SIZE,
        'legend.fontsize': BASE_FONT_SIZE,
    })

    # Sort results by: search_type (bosonic first), graph_type, N, p, then M
    def sort_key(result):
        # search_type: bosonic=0, fermionic=1 (so bosonic comes first)
        search_order = 0 if result['search_type'] == 'bosonic' else 1
        # Get p value from task_config if it exists, otherwise use 0 (for non-Erdos-Renyi graphs)
        p_value = 0
        if 'task_config' in result and 'graph_config' in result['task_config']:
            p_value = result['task_config']['graph_config'].get('p', 0)
        return (search_order, result['graph_type'], result['N'], p_value, result['M'])

    sorted_results = sorted(results, key=sort_key)

    # Group results by search_type, graph_type, N, and p for row organization
    from collections import defaultdict
    results_by_group = defaultdict(list)
    for result in sorted_results:
        # Get p value
        p_value = 0
        if 'task_config' in result and 'graph_config' in result['task_config']:
            p_value = result['task_config']['graph_config'].get('p', 0)
        # Group by (search_type, graph_type, N, p)
        group_key = (result['search_type'], result['graph_type'], result['N'], p_value)
        results_by_group[group_key].append(result)

    # Get ordered groups (already sorted by our sort_key)
    ordered_groups = []
    seen_groups = set()
    for result in sorted_results:
        p_value = 0
        if 'task_config' in result and 'graph_config' in result['task_config']:
            p_value = result['task_config']['graph_config'].get('p', 0)
        group_key = (result['search_type'], result['graph_type'], result['N'], p_value)
        if group_key not in seen_groups:
            ordered_groups.append(group_key)
            seen_groups.add(group_key)

    # Split results into rows with max plots_per_row columns
    # Each group (search_type, graph_type, N) can span multiple rows if it has more than plots_per_row M values
    plot_rows = []
    for group_key in ordered_groups:
        results_for_group = results_by_group[group_key]
        # Chunk into groups of plots_per_row
        for i in range(0, len(results_for_group), plots_per_row):
            plot_rows.append(results_for_group[i:i+plots_per_row])

    # Determine grid dimensions
    n_rows = len(plot_rows)
    n_cols = plots_per_row

    base_cols = 3  # reference width (original layout)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * base_cols, 5 * n_rows),
        squeeze=False
    )


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

                        if 'probabilities' in est:
                            probs = est['probabilities']
                            label = f"R={rounds}"
                            ax.plot(times, probs, '-', linewidth=2, label=label, color=colors[est_idx])
                        elif 'estimated_locations' in est:
                            label = f"R={rounds} (fast mode)"
                            ax.plot([], [], '-', linewidth=2, label=label, color=colors[est_idx])

                        if 'threshold' in est and 'lower_running_time' in est:
                            if threshold_value is None:
                                threshold_value = est['threshold']

                            lower_rt = est['lower_running_time']
                            upper_rt = est['upper_running_time']

                            if not np.isinf(lower_rt):
                                avg_rt = (lower_rt + upper_rt) / 2
                                avg_t = avg_rt / rounds
                                ax.axvline(avg_t, color=colors[est_idx], linestyle='--',
                                           alpha=0.4, linewidth=1.5)

                    if threshold_value is not None:
                        ax.axhline(threshold_value, color='gray', linestyle='--',
                                   alpha=0.7, linewidth=1.5, label=f'Threshold={threshold_value}')

            ax.set_xlabel('Time t')
            ax.set_ylabel('Success Probability')

            p = None
            if 'task_config' in result and 'graph_config' in result['task_config']:
                p = result['task_config']['graph_config'].get('p')

            if p is not None and p > 0:
                title = f'{search_type.capitalize()} search (M={M}) on the {graph_type} graph (N={N_val}, p={p})'
            else:
                title = f'{search_type.capitalize()} search (M={M}) on the {graph_type} graph (N={N_val})'

            ax.set_title(title, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            ax.set_ylim([0, 1.05])

        # Hide unused subplots in this row
        for col_idx in range(len(row_results), n_cols):
            axes[row_idx, col_idx].axis('off')

    plt.tight_layout()

    if timestamp:
        filename = f'estimated_success_probabilities_{timestamp}.png'
    else:
        filename = 'estimated_success_probabilities.png'

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {filepath}")

# --- Plot rounds comparison ---
def plot_rounds(results, output_dir='results/plots', timestamp=None, plots_per_row=3, main_round=2, rounds_plotted=None):
    """
    Plot success probabilities as a function of number of rounds for multiple tasks.

    Creates scatterplots showing the relationship between success probability at main_round
    versus success probabilities at other rounds, for each time point.

    Parameters:
    -----------
    results : list
        List of result dictionaries from load_results, each containing:
        - task_id, graph_type, N, search_type, M
        - times: array of time points
        - estimated_success_probabilities: 2D numpy array of shape (len(rounds), len(times))
        - task_config['estimation_config']['number_of_rounds']: list of round numbers
    output_dir : str
        Directory to save plots
    timestamp : str, optional
        Timestamp string to include in filenames
    plots_per_row : int, optional
        Number of subplots to display per row
    main_round : int, optional
        The reference number of rounds for x-axis (default: 2)
    rounds_plotted : list of int, optional
        List of round numbers to plot on y-axis (default: [2, 3, 4])

    Returns:
    --------
    None
    """
    import os
    from collections import defaultdict

    if rounds_plotted is None:
        rounds_plotted = [2, 3, 4]

    os.makedirs(output_dir, exist_ok=True)

    # Extract success_probabilities from each result and validate
    for result in results:
        if 'estimated_success_probabilities' not in result or result['estimated_success_probabilities'] is None:
            raise ValueError(f"Task {result['task_id']}: No estimated success probabilities found.")

        success_probs = result['estimated_success_probabilities']

        # Check if success_probabilities is a numpy array
        if not isinstance(success_probs, np.ndarray):
            raise ValueError(f"Task {result['task_id']}: Success probabilities must be a numpy array.")

        # Check if fast mode was used by looking for integer values (-1, 0, 1)
        # Fast mode uses integers, slow mode uses floats between 0 and 1
        unique_values = np.unique(success_probs)
        if np.all(np.isin(unique_values, [-1, 0, 1])):
            raise ValueError(f"Task {result['task_id']}: Success probabilities were calculated using fast mode. "
                           "This function requires slow mode (fast_mode=False) to plot actual probability values.")

        # Verify it's 2D
        if success_probs.ndim != 2:
            raise ValueError(f"Task {result['task_id']}: Success probabilities must be a 2D array of shape (rounds, times).")

        # Validate that rounds information exists
        if 'task_config' not in result or 'estimation_config' not in result['task_config']:
            raise ValueError(f"Task {result['task_id']}: Missing task_config or estimation_config.")

        if 'number_of_rounds' not in result['task_config']['estimation_config']:
            raise ValueError(f"Task {result['task_id']}: Missing number_of_rounds in estimation_config.")

    # Define colors for different rounds
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(rounds_plotted)))

    # Create a separate figure for each result
    saved_files = []
    for result in results:
        # Create a new figure for this result
        fig, ax = plt.subplots(figsize=(8, 6))

        times = result['times']
        task_id = result['task_id']
        graph_type = result['graph_type']
        N_val = result['N']
        search_type = result['search_type']
        M = result['M']

        # Extract success probabilities and rounds information
        success_probs = result['estimated_success_probabilities']
        number_of_rounds = result['task_config']['estimation_config']['number_of_rounds']

        # Convert to list if single integer
        if isinstance(number_of_rounds, int):
            number_of_rounds = [number_of_rounds]

        # Create mapping from round number to row index
        rounds_to_idx = {r: idx for idx, r in enumerate(number_of_rounds)}

        # Validate that main_round and rounds_plotted exist in the data
        if main_round not in rounds_to_idx:
            raise ValueError(f"Task {task_id}: main_round={main_round} not found in number_of_rounds={number_of_rounds}")

        for r in rounds_plotted:
            if r not in rounds_to_idx:
                raise ValueError(f"Task {task_id}: round {r} from rounds_plotted not found in number_of_rounds={number_of_rounds}")

        # Get the row index for main_round
        main_round_idx = rounds_to_idx[main_round]

        # For each time point (each column), create scatter points
        for time_idx in range(len(times)):
            # X-coordinate: success probability for main_round
            x_val = success_probs[main_round_idx, time_idx]

            # Y-coordinates: success probabilities for each round in rounds_plotted
            for plot_idx, round_num in enumerate(rounds_plotted):
                round_idx = rounds_to_idx[round_num]
                y_val = success_probs[round_idx, time_idx]

                # Plot the point
                ax.scatter(x_val, y_val, color=colors[plot_idx], alpha=0.6, s=30)

        # Add diagonal line y=x for reference
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='y=x')

        # Create legend for rounds
        legend_handles = []
        for plot_idx, round_num in enumerate(rounds_plotted):
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=colors[plot_idx],
                                             markersize=10, label=f'R={round_num}'))
        ax.legend(handles=legend_handles, fontsize=11, loc='upper left')

        ax.set_xlabel(f'Success Probability (R={main_round})', fontsize=12)
        ax.set_ylabel('Success Probability (other rounds)', fontsize=12)

        p = None
        if 'task_config' in result and 'graph_config' in result['task_config']:
            p = result['task_config']['graph_config'].get('p')

        if p is not None and p > 0:
            title = f'{search_type.capitalize()} search (M={M}) on the {graph_type} graph (N={N_val}, p={p})'
        else:
            title = f'{search_type.capitalize()} search (M={M}) on the {graph_type} graph (N={N_val})'

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()

        # Generate filename for this specific result
        if timestamp:
            filename = f'rounds_comparison_{search_type}_{graph_type}_N{N_val}_M{M}_{timestamp}.png'
        else:
            filename = f'rounds_comparison_{search_type}_{graph_type}_N{N_val}_M{M}.png'

        # Add p value to filename if it exists
        if p is not None and p > 0:
            if timestamp:
                filename = f'rounds_comparison_{search_type}_{graph_type}_N{N_val}_p{p}_M{M}_{timestamp}.png'
            else:
                filename = f'rounds_comparison_{search_type}_{graph_type}_N{N_val}_p{p}_M{M}.png'

        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

        saved_files.append(filepath)
        print(f"Plot saved to: {filepath}")

    print(f"\nTotal: {len(saved_files)} plot(s) saved.")
