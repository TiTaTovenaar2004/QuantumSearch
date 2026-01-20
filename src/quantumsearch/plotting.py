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
def plot_estimated_success_probabilities(results, output_dir='results/plots', timestamp=None, plots_per_row=3, dashed_lines='all'):
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
        'legend.fontsize': BASE_FONT_SIZE + 0.5,
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

                                if dashed_lines == 'all': # all dashed lines are colored
                                    ax.axvline(avg_t, color=colors[idx], linestyle='--',
                                            alpha=0.4, linewidth=1.5)
                                elif dashed_lines == 'new_best': # only new best dashed lines are colored
                                    ave_rts = (lower_rts + upper_rts) / 2
                                    ave_rts_until_idx = ave_rts[:idx + 1]
                                    best_idx = np.argmin(ave_rts_until_idx)
                                    if best_idx == idx:
                                        ax.axvline(avg_t, color=colors[idx],
                                                   alpha=0.4, linewidth=2)
                                        ax.text(avg_t, -0.1, f'R={rounds}',
                                            ha='center', va='top',
                                            fontsize=15, color='black')
                                    else:
                                        ax.axvline(avg_t, color='gray', linestyle='--',
                                                   alpha=0.2, linewidth=1.5)
                                else:
                                    raise ValueError("Invalid value for dashed_lines parameter.")

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

# --- Plot fermionic runtimes ---
def plot_fermionic_runtimes(results, output_dir='results/plots', timestamp=None, ignore_first_N=7, plot_fits=False):
    """
    Plot fermionic runtimes as a function of N with optional power-law fits.

    For each result, extracts the best running time (lowest average of lower and upper bounds)
    and the corresponding M and R values. Then, for each N, selects the configuration with
    the lowest running time and plots N vs running time with error intervals. Optionally fits
    three power-law models to the data.

    Parameters:
    -----------
    results : list
        List of result dictionaries from load_results, each containing:
        - N, M
        - lower_running_times: array of lower runtime bounds
        - upper_running_times: array of upper runtime bounds
        - task_config['estimation_config']['number_of_rounds']: list of round numbers
    output_dir : str
        Directory to save plots
    timestamp : str, optional
        Timestamp string to include in filenames
    ignore_first_N : int, optional
        Ignore data with N <= ignore_first_N for fitting (default: 7)
    plot_fits : bool, optional
        Whether to compute and plot power-law fits (default: True). When False, M and R
        text labels are shown for each data point

    Returns:
    --------
    None
    """
    import os
    from scipy.optimize import curve_fit

    os.makedirs(output_dir, exist_ok=True)

    import matplotlib as mpl

    BASE_FONT_SIZE = 13  # slightly larger than default

    mpl.rcParams.update({
        'font.size': BASE_FONT_SIZE,
        'axes.titlesize': BASE_FONT_SIZE + 3,
        'axes.labelsize': BASE_FONT_SIZE + 3,
        'xtick.labelsize': BASE_FONT_SIZE,
        'ytick.labelsize': BASE_FONT_SIZE,
        'legend.fontsize': BASE_FONT_SIZE + 0.5,
    })

    # Step 1: Extract data from each result
    all_data = []

    for result in results:
        N = result['N']
        M = result['M']
        lower_running_times = result['lower_running_times']
        upper_running_times = result['upper_running_times']

        # Get number_of_rounds from task_config
        if 'task_config' in result and 'estimation_config' in result['task_config']:
            number_of_rounds = result['task_config']['estimation_config']['number_of_rounds']
            if isinstance(number_of_rounds, int):
                number_of_rounds = [number_of_rounds]
        else:
            # Fallback
            number_of_rounds = list(range(1, len(lower_running_times) + 1))

        # Find the index with the lowest average running time
        avg_running_times = 0.5 * (lower_running_times + upper_running_times)
        best_idx = np.argmin(avg_running_times)

        # Extract the best values
        best_lower_rt = lower_running_times[best_idx]
        best_upper_rt = upper_running_times[best_idx]
        best_R = number_of_rounds[best_idx]

        # Store in dictionary
        data_dict = {
            'N': N,
            'M': M,
            'R': best_R,
            'lower_running_time': best_lower_rt,
            'upper_running_time': best_upper_rt
        }
        all_data.append(data_dict)

    # Step 2: For each N, find the configuration with the lowest average running time
    from collections import defaultdict
    data_by_N = defaultdict(list)

    for data_dict in all_data:
        data_by_N[data_dict['N']].append(data_dict)

    best_data_by_N = []
    for N, data_list in data_by_N.items():
        # Find the entry with the lowest average running time
        best_entry = min(data_list, key=lambda d: 0.5 * (d['lower_running_time'] + d['upper_running_time']))
        best_data_by_N.append(best_entry)

    # Step 3: Sort by N from low to high
    best_data_by_N.sort(key=lambda d: d['N'])

    # Step 4: Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    N_values = [d['N'] for d in best_data_by_N]
    lower_rts = [d['lower_running_time'] for d in best_data_by_N]
    upper_rts = [d['upper_running_time'] for d in best_data_by_N]
    avg_rts = [0.5 * (d['lower_running_time'] + d['upper_running_time']) for d in best_data_by_N]

    # Plot the intervals as error bars
    y_err_lower = np.array(avg_rts) - np.array(lower_rts)
    y_err_upper = np.array(upper_rts) - np.array(avg_rts)

    ax.errorbar(N_values, avg_rts, yerr=[y_err_lower, y_err_upper],
                fmt='none', capsize=5, capthick=2,
                color='steelblue', ecolor='steelblue',
                label='Running time interval')

    # Fit power-law models to the data if requested
    if plot_fits:
        N_fit_mask = np.array(N_values) > ignore_first_N
        N_fit = np.array(N_values)[N_fit_mask]
        avg_rts_fit = np.array(avg_rts)[N_fit_mask]

        if len(N_fit) >= 2:  # Need at least 2 points to fit
            # Define power-law functions
            def power_law_1_3(N, alpha):
                return alpha * N**(1/3)

            def power_law_1_4(N, alpha):
                return alpha * N**(1/4)

            def power_law_beta(N, alpha, beta):
                return alpha * N**beta

            # Fit the models
            try:
                # Fit T_1 = alpha_1 * N^(1/3)
                popt1, _ = curve_fit(power_law_1_3, N_fit, avg_rts_fit, p0=[1.0])
                alpha_1 = popt1[0]
                T_fit_1 = power_law_1_3(np.array(N_values), alpha_1)
                rmse_1 = np.sqrt(np.mean((avg_rts_fit - power_law_1_3(N_fit, alpha_1))**2))

                # Fit T_2 = alpha_2 * N^(1/4)
                popt2, _ = curve_fit(power_law_1_4, N_fit, avg_rts_fit, p0=[1.0])
                alpha_2 = popt2[0]
                T_fit_2 = power_law_1_4(np.array(N_values), alpha_2)
                rmse_2 = np.sqrt(np.mean((avg_rts_fit - power_law_1_4(N_fit, alpha_2))**2))

                # Fit T_3 = alpha_3 * N^beta
                popt3, _ = curve_fit(power_law_beta, N_fit, avg_rts_fit, p0=[1.0, 0.3])
                alpha_3, beta = popt3
                T_fit_3 = power_law_beta(np.array(N_values), alpha_3, beta)
                rmse_3 = np.sqrt(np.mean((avg_rts_fit - power_law_beta(N_fit, alpha_3, beta))**2))

                # Plot the fits
                ax.plot(N_values, T_fit_1, '--', color='red', linewidth=2,
                       label=f'$T_1 = {alpha_1:.3f} N^{{1/3}}$ (RMSE: {rmse_1:.3f})')
                ax.plot(N_values, T_fit_2, '--', color='green', linewidth=2,
                       label=f'$T_2 = {alpha_2:.3f} N^{{1/4}}$ (RMSE: {rmse_2:.3f})')
                ax.plot(N_values, T_fit_3, '--', color='purple', linewidth=2,
                       label=f'$T_3 = {alpha_3:.3f} N^{{{beta:.3f}}}$ (RMSE: {rmse_3:.3f})')

                # Print fit parameters
                print("\nPower-law fit results (using N > {} for fitting):".format(ignore_first_N))
                print(f"  T_1 = {alpha_1:.6f} * N^(1/3),  RMSE = {rmse_1:.6f}")
                print(f"  T_2 = {alpha_2:.6f} * N^(1/4),  RMSE = {rmse_2:.6f}")
                print(f"  T_3 = {alpha_3:.6f} * N^{beta:.6f},  RMSE = {rmse_3:.6f}")

            except Exception as e:
                print(f"Warning: Could not fit power-law models: {e}")

    # Set y-axis to start from 0 and calculate the upper limit
    max_upper_rt = max(upper_rts)
    y_max = max_upper_rt * 1.25  # Add 25% margin at top
    ax.set_ylim(bottom=0, top=y_max)

    # Add text labels only when fits are not shown
    if not plot_fits:
        text_offset = y_max * 0.03  # Fixed offset of 3% of y-axis range

        for i, d in enumerate(best_data_by_N):
            N = d['N']
            M = d['M']
            R = d['R']
            upper_rt = d['upper_running_time']
            lower_rt = d['lower_running_time']

            # Calculate potential text position above the upper error bar
            text_y_above = upper_rt + text_offset

            # Check if text would go outside the plot (approximate height of 2-line text)
            text_height_approx = y_max * 0.08  # Approximate 8% of plot height for 2 lines

            if text_y_above + text_height_approx > y_max:
                # Place text below the error bar
                text_y = lower_rt - text_offset
                va = 'top'
            else:
                # Place text above the error bar
                text_y = text_y_above
                va = 'bottom'

            ax.text(N, text_y, f'M = {M}\nR = {R}',
                    ha='center', va=va,
                    fontsize=BASE_FONT_SIZE - 1,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='gray', alpha=0.8))

    ax.set_xlabel('Number of vertices N')
    ax.set_ylabel('Runtime t')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=BASE_FONT_SIZE - 1)

    plt.tight_layout()

    if timestamp:
        filename = f'fermionic_runtimes_{timestamp}.png'
    else:
        filename = 'fermionic_runtimes.png'

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {filepath}")