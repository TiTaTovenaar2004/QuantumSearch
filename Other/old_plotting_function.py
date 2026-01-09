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

    # Split results into rows with max 3 columns
    # Each group (search_type, graph_type, N) can span multiple rows if it has more than 3 M values
    plot_rows = []
    for group_key in ordered_groups:
        results_for_group = results_by_group[group_key]
        # Chunk into groups of 3
        for i in range(0, len(results_for_group), 3):
            plot_rows.append(results_for_group[i:i+3])

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

            # Extract p parameter if available
            p = None
            if 'task_config' in result and 'graph_config' in result['task_config']:
                p = result['task_config']['graph_config'].get('p')

            if p is not None and p > 0:
                title = f'{search_type.capitalize()} search (M={M}) on the {graph_type} graph (N={N_val}, p={p})'
            else:
                title = f'{search_type.capitalize()} search (M={M}) on the {graph_type} graph (N={N_val})'

            ax.set_title(title, fontsize=11, fontweight='bold')
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