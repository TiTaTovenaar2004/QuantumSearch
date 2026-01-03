"""Script to analyze fermionic search running times.

This script loads simulation results and analyzes the running times as a function
of N (number of vertices) and M (number of fermions).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path to import quantumsearch modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantumsearch.parallel.mpi_runner import load_results


def analyze_runtimes(results):
    """
    Extract and analyze running times from results.

    Returns:
    --------
    runtime_data : dict
        Dictionary with structure: {N: {M: [(rounds_idx, lower_rt, upper_rt), ...]}}
    """
    runtime_data = {}

    for result in results:
        N = result['N']
        M = result['M']

        if N not in runtime_data:
            runtime_data[N] = {}
        if M not in runtime_data[N]:
            runtime_data[N][M] = []

        # Extract running times from arrays
        if 'lower_running_times' in result and 'upper_running_times' in result:
            lower_rts = result['lower_running_times']
            upper_rts = result['upper_running_times']
            
            # Iterate through each round index
            for idx in range(len(lower_rts)):
                lower_rt = lower_rts[idx]
                upper_rt = upper_rts[idx]
                
                # Only include valid running times (not infinite)
                if not np.isinf(lower_rt) and not np.isinf(upper_rt):
                    runtime_data[N][M].append((idx, lower_rt, upper_rt))

    return runtime_data


def find_best_M_per_N(runtime_data):
    """
    For each N, find the M with the lowest average running time.

    Returns:
    --------
    best_M_data : dict
        Dictionary with structure: {N: (M, avg_lower_rt, avg_upper_rt)}
    """
    best_M_data = {}

    for N in sorted(runtime_data.keys()):
        best_M = None
        best_avg_rt = np.inf
        best_lower = None
        best_upper = None

        for M, rt_list in runtime_data[N].items():
            if len(rt_list) == 0:
                continue

            # Calculate average lower and upper running times across all rounds
            avg_lower = np.mean([rt[1] for rt in rt_list])
            avg_upper = np.mean([rt[2] for rt in rt_list])
            avg_rt = (avg_lower + avg_upper) / 2

            if avg_rt < best_avg_rt:
                best_avg_rt = avg_rt
                best_M = M
                best_lower = avg_lower
                best_upper = avg_upper

        if best_M is not None:
            best_M_data[N] = (best_M, best_lower, best_upper)

    return best_M_data


def plot_runtime_vs_N(best_M_data, output_dir='results/plots'):
    """
    Plot running time vs N (for optimal M at each N).
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    N_values = sorted(best_M_data.keys())
    M_values = [best_M_data[N][0] for N in N_values]
    lower_rts = [best_M_data[N][1] for N in N_values]
    upper_rts = [best_M_data[N][2] for N in N_values]

    # Calculate center and error
    centers = [(lower + upper) / 2 for lower, upper in zip(lower_rts, upper_rts)]
    errors_lower = [center - lower for center, lower in zip(centers, lower_rts)]
    errors_upper = [upper - center for center, upper in zip(centers, upper_rts)]

    plt.figure(figsize=(10, 6))
    plt.errorbar(N_values, centers, yerr=[errors_lower, errors_upper],
                 fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)

    plt.xlabel('Number of vertices (N)', fontsize=12)
    plt.ylabel('Running time', fontsize=12)
    plt.title('Optimal Running Time vs N\n(M chosen to minimize average runtime)',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add M values as text annotations
    for N, M, center in zip(N_values, M_values, centers):
        plt.text(N, center, f'  M={M}', fontsize=9, va='center')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'runtime_vs_N_optimal_M.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved: {filepath}")


def plot_runtime_vs_M_for_each_N(runtime_data, output_dir='results/plots'):
    """
    For each N, plot running time vs M.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    N_values = sorted(runtime_data.keys())
    n_plots = len(N_values)

    # Determine grid layout
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), squeeze=False)

    for idx, N in enumerate(N_values):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        M_values_sorted = sorted(runtime_data[N].keys())

        # Calculate average lower and upper running times for each M
        M_values = []
        centers = []
        errors_lower = []
        errors_upper = []

        for M in M_values_sorted:
            rt_list = runtime_data[N][M]
            if len(rt_list) == 0:
                continue

            avg_lower = np.mean([rt[1] for rt in rt_list])
            avg_upper = np.mean([rt[2] for rt in rt_list])
            center = (avg_lower + avg_upper) / 2

            M_values.append(M)
            centers.append(center)
            errors_lower.append(center - avg_lower)
            errors_upper.append(avg_upper - center)

        if len(centers) > 0:
            ax.errorbar(M_values, centers, yerr=[errors_lower, errors_upper],
                       fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)

        ax.set_xlabel('Number of fermions (M)', fontsize=10)
        ax.set_ylabel('Running time', fontsize=10)
        ax.set_title(f'Running Time vs M for N={N}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(M_values)

    # Hide unused subplots
    for idx in range(n_plots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'runtime_vs_M_for_each_N.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved: {filepath}")


def main():
    """Main function to analyze fermionic search running times."""

    # Set data directory
    data_dir = 'results/data'

    # Load results
    print("Loading simulation results...")
    results, summary = load_results(input_dir=data_dir)
    print(f"Successfully loaded {len(results)} simulation results from {summary['timestamp']}\n")

    # Analyze running times
    print("Analyzing running times...")
    runtime_data = analyze_runtimes(results)

    if len(runtime_data) == 0:
        print("No valid running time data found (all running times are infinite).")
        return

    # Find best M for each N
    best_M_data = find_best_M_per_N(runtime_data)

    print("\nOptimal M for each N:")
    print("-" * 50)
    for N in sorted(best_M_data.keys()):
        M, lower_rt, upper_rt = best_M_data[N]
        avg_rt = (lower_rt + upper_rt) / 2
        print(f"N={N}: M={M}, Running time ∈ [{lower_rt:.4f}, {upper_rt:.4f}], avg={avg_rt:.4f}")

    # Generate plots
    print("\nGenerating plots...")
    plot_runtime_vs_N(best_M_data)
    plot_runtime_vs_M_for_each_N(runtime_data)

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
