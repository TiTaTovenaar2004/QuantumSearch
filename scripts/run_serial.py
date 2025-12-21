#!/usr/bin/env python3
"""Run a single quantum search simulation."""

from quantumsearch.core.bosonic_search import bosonic_search
from quantumsearch.core.fermionic_search import fermionic_search
from quantumsearch.core.graph import Graph
from quantumsearch.plotting import plot_site_occupations


def main():
    # Configure simulation
    graph = Graph(graph_type='line', N=5)
    graph.calculate_hopping_rate()

    simulation = fermionic_search(
        M=3,
        graph=graph,
        output='occupations',
        T=50,
        number_of_time_steps=200,
        simulation_time_adjustment=False
    )

    # Plot results
    plot_site_occupations(simulation, filename='results/plots/site_occupations.png')
    print(f"Plot saved to results/plots/site_occupations.png")


if __name__ == '__main__':
    main()
