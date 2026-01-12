import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
from matplotlib.cm import viridis
import time

def orthonormalize_vectors(vecs):
    M = np.stack(vecs, axis=1)
    Q, R = np.linalg.qr(M, mode='reduced')
    return Q.T
def orthonormalize_eigenvectors(eigenvalues, eigenvectors):
    new_eigenvectors = np.copy(eigenvectors)
    eigenvalues_shifted = np.roll(eigenvalues, -1)
    degeneracy = (eigenvalues == eigenvalues_shifted).astype(int) # element i is 1 if eigenvalue i is degenerate with eigenvalue i+1
    current_idx = 0
    while current_idx < len(degeneracy):
        if degeneracy[current_idx] == 1:
            start_idx = current_idx
            while current_idx < len(degeneracy) and degeneracy[current_idx] == 1:
                current_idx += 1
            end_idx = current_idx + 1 # +1 to include the last degenerate eigenvalue
            degenerate_vecs = [eigenvectors[:, i] for i in range(start_idx, end_idx)]
            orthonormal_vecs = orthonormalize_vectors(degenerate_vecs)
            for i, vec in enumerate(orthonormal_vecs):
                new_eigenvectors[:, start_idx + i] = vec
        else:
            current_idx += 1

    return new_eigenvectors
class Graph:
    def __init__(self, graph_type, N, p=0.5, m=2, marked_vertex=0):
        self.graph_type = graph_type
        self.N = N
        self.p = p
        self.m = m
        self.marked_vertex = marked_vertex

        self.eigenvalues = None
        self.hopping_rate_calculation_time = None

        # Create graph
        if graph_type == 'complete':
            self.graph = nx.complete_graph(N)
        elif graph_type == 'cycle':
            self.graph = nx.cycle_graph(N)
        elif graph_type == 'line':
            self.graph = nx.path_graph(N)
        elif graph_type == 'erdos-renyi':
            self.graph = nx.erdos_renyi_graph(N, p)
            while not nx.is_connected(self.graph):
                self.graph = nx.erdos_renyi_graph(N, p)
        elif graph_type == 'barabasi-albert':
            self.graph = nx.barabasi_albert_graph(N, m)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

        self.adjacency = nx.to_numpy_array(self.graph)

    def calculate_c(self):
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(self.adjacency)
        idx = eigenvalues.argsort()
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx] # Columns are eigenvectors
        self.spectral_radius = np.max([abs(self.eigenvalues[0]), self.eigenvalues[-1]])

        # Orthogonalize eigenvectors of degenerate eigenvalues
        self.eigenvectors = orthonormalize_eigenvectors(self.eigenvalues, self.eigenvectors)

        # Calculate normalized eigenvalues
        if self.spectral_radius == 0:
            raise ValueError("The adjacency matrix cannot be normalized if its spectral radius is zero.")
        else:
            self.eigenvalues_normalized = 0.5*((self.eigenvalues / self.spectral_radius) + 1)
            self.eigenvectors_normalized = self.eigenvectors

        # Calculate spectral gap and spectral moments (of the normalized eigenvalues)
        if len(self.eigenvalues) < 2:
            raise ValueError("S_k cannot be calculated for an adjacency matrix with less than two eigenvalues.")
        else:
            self.spectral_gap = self.eigenvalues_normalized[-1] - self.eigenvalues_normalized[-2]
            marked_vertex_ket = np.eye(self.N)[self.marked_vertex]
            self.marked_vertex_projection = [np.vdot(self.eigenvectors_normalized[:, i], marked_vertex_ket) for i in range(self.N)] # [a_1, ..., a_N]
            self.sqrt_epsilon = abs(np.vdot(np.eye(self.N)[self.marked_vertex], self.eigenvectors_normalized[:, -1]))
            self.S1 = np.sum(np.abs(self.marked_vertex_projection[0:-1])**2 / (1 - self.eigenvalues_normalized[0:-1]))
            self.S2 = np.sum(np.abs(self.marked_vertex_projection[0:-1])**2 / (1 - self.eigenvalues_normalized[0:-1])**2)
            self.S3 = np.sum(np.abs(self.marked_vertex_projection[0:-1])**2 / (1 - self.eigenvalues_normalized[0:-1])**3)
            self.hopping_rate = self.S1 / (2*self.spectral_radius)
            self.c = self.sqrt_epsilon / np.min([(self.S1 * self.S2)/self.S3, self.spectral_gap*np.sqrt(self.S2)])

    def calculate_hopping_rate(self):
        start_time = time.time()

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(self.adjacency)
        idx = eigenvalues.argsort()
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx] # Columns are eigenvectors
        self.spectral_radius = np.max([abs(self.eigenvalues[0]), self.eigenvalues[-1]])

        # Orthogonalize eigenvectors of degenerate eigenvalues
        self.eigenvectors = orthonormalize_eigenvectors(self.eigenvalues, self.eigenvectors)

        # Calculate normalized eigenvalues
        if self.spectral_radius == 0:
            raise ValueError("The adjacency matrix cannot be normalized if its spectral radius is zero.")
        else:
            self.eigenvalues_normalized = 0.5*((self.eigenvalues / self.spectral_radius) + 1)
            self.eigenvectors_normalized = self.eigenvectors

        # Calculate spectral gap and spectral moments (of the normalized eigenvalues)
        if len(self.eigenvalues) < 2:
            raise ValueError("S_k cannot be calculated for an adjacency matrix with less than two eigenvalues.")
        else:
            marked_vertex_ket = np.eye(self.N)[self.marked_vertex]
            self.marked_vertex_projection = [np.vdot(self.eigenvectors_normalized[:, i], marked_vertex_ket) for i in range(self.N)] # [a_1, ..., a_N]
            self.S1 = np.sum(np.abs(self.marked_vertex_projection[0:-1])**2 / (1 - self.eigenvalues_normalized[0:-1]))
            self.hopping_rate = self.S1 / (2*self.spectral_radius)

        end_time = time.time()
        self.hopping_rate_calculation_time = end_time - start_time

    def summary(self):
        """Return a formatted summary of the graph."""
        lines = []
        lines.append("="*60)
        lines.append("GRAPH SUMMARY")
        lines.append("="*60)

        # Basic properties
        lines.append(f"Type: {self.graph_type}")
        lines.append(f"Vertices (N): {self.N}")
        lines.append(f"Marked vertex: {self.marked_vertex}")

        # Graph-specific parameters
        if self.graph_type == 'erdos-renyi':
            lines.append(f"Connection probability (p): {self.p}")
        elif self.graph_type == 'barabasi-albert':
            lines.append(f"Edges to attach (m): {self.m}")

        # Network properties
        lines.append(f"\nNetwork properties:")
        lines.append(f"  Edges: {self.graph.number_of_edges()}")
        lines.append(f"  Average degree: {2 * self.graph.number_of_edges() / self.N:.2f}")

        # Spectral properties
        if self.eigenvalues is not None:
            lines.append(f"\nSpectral properties:")
            lines.append(f"  Spectral radius: {self.spectral_radius:.4f}")
            lines.append(f"  Min eigenvalue: {self.eigenvalues[0]:.4f}")
            lines.append(f"  Max eigenvalue: {self.eigenvalues[-1]:.4f}")
            if hasattr(self, 'hopping_rate') and self.hopping_rate is not None:
                lines.append(f"  Hopping rate (γ): {self.hopping_rate:.4f}")
            if hasattr(self, 'S1') and self.S1 is not None:
                lines.append(f"  S₁: {self.S1:.4f}")

        # Computation time
        if self.hopping_rate_calculation_time is not None:
            lines.append(f"\nHopping rate calculation time: {self.hopping_rate_calculation_time:.4f}s")

        lines.append("="*60)
        return "\n".join(lines)

graphs_complete = [
    Graph(graph_type='complete', N=N) for N in range(3, 31)
]
graphs_cycle = [
    Graph(graph_type='cycle', N=N) for N in range(3, 31)
]
graphs_line = [
    Graph(graph_type='line', N=N) for N in range(3, 31)
]

graphs_erdos_renyi = [
    [Graph(graph_type='erdos-renyi', N=N, p=0.5) for _ in range(1000)] for N in range(3, 31)
]
graphs_barabasi_albert = [
    [Graph(graph_type='barabasi-albert', N=N, m=2) for _ in range(1000)] for N in range(3, 31)
]

for graph in graphs_complete + graphs_cycle + graphs_line:
    graph.calculate_c()

for graph_list in graphs_erdos_renyi + graphs_barabasi_albert:
    for graph in graph_list:
        graph.calculate_c()

def plot_c_vs_N(
    graphs_complete,
    graphs_cycle,
    graphs_line,
    graphs_erdos_renyi,
    graphs_barabasi_albert,
    central_percentage=90,
    output_path="/home/titatovenaar/QuantumSearch/Other/c_vs_N_graph_types.png",
    dpi=300
):
    """
    Plot c as a function of N for several graph types and save the figure to disk.

    Parameters
    ----------
    graphs_complete : list
        List of Graph objects of type 'complete'
    graphs_cycle : list
        List of Graph objects of type 'cycle'
    graphs_line : list
        List of Graph objects of type 'line'
    graphs_erdos_renyi : list of lists
        For each N, a list of Graph objects (random realizations)
    graphs_barabasi_albert : list of lists
        For each N, a list of Graph objects (random realizations)
    central_percentage : float
        Percentage of the distribution to keep (e.g. 90 keeps the central 90%)
    output_path : str
        File path where the plot will be saved
    dpi : int
        Resolution of the saved figure
    """

    # Sanity check
    if not (0 < central_percentage <= 100):
        raise ValueError("central_percentage must be in (0, 100].")

    lower_q = (100 - central_percentage) / 2
    upper_q = 100 - lower_q

    # Extract N and c values (deterministic graphs)
    N_complete = [graph.N for graph in graphs_complete]
    N_cycle = [graph.N for graph in graphs_cycle]
    N_line = [graph.N for graph in graphs_line]

    c_complete = [graph.c for graph in graphs_complete]
    c_cycle = [graph.c for graph in graphs_cycle]
    c_line = [graph.c for graph in graphs_line]

    # Extract N and percentile envelopes (random graphs)
    N_random = [graph_list[0].N for graph_list in graphs_erdos_renyi]

    c_erdos_vals = [[g.c for g in graph_list] for graph_list in graphs_erdos_renyi]
    c_barabasi_vals = [[g.c for g in graph_list] for graph_list in graphs_barabasi_albert]

    c_erdos_min = [np.percentile(vals, lower_q) for vals in c_erdos_vals]
    c_erdos_max = [np.percentile(vals, upper_q) for vals in c_erdos_vals]

    c_barabasi_min = [np.percentile(vals, lower_q) for vals in c_barabasi_vals]
    c_barabasi_max = [np.percentile(vals, upper_q) for vals in c_barabasi_vals]

    # Choose five well-separated colors from viridis
    colors = viridis(np.linspace(0.15, 0.85, 5))

    plt.figure(figsize=(8, 6))

    # Line plots (deterministic)
    plt.plot(N_complete, c_complete, marker='o', linestyle='-',
             color=colors[0], label='Complete graph')

    plt.plot(N_cycle, c_cycle, marker='s', linestyle='-',
             color=colors[1], label='Cycle graph')

    plt.plot(N_line, c_line, marker='^', linestyle='-',
             color=colors[2], label='Line graph')

    # Filled percentile bands (random graphs)
    plt.fill_between(
        N_random, c_erdos_min, c_erdos_max,
        color=colors[3], alpha=0.3,
        label=f'Erdos–Renyi (central {central_percentage}%)'
    )

    plt.fill_between(
        N_random, c_barabasi_min, c_barabasi_max,
        color=colors[4], alpha=0.3,
        label=f'Barabasi–Albert (central {central_percentage}%)'
    )

    plt.xlabel('N')
    plt.ylabel('c')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

plot_c_vs_N(
    graphs_complete,
    graphs_cycle,
    graphs_line,
    graphs_erdos_renyi,
    graphs_barabasi_albert,
    central_percentage=60,
    output_path="/home/titatovenaar/QuantumSearch/Other/c_vs_N_graph_types.png",
    dpi=300
)