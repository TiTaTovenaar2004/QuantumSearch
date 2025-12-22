import numpy as np
import networkx as nx
import time

from quantumsearch.core.utils import orthonormalize_eigenvectors

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