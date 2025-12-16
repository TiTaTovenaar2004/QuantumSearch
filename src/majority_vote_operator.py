from qutip import *

from utils import distribute_with_cap

def majority_vote_operator(N, M, r, marked_vertex, dim_per_site):

    # --- Construct a projector onto states with m_1 particles on vertex i (1), ..., m_r particles on vertex i (r) ---
    def vertex_projector_specific(vertex_i, particles): # particles: tuple containing (m_1, ..., m_r)
        ops = []
        for j in range(r):
            for i in range(N):
                if i == vertex_i:
                    ops.append(basis(dim_per_site, particles[j]) * basis(dim_per_site, particles[j]).dag())
                else:
                    ops.append(qeye(dim_per_site))
        
        return tensor(ops)

    # --- Construct a projector onto states with m_total particles in total on vertex i ---
    def vertex_projector_total(vertex_i, m_total):
        partitions = list(distribute_with_cap(m_total, r, dim_per_site))
        projector = 0
        for partition in partitions:
            projector += vertex_projector_specific(vertex_i, partition)
        
        return projector

    # --- Projector onto states where the number of particles on vertex i is less than that on vertex w ---
    def w_larger_projector(vertex_i): # particles on w > particles on vertex
        projector = 0
        for m_total_w in range(r*(dim_per_site - 1) + 1):
            for m_total_i in range(m_total_w):
                projector += vertex_projector_total(marked_vertex, m_total_w) * vertex_projector_total(vertex_i, m_total_i)
        
        return projector

    # --- Projector onto states where w has the most particles ---
    def w_largest_projector():
        projector = 1
        for vertex in range(N):
            if vertex != marked_vertex:
                projector = projector * w_larger_projector(vertex)
        
        return projector

    return w_largest_projector()
