import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation
import networkx as nx

from utils import show_superposition, to_fock, distribute,distribute_with_cap

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

# --- --- --- Testing --- --- ---
# N = 2
# M = 1
# r = 2
# marked_vertex = 0

# dim_per_site = M + 1

# op = majority_vote_operator(N, M, r, marked_vertex)

# state = math.sqrt(3)/2 * to_fock([[1,0]], dim_per_site) + 1/2 * to_fock([[0,1]], dim_per_site)
# total_state = tensor(state, state)
# print(expect(op, total_state))
# print(9/16)