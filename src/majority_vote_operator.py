import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation
import networkx as nx

def majority_vote_operator(N, M, dim_per_site, r):

    # --- Projector onto states with m particles in total on vertex i ---

    # Determining all combinations m_1, ..., m_r such that m1 + ... + mr = m_tot
    def distribute(k, n): # Generates all ways to distribute k indistinguishable items into n distinguishable boxes
        if n == 1:
            yield (k,)
        else:
            for i in range(k + 1):
                for rest in distribute(k - i, n - 1):
                    yield (i,) + rest

    # Construct a projector onto states with m_1 particles on vertex 1, ..., m_r particles on vertex r