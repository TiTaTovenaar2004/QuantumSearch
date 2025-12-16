import numpy as np
import math
from matplotlib import pyplot as plt
from qutip import *
from matplotlib.animation import FuncAnimation
import networkx as nx
import time

from bosonic_search import bosonic_search
from fermionic_search import fermionic_search
from graph import Graph

import numpy as np
import matplotlib.pyplot as plt

c_complete = []

for i in range(16, 33, 1):
    g = Graph('complete', i)
    g.calculate_eig()
    c_complete.append(g.c)
    print(f"Done with N={i}")

plt.plot(range(16, 33, 1), c_complete, marker='o')
plt.xlabel('Number of vertices N')
plt.ylabel('c')
plt.grid()
plt.show()