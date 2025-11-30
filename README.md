# QuantumSearch
Simulation of multi-boson and multi-fermion quantum search over arbitrary graphs containing a marked vertex. Performance analysis of each search algorithm (boson versus fermion).

# Manual
## bosonic_search and fermionic_search
Outputs:
- result - the results of the simulation
- times - list of the times for which the simulation is run
- G - the graph in the simulation (a networkx graph)
- params - parameters used in the simulation

Attributes:
- output = 'states'
result.states gives a list of the quantum states over time
- output = 'occupations'
result.expect gives a matrix where each row is the expectation value of some observable over time
- output = 'success probabilities'
result gives a matrix where row i is the success probability after a majority vote of R[i] rounds over time

# To-do
- Make 'output' attribute of bosonic/fermionic_search more logical. Output format of 'states', 'occupations' and 'success probabilities' should be intuitive. Use of these data in plot functions should be intuitive.
- Format plots so that they are uniform, have correct axis and units, and have no title.