# QuantumSearch
Simulation of multi-boson and multi-fermion quantum search over arbitrary graphs containing a marked vertex. Performance analysis of each search algorithm (boson versus fermion).

# Plan DelftBlue
Assuming optimal hopping rate can be determined.

## Run 1
- Simulate fermionic search for different N, M, T and R (determine running times and plot success probabilities), in order to find how large an N, M and R the computer can handle, how large T has to be in order for the simulation to contain multiple peaks in success probability, and how the running time changes for these larger graphs as R grows (Is the running time infty for threshold>0.8? Then it is not insightful to set the threshold in our final simulation to > 0.8. Does the running time increase monotonically with R? This influences which stop_condition is best).

## Run 2
- Plan: Use the best T, threshold(s) and stop condition found in Run 1. Simulate fermionic search on a complete graph for all N = 1, 2, ..., as large as possible, optimize the lowest running time over M for each N.
- Output: For each N, the running time of the fermionic search on the complete graph (minimized over M). Compare with Karthigeyan2025 (N^1/3 ?)
- Output: For some values of N, a plot of the lowest running time as a function of M. Compare with Karthigeyan2025 (M goes as sqrt(N)?)

## Run 3
- Plan: Use same T, threshold(s) and stop condition. Simulate fermionic and bosonic search on a lot of erdos-renyi graphs (for different p's) and for each graph optimize the lowest running times over M (fermionic and bosonic separately of course).
- Output: Plot the optimal lowest running times for both the fermionic and the bosonic search as a function of p. Does the fermionic search have an advantage for low p?