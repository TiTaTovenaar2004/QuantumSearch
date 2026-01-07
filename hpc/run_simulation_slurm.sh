#!/bin/bash

#SBATCH --job-name="Quantum_search"
#SBATCH --time=00:10:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=education-as-bsc-tn

module load 2025
module load openmpi
module load python

cd ~/scratch/tbvangelder/quantumsearch/QuantumSearch

mkdir -p results/data   
mkdir -p results/logs   

srun python scripts/run_parallel_simulations.py > results/logs/simulation_$(date +%Y%m%d_%H%M%S).log 2>&1
