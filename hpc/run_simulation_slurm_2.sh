#!/bin/bash

#SBATCH --job-name="QS_2"
#SBATCH --time=05:59:00
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=education-as-bsc-tn

module load 2025
module load openmpi
module load python

cd $SLURM_SUBMIT_DIR
cd QuantumSearch

mkdir -p results/data
mkdir -p results/logs

srun python scripts/run_parallel_simulations_2.py > results/logs/simulation_2_$(date +%Y%m%d_%H%M%S).log 2>&1
