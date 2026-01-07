#!/bin/bash

#SBATCH --job-name="Quantum_search"
#SBATCH --time=00:10:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=education-as-bsc-tn

module load 2025
module load python
module load py-numpy

srun python delftblue_test.py > test.log