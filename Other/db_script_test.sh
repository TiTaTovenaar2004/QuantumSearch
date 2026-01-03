#!/bin/bash

#SBATCH --job-name="Py_test"
#SBATCH --time=00:01:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=10MB
#SBATCH --account=education-as-bsc-tn

module load 2025
module load python
module load py-numpy

srun python delftblue_test.py > test.log