#!/bin/bash

#SBATCH --partition=main            # Partition (job queue)

#SBATCH --requeue                   # Return job to the queue if preempted

#SBATCH --job-name=bvar_Nvar_etavar         # Assign a short name to your job

#SBATCH --nodes=8                  # Number of nodes you require

#SBATCH --ntasks=512                 # Total # of tasks across all nodes

#SBATCH --cpus-per-task=1           # Cores per task (>1 if multithread tasks)

#SBATCH --mem=512000               # Real memory (RAM) required (MB)

#SBATCH --time=48:00:00             # Total run time limit (HH:MM:SS)

#SBATCH --output=slurm.%N.%j.out    # STDOUT output file

#SBATCH --error=slurm.%N.%j.err     # STDERR output file (optional)

module load openmpi

export OMP_NUM_THREADS=1

 ~/.juliaup/bin/julia julia_code/lyapunov_exponents/get_good_data_severalL.jl

