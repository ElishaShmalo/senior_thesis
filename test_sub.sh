#!/bin/bash

#SBATCH --partition=main            # Partition (job queue)

#SBATCH --requeue                   # Return job to the queue if preempted

#SBATCH --job-name=bvar_Nvar_etavar         # Assign a short name to your job

#SBATCH --nodes=2                   # Number of nodes you require

#SBATCH --ntasks=20               # Total # of tasks across all nodes

#SBATCH --cpus-per-task=1           # Cores per task (>1 if multithread tasks)

#SBATCH --mem=12800               # Real memory (RAM) required (MB)

#SBATCH --time=00:05:00             # Total run time limit (HH:MM:SS)

#SBATCH --output=slurm.%N.%j.out    # STDOUT output file

#SBATCH --error=slurm.%N.%j.err     # STDERR output file (optional)

module load openmpi

export OMP_NUM_THREADS=1

srun ~/.juliaup/bin/julia -p $((SLURM_NTASKS - 1)) julia_code/lyapunov_exponents/get_good_data_severalL.jl


