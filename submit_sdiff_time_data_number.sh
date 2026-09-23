#!/bin/bash

#SBATCH --partition=main

#SBATCH --requeue

#SBATCH --job-name=bvar_Nvar_etavar

#SBATCH --nodes=4

#SBATCH --ntasks=250

#SBATCH --cpus-per-task=1

#SBATCH --mem=250000

#SBATCH --time=25:00:00

#SBATCH --output=slurm.%N.%j.out

#SBATCH --error=slurm.%N.%j.err

module load openmpi

export OMP_NUM_THREADS=1

# Get digit from first command-line argument
digit=$1

# Check that an argument was supplied and that it is a single digit 0-9
if [[ ! "$digit" =~ ^[0-9]$ ]]; then
    echo "Error: Please supply a single digit from 0 to 9."
    echo "Usage: sbatch submit_sdiff_time_data_number.sh <digit>"
    echo "Example: sbatch submit_sdiff_time_data_number.sh 0"
    exit 1
fi

echo "Running Julia script number $digit"

~/julia-1.11.6/bin/julia \
    heisen_spin_chain/lyapunov_exponents/get_sdiff_data_severalL_${digit}.jl