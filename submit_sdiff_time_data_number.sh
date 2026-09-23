#!/bin/bash

#SBATCH --partition=main

#SBATCH --requeue

#SBATCH --job-name=bvar_Nvar_etavar

#SBATCH --nodes=4

#SBATCH --ntasks=250

#SBATCH --cpus-per-task=1

#SBATCH --mem=250000

#SBATCH --time=24:00:00
#SBATCH --output=slurm.%N.%j.out
#SBATCH --error=slurm.%N.%j.err

module load openmpi

export OMP_NUM_THREADS=1

# Prompt for a digit 0-9
read -p "Enter a digit (0-9): " digit

# Check that the input is exactly one digit
if [[ ! "$digit" =~ ^[0-9]$ ]]; then
    echo "Error: You must enter exactly one digit from 0 to 9."
    exit 1
fi

# Run the corresponding Julia script
~/julia-1.11.6/bin/julia \
    heisen_spin_chain/lyapunov_exponents/get_sdiff_data_severalL_${digit}.jl