#!/bin/bash

#SBATCH --partition=main         # Partition (job queue)

#SBATCH --requeue                   # Return job to the queue if preempted

#SBATCH --job-name=sp_bvh_b1         # Assign a short name to your job

#SBATCH --nodes=8                  # Number of nodes you require

#SBATCH --ntasks=500                 # Total # of tasks across all nodes

#SBATCH --cpus-per-task=1           # Cores per task (>1 if multithread tasks)

#SBATCH --mem=250000               # Real memory (RAM) required (MB)

#SBATCH --time=24:00:00             # Total run time limit (HH:MM:SS)

#SBATCH --output=slurm.%N.%j.out    # STDOUT output file

#SBATCH --error=slurm.%N.%j.err     # STDERR output file (optional)

# Submit from stavskya_mc/:  sbatch block_disorder/submit/submit_upper_lower_spreading_bvh_b1.sh
# (also works from this folder; Slurm logs land in the folder you submit from).
# The generator writes to stavskya_mc/data/spreading/... no matter where it is launched from.

NAME=get_upper_lower_binary_spreading_bvh_b1.jl
GEN=""
for d in block_disorder/generators ../generators stavskya_mc/block_disorder/generators; do
    if [ -f "$d/$NAME" ]; then GEN="$d/$NAME"; break; fi
done
if [ -z "$GEN" ]; then
    echo "Cannot find $NAME - submit from stavskya_mc/:  sbatch block_disorder/submit/submit_upper_lower_spreading_bvh_b1.sh" >&2
    exit 1
fi

module load openmpi

export OMP_NUM_THREADS=1

 ~/julia-1.11.6/bin/julia "$GEN"
