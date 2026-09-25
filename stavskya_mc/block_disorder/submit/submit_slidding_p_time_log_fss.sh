#!/bin/bash

#SBATCH --partition=main         # Partition (job queue)

#SBATCH --requeue                   # Return job to the queue if preempted

#SBATCH --job-name=sp_tlog_fss         # Assign a short name to your job

#SBATCH --nodes=8                  # Number of nodes you require

#SBATCH --ntasks=500                 # Total # of tasks across all nodes

#SBATCH --cpus-per-task=1           # Cores per task (>1 if multithread tasks)

#SBATCH --mem=250000               # Real memory (RAM) required (MB)

#SBATCH --time=24:00:00             # Total run time limit (HH:MM:SS)

#SBATCH --output=slurm.%N.%j.out    # STDOUT output file

#SBATCH --error=slurm.%N.%j.err     # STDERR output file (optional)

# Submit from this directory:  cd stavskya_mc/block_disorder/submit && sbatch submit_slidding_p_time_log_fss.sh
# (same idea as the old stavskya_mc/submit_*.sh, which were submitted from stavskya_mc/).
# The generator writes to stavskya_mc/data/time_log/... no matter where it is launched from.

GEN=../generators/get_slidding_p_time_log_fss.jl
if [ ! -f "$GEN" ]; then
    echo "Cannot find $GEN - submit this script from stavskya_mc/block_disorder/submit" >&2
    exit 1
fi

module load openmpi

export OMP_NUM_THREADS=1

 ~/julia-1.11.6/bin/julia "$GEN"
