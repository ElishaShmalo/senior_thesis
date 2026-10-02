#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=launch_test
#SBATCH --nodes=4
#SBATCH --ntasks=250
#SBATCH --cpus-per-task=1
#SBATCH --mem=125000
#SBATCH --time=00:40:00
#SBATCH --output=launch_test-%j.out
#SBATCH --error=launch_test-%j.err

# Why does addprocs(SlurmManager(...)) time out? (2026-10-01)
# Same resources as submit_severalL_data_job*.sh. Three stages, each timed:
#   A. plain Julia started on every task by srun (no Distributed): do all 250 start, and how fast?
#   B. `julia --worker` on every task, exactly what SlurmClusterManager launches: how many print their
#      "julia_worker:port#ip" line, and when? (the line the master waits for)
#   C. the real addprocs(SlurmManager) with a 15-minute launch timeout.
# Run from the repository root:  sbatch heisen_spin_chain/tests/cluster_launch_test.sh
# Everything ends up in launch_test-<jobid>.out (summary) and .err (any worker error messages).

module load openmpi
export OMP_NUM_THREADS=1
JULIA=~/julia-1.11.6/bin/julia
echo "job $SLURM_JOB_ID | nodes $SLURM_JOB_NODELIST | $SLURM_NTASKS tasks | $(date)"
echo "julia: $($JULIA --version)"
echo "~/.julia/config (a startup.jl here also runs in every worker unless --startup-file=no):"
ls -la ~/.julia/config 2>&1 | sed 's/^/   /'

echo; echo "== A. plain Julia on every task (timeout 10 min)"
export T0=$(date +%s.%N)
timeout 600 srun $JULIA --startup-file=no -e 'println(gethostname(), " ", round(time() - parse(Float64, ENV["T0"]); digits=1))' > launch_A_$SLURM_JOB_ID.txt
echo "srun exit code: $?   (124 = hit the 10-minute timeout)"
echo "tasks that started: $(wc -l < launch_A_$SLURM_JOB_ID.txt) of $SLURM_NTASKS"
echo "per node:"; awk '{print $1}' launch_A_$SLURM_JOB_ID.txt | sort | uniq -c | sed 's/^/   /'
echo "slowest start-ups (host, seconds):"; sort -k2 -n launch_A_$SLURM_JOB_ID.txt | tail -3 | sed 's/^/   /'

echo; echo "== B. julia --worker on every task (what SlurmClusterManager runs; timeout 5 min)"
START=$(date +%s)
timeout 300 srun $JULIA --startup-file=no --worker=0123456789abcdef 2>/dev/null \
  | while IFS= read -r line; do echo "$(( $(date +%s) - START )) $line"; done > launch_B_$SLURM_JOB_ID.txt
echo "worker lines received: $(grep -c julia_worker launch_B_$SLURM_JOB_ID.txt) of $SLURM_NTASKS"
echo "seconds until the last one: $(grep julia_worker launch_B_$SLURM_JOB_ID.txt | tail -1 | awk '{print $1}')"
echo "IPs workers advertise (the master must reach these):"; grep -o '#[0-9.]*' launch_B_$SLURM_JOB_ID.txt | sort | uniq -c | sed 's/^/   /'

echo; echo "== C. real addprocs(SlurmManager), launch timeout 15 min"
$JULIA --startup-file=no heisen_spin_chain/tests/cluster_launch_test.jl
echo "julia exit code: $?"
echo; echo "done $(date)"
rm -f launch_A_$SLURM_JOB_ID.txt launch_B_$SLURM_JOB_ID.txt
