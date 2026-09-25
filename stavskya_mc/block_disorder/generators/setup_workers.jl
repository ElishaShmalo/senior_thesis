# Adds worker processes for the block_disorder generators.
#
#   * On the cluster (default), workers come from SlurmClusterManager, exactly as
#     in the older stavskya_mc/get_*.jl scripts.
#   * With the environment variable STAV_LOCAL_TEST=1 the script runs in local
#     smoke-test mode: 2 local workers (or STAV_LOCAL_WORKERS), and every generator
#     switches to tiny system sizes and writes to block_disorder/_local_test_output/
#     (git-ignored, overwritten on every run). tests/check_local_test_output.py then
#     checks those files. Use this to check a generator before submitting it:
#
#         STAV_LOCAL_TEST=1 julia stavskya_mc/block_disorder/generators/<script>.jl
#
# After this file runs, the constant LOCAL_TEST is defined on the master and on
# every worker.

using Distributed

const LOCAL_TEST = get(ENV, "STAV_LOCAL_TEST", "0") == "1"

if LOCAL_TEST
    n_local = parse(Int, get(ENV, "STAV_LOCAL_WORKERS", "2"))
    nprocs() == 1 && addprocs(n_local)
    println("LOCAL TEST MODE: $(nworkers()) local workers")
else
    @eval using SlurmClusterManager
    Base.invokelatest() do
        println("We are adding $(SlurmManager()) workers")
        addprocs(SlurmManager(launch_timeout=600.0))   # 2026-09: default 60 s timed out while 250 workers started
    end
end

@everywhere workers() const LOCAL_TEST = $LOCAL_TEST
