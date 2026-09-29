# Spreading runs for the upper/lower binary time-random Stavskaya model (added 2026-09-29):
# every run starts from ONE active site on an otherwise healthy chain, as in BVH
# (Barghathi, Vojta, Hoyos, arXiv:1603.08075, Sec. IV: spreading simulations).
#
# Knobs mean the same as in get_upper_lower_binary_time_log.jl. Differences:
#   * no L: time_random_spreading (utils/dynamics.jl) simulates the infinite chain exactly
#     up to t_max (only the active window is updated);
#   * each run has its own disorder sequence (one run per disorder realization);
#   * runs are grouped in chunks of runs_per_chunk; each chunk writes ONE csv with sums
#     over its runs at every output time (columns time, runs, surv, sum_n, sum_n2,
#     sum_x2, sum_r2, sum_r2sq), so 10^5 runs make a few hundred files, not 10^5;
#   * all (block_len, parameter set, chunk) jobs share one pmap queue, so every worker
#     stays busy.
#
# This copy: BVH-like strong disorder, block_len = 1 (same parameters as
# get_upper_lower_binary_time_log_bvh_b1.jl).
#
# Output: stavskya_mc/data/spreading/time_rand_window_binary/spreading/tmax<T>/epsilonu<u>/epsilonl<l>/pval<p>/blocklen<b>/
#             spreading_tmax<T>_epsilonu<u>_epsilonl<l>_pval<p>_blocklen<b>_ppd<ppd>_runs<R>_chunk<k>.csv
#
# Cluster:          cd stavskya_mc/block_disorder/submit && sbatch submit_upper_lower_spreading_bvh_b1.sh
# Local smoke test: STAV_LOCAL_TEST=1 julia stavskya_mc/block_disorder/generators/get_upper_lower_binary_spreading_bvh_b1.jl
# Analysis:         analysis/analyze_spreading.ipynb

const GEN_DIR = @__DIR__
include(joinpath(GEN_DIR, "setup_workers.jl"))

@everywhere begin
    using Random, Statistics, CSV, DataFrames
    include($(joinpath(GEN_DIR, "..", "..", "utils", "dynamics.jl")))
    include($(joinpath(GEN_DIR, "naming.jl")))

    MODEL_DIR = "time_rand_window_binary"
    DATA_ROOT = LOCAL_TEST ? normpath(joinpath($GEN_DIR, "..", "_local_test_output", "spreading")) :
                             normpath(joinpath($GEN_DIR, "..", "..", "data", "spreading"))

    # ---- knobs --------------------------------------------------------------
    block_len_vals = [1]                 # steps per disorder block

    average_epsilon_c    = 0.144         # epsilon_bar; with p_val = 0.2: epsilon_u = epsilon_bar / 0.24 = 0.6
    average_epsilon_rate = 0.0012        # -> steps of 0.005 in epsilon_u
    p_val     = 0.2                      # probability of the inactive step epsilon_u (BVH: 1 - 0.8)
    lower_div = 20                       # epsilon_l = epsilon_u / 20 (BVH: lambda_l = lambda_h / 20)
    # epsilon_bar = p*epsilon_u + (1-p)*epsilon_l = (p + (1-p)/lower_div) * epsilon_u
    upper_epsilon_c    = average_epsilon_c    / (p_val + (1-p_val)/lower_div)
    upper_epsilon_rate = average_epsilon_rate / (p_val + (1-p_val)/lower_div)
    upper_epsilons = [round(upper_epsilon_c + i * upper_epsilon_rate, digits=6) for i in -2:2]
    lower_epsilons = [round(upper_ep / lower_div, digits=6) for upper_ep in upper_epsilons]
    p_vals = fill(p_val, length(upper_epsilons))

    t_max             = 100000           # last output time
    points_per_decade = 20
    num_chunks        = 200              # chunks per parameter set
    runs_per_chunk    = 500              # runs per chunk (num_chunks * runs_per_chunk runs per set)
    chunk_offset      = 0                # set to the number of chunks already written to add more

    if LOCAL_TEST                        # tiny sizes for a quick check
        block_len_vals = [1, 3]; t_max = 200; num_chunks = 2; runs_per_chunk = 5
        upper_epsilons = upper_epsilons[2:3]; lower_epsilons = lower_epsilons[2:3]; p_vals = p_vals[2:3]
    end
end

println("Writing to $(DATA_ROOT)")

jobs = [(block_len=b, u=u, l=l, p=p, chunk=c + chunk_offset)
        for b in block_len_vals for (u, l, p) in zip(upper_epsilons, lower_epsilons, p_vals) for c in 1:num_chunks]
println("t_max=$(t_max) | block_len=$(block_len_vals) | epsilon_u=$(upper_epsilons) | p=$(p_val) | ",
        "$(num_chunks) chunks x $(runs_per_chunk) runs per set | $(length(jobs)) jobs on $(nworkers()) workers")

@time pmap(jobs) do J
    record_times = make_log_times(t_max; points_per_decade=points_per_decade)
    sums = spreading_chunk(record_times, J.u, J.l, J.p, runs_per_chunk; block_len=J.block_len)
    path = spreading_chunk_path(DATA_ROOT, MODEL_DIR, t_max, J.u, J.l, J.p, J.block_len, points_per_decade,
                                runs_per_chunk, J.chunk)
    make_path_exist(path)
    CSV.write(path, DataFrame(sums))
    println("done: block_len=$(J.block_len) epsilon_u=$(J.u) chunk $(J.chunk)")
    nothing
end
