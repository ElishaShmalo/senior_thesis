# Log-time decay runs for the upper/lower binary time-random Stavskaya model:
# several control values at one (or a few) system sizes.
#
# 2026-09 successor of stavskya_mc/get_time_random_uppper_lower_binary_time_data.jl.
# The parameters mean the same thing as there. Differences:
#   * output times are log spaced (make_log_times) from t = 0 up to
#     T_f = round(L * time_prefact), with `points_per_decade` points per decade;
#   * disorder is held fixed over blocks of `block_len` steps
#     (block_len = 1 is the old model); block_len_vals is looped over like L_vals;
#   * each sample is one uninterrupted evolution (time_random_p_record_rho), so
#     the disorder blocks stay aligned to absolute time;
#   * data go to stavskya_mc/data/time_log/time_rand_window_binary/..., and every
#     file name contains "blocklen<b>" and "time_log" (see naming.jl).
#     The path is absolute (built from this file's location), so it no longer
#     depends on the directory you launch from.
#
# NOTE: the critical point depends on block_len. The values below were found for
# block_len = 1; rescan them before using block_len > 1.
#
# Cluster:          cd stavskya_mc/block_disorder/submit && sbatch submit_upper_lower_time_log.sh
# Local smoke test: STAV_LOCAL_TEST=1 julia stavskya_mc/block_disorder/generators/get_upper_lower_binary_time_log.jl

const GEN_DIR = @__DIR__
include(joinpath(GEN_DIR, "setup_workers.jl"))

@everywhere begin
    using Random, Statistics, CSV, DataFrames
    include($(joinpath(GEN_DIR, "..", "..", "utils", "dynamics.jl")))
    include($(joinpath(GEN_DIR, "naming.jl")))

    MODEL_DIR = "time_rand_window_binary"
    DATA_ROOT = LOCAL_TEST ? normpath(joinpath($GEN_DIR, "..", "_local_test_output", "time_log")) :
                             normpath(joinpath($GEN_DIR, "..", "..", "data", "time_log"))

    # ---- knobs --------------------------------------------------------------
    L_vals = [35000]
    block_len_vals = [1]                 # NEW: steps per disorder block (1 = old model)

    average_epsilon_c    = 0.27033
    average_epsilon_rate = 0.00005
    p_val     = 0.8
    lower_div = 20
    # epsilon_bar = p*epsilon_u + (1-p)*epsilon_l = (p + (1-p)/lower_div) * epsilon_u = 0.81 epsilon_u
    upper_epsilon_c    = average_epsilon_c    / (p_val + (1-p_val)/lower_div)
    upper_epsilon_rate = average_epsilon_rate / (p_val + (1-p_val)/lower_div)
    upper_epsilons = [round(upper_epsilon_c + i * upper_epsilon_rate, digits=6) for i in -3:3]
    lower_epsilons = [round(upper_ep / lower_div, digits=6) for upper_ep in upper_epsilons]
    p_vals = fill(p_val, length(upper_epsilons))

    time_prefact      = 100.0            # T_f = round(Int, L * time_prefact); see review 4.2.3 before choosing
    points_per_decade = 20
    num_initial_conds = 3000
    num_init_conds_offset = 0
    initial_state_prob = 0.5

    if LOCAL_TEST                        # tiny sizes for a quick check
        L_vals = [64]; block_len_vals = [1, 3]; time_prefact = 4.0
        num_initial_conds = 4; upper_epsilons = upper_epsilons[3:5]
        lower_epsilons = lower_epsilons[3:5]; p_vals = p_vals[3:5]
    end
end

println("Writing to $(DATA_ROOT)")

@time begin
    for L_val in L_vals, block_len in block_len_vals
        T_f = round(Int, L_val * time_prefact)
        record_times = make_log_times(T_f; points_per_decade=points_per_decade)
        for (upper_ep, lower_ep, p_v) in zip(upper_epsilons, lower_epsilons, p_vals)
            println("L=$(L_val) | block_len=$(block_len) | epsilon_u=$(upper_ep) | epsilon_l=$(lower_ep) | p=$(p_v) | T_f=$(T_f) | $(length(record_times)) output times")
            let L_val=L_val, block_len=block_len, record_times=record_times, upper_ep=upper_ep, lower_ep=lower_ep, p_v=p_v
                @sync @distributed for init_cond in 1:num_initial_conds
                    state = make_rand_state(L_val, initial_state_prob)
                    rho = time_random_p_record_rho(state, record_times, upper_ep, lower_ep, p_v; block_len=block_len)
                    sample_filepath = time_log_sample_path(DATA_ROOT, MODEL_DIR, L_val, upper_ep, lower_ep, p_v, block_len,
                                                           time_prefact, points_per_decade, init_cond + num_init_conds_offset)
                    make_path_exist(sample_filepath)
                    CSV.write(sample_filepath, DataFrame("time" => record_times, "rho" => rho))
                end
            end
        end
    end
end
