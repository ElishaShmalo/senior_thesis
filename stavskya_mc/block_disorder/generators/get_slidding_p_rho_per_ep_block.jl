# rho at t = L^z for many control values and system sizes (finite-size-scaling scan)
# for the sliding-p time-random Stavskaya model, with block disorder.
#
# 2026-09 successor of stavskya_mc/get_time_random_slidding_p.jl. The only
# change is the block_len knob: disorder is held fixed over blocks of block_len
# steps (1 = old model). block_len_vals is looped over like L_vals, and block_len
# appears in the file name. Data go to
# stavskya_mc/data/block_rho_per_ep/time_rand_slidding_p/... (absolute path, see naming.jl).
#
# Cluster:          cd stavskya_mc/block_disorder/submit && sbatch submit_slidding_p_rho_per_ep_block.sh
# Local smoke test: STAV_LOCAL_TEST=1 julia stavskya_mc/block_disorder/generators/get_slidding_p_rho_per_ep_block.jl

const GEN_DIR = @__DIR__
include(joinpath(GEN_DIR, "setup_workers.jl"))

@everywhere begin
    using Random, Statistics, CSV, DataFrames
    include($(joinpath(GEN_DIR, "..", "..", "utils", "calculations.jl")))   # also includes dynamics.jl and general.jl
    include($(joinpath(GEN_DIR, "naming.jl")))

    MODEL_DIR = "time_rand_slidding_p"
    DATA_ROOT = LOCAL_TEST ? normpath(joinpath($GEN_DIR, "..", "_local_test_output", "block_rho_per_ep")) :
                             normpath(joinpath($GEN_DIR, "..", "..", "data", "block_rho_per_ep"))

    # ---- knobs --------------------------------------------------------------
    L_vals = [1000, 2000, 4000, 8000, 16000]
    block_len_vals = [1]                 # NEW: steps per disorder block (1 = old model)
    upper_val = 0.43
    lower_div = 20
    lower_val = round(upper_val / lower_div, digits=6)
    p_vals = sort(union([round(i, digits=6) for i in 0.465:0.002:0.505]))
    upper_epsilons = [upper_val for _ in 1:length(p_vals)]
    lower_epsilons = [lower_val for _ in 1:length(p_vals)]
    z_val = 1.45                         # t_f = round(Int, L^z_val)

    num_initial_conds = 3000
    initial_state_prob = 0.5

    if LOCAL_TEST
        L_vals = [16, 32]; block_len_vals = [1, 3]; num_initial_conds = 4
        upper_epsilons = upper_epsilons[1:3]; lower_epsilons = lower_epsilons[1:3]; p_vals = p_vals[1:3]
    end
end

println("Writing to $(DATA_ROOT)")

@time begin
    for L_val in L_vals, block_len in block_len_vals
        T_f = Int(round(L_val^z_val))
        for (upper_ep, lower_ep, p_v) in zip(upper_epsilons, lower_epsilons, p_vals)
            println("L=$(L_val) | block_len=$(block_len) | epsilon_u=$(upper_ep) | epsilon_l=$(lower_ep) | p=$(p_v)")
            all_init_outputs = let L_val=L_val, block_len=block_len, T_f=T_f, upper_ep=upper_ep, lower_ep=lower_ep, p_v=p_v
                @distributed (vcat) for init_cond in 1:num_initial_conds
                    state = make_rand_state(L_val, initial_state_prob)
                    evolved_state = time_random_p_evolve_state(state, T_f, upper_ep, lower_ep, p_v; block_len=block_len)
                    [calculate_avg_alive(evolved_state)]
                end
            end
            sample_filepath = block_rho_per_ep_path(DATA_ROOT, MODEL_DIR, L_val, upper_ep, lower_ep, p_v,
                                                    block_len, z_val, num_initial_conds)
            make_path_exist(sample_filepath)
            CSV.write(sample_filepath, DataFrame("sample" => 1:num_initial_conds, "rho" => all_init_outputs))
        end
    end
end
