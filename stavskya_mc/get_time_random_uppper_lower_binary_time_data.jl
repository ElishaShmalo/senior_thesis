using Distributed
using SlurmClusterManager

println("We are adding $(SlurmManager()) workers")
addprocs(SlurmManager())

@everywhere begin
    using Random, LinearAlgebra, Plots, Serialization, Statistics
    using DelimitedFiles, SharedArrays, CSV, DataFrames

    include("utils/general.jl")
    include("utils/calculations.jl")
    include("utils/dynamics.jl")

    float_str(f) = replace("$f", "." => "p")

    L_vals = [20000]
    average_epsilon_c = 0.223
    average_epsilon_rate = 0.00005
    p_val = 1 / 2
    # epsilon_bar = (epsilon_u + epsilon_l) / 2 = 0.55 * epsilon_u.
    upper_epsilon_c = average_epsilon_c / 0.55
    upper_epsilon_rate = average_epsilon_rate / 0.55
    upper_epsilons = [round(upper_epsilon_c + i * upper_epsilon_rate, digits=6) for i in -3:3]
    lower_epsilons = [round(upper_ep / 10, digits=6) for upper_ep in upper_epsilons]
    p_vals = fill(p_val, length(upper_epsilons))

    time_prefact = 100
    time_step = 2000
    num_initial_conds = 2000
    num_init_conds_offset = 0
    initial_state_prob = 0.5
end

@time begin
    for L_val in L_vals
        T_f = L_val * time_prefact
        for (upper_ep, lower_ep, p_val) in zip(upper_epsilons, lower_epsilons, p_vals)
            upper_name, lower_name, p_name = float_str(upper_ep), float_str(lower_ep), float_str(p_val)
            println("L=$(L_val) | epsilon_u=$(upper_ep) | epsilon_l=$(lower_ep) | p=$(p_val)")
            @sync @distributed for init_cond in 1:num_initial_conds
                state = make_rand_state(L_val, initial_state_prob)
                current_rho = zeros(Float64, Int(round(T_f / time_step)))
                for t in eachindex(current_rho)
                    state = time_random_p_evolve_state(state, time_step, upper_ep, lower_ep, p_val)
                    current_rho[t] = calculate_avg_alive(state)
                end
                sample_filepath = "stavskya_mc/data/time_rand_window_binary/rho_per_time/IC1/L$(L_val)/epsilonu$(upper_name)/epsilonl$(lower_name)/pval$(p_name)/IC1_L$(L_val)_epsilonu$(upper_name)_epsilonl$(lower_name)_pval$(p_name)_timepref$(time_prefact)_timestep$(time_step)_sample$(init_cond + num_init_conds_offset).csv"
                make_path_exist(sample_filepath)
                CSV.write(sample_filepath, DataFrame("time" => (1:length(current_rho)) * time_step, "rho" => current_rho))
            end
        end
    end
end
