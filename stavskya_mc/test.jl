using Distributed

# Small local sliding-p run for validating analyze_random_slidding_p.ipynb.
# Run from the repository root with: julia stavskya_mc/test.jl
script_dir = @__DIR__
const LOCAL_WORKERS = min(4, max(1, Sys.CPU_THREADS - 1))
addprocs(LOCAL_WORKERS)

@everywhere begin
    const SCRIPT_DIR = $script_dir
    println("Hello from worker $(myid()) on host $(gethostname())")

    # Imports
    using Random
    using LinearAlgebra
    using Plots
    using Serialization
    using Statistics
    using DelimitedFiles, SharedArrays, CSV, DataFrames

    include(joinpath(SCRIPT_DIR, "utils", "general.jl"))
    include(joinpath(SCRIPT_DIR, "utils", "calculations.jl"))
    include(joinpath(SCRIPT_DIR, "utils", "dynamics.jl"))

    # This profile is intentionally cheap but has five p values, enough for
    # the collapse code's minimum-data check.
    L_vals = [10, 20, 40, 80, 160]
    upper_val = 0.322
    lower_val = upper_val / 10
    p_vals = sort(union([round(i, digits=6) for i in 0.0:0.05:0.9], [round(i, digits=6) for i in 0.25:0.025:0.75]))
    upper_epsilons = fill(upper_val, length(p_vals))
    lower_epsilons = fill(lower_val, length(p_vals))
    
    z_val = 1.45
    z_val_name = replace("$(z_val)", "." => "p")
    
    num_initial_conds = 200
    initial_state_prob = 0.5

    function float_str(f)
        return replace("$f", "." => "p")
    end
end

@time begin
    
# get all the data
for L_val in L_vals

    T_f = Int(round(L_val^z_val))

    for (upper_ep, lower_ep, p_val) in zip(upper_epsilons, lower_epsilons, p_vals)
        println("L_val: $(L_val) | EpsilonU: $(upper_ep) | EpsilonL: $(lower_ep) | P: $(p_val)")
        all_init_outputs = [0.0 for _ in 1:num_initial_conds]

        upper_epsilon_val_name = float_str(upper_ep)
        lower_epsilon_val_name = float_str(lower_ep)
        p_val_name = float_str(p_val)

        let upper_ep=upper_ep, lower_ep=lower_ep, p_val=p_val, L_val=L_val, num_initial_conds=num_initial_conds
            all_init_outputs = @distributed (vcat) for init_cond in 1:num_initial_conds
                
                state = make_rand_state(L_val, initial_state_prob)

                evolved_state = time_random_p_evolve_state(state, T_f, upper_ep, lower_ep, p_val)
                current_rho = calculate_avg_alive(evolved_state)

                [current_rho]
            end
        end

        # Save init cond data as csv
        sample_filepath = joinpath(SCRIPT_DIR, "data", "time_rand_slidding_p", "rho_per_epsilon", "IC1", "L$(L_val)", "IC$(num_initial_conds)_L$(L_val)_epsilonu$(upper_epsilon_val_name)_epsilonl$(lower_epsilon_val_name)_pval$(p_val_name)_z$(z_val_name).csv")
        make_path_exist(sample_filepath)
        df = DataFrame("sample" => 1:num_initial_conds, "rho" => all_init_outputs)
        CSV.write(sample_filepath, df)
    end
end

end
