using Distributed
using SlurmClusterManager

println("We are adding $(SlurmManager()) workers")
addprocs(SlurmManager())

@everywhere begin
    println("Hello from worker $(myid()) on host $(gethostname())")

    # Imports
    using Random
    using LinearAlgebra
    using Plots
    using Serialization
    using Statistics
    using DelimitedFiles, SharedArrays, CSV, DataFrames

    include("utils/general.jl")
    include("utils/calculations.jl")
    include("utils/dynamics.jl")

    function float_str(f)
        return replace("$f", "." => "p")
    end

    # L_vals = [8000, 10_000, 12_000, 14_000, 16_000, 18_000, 20_000]
    L_vals = [1000, 2000, 4000, 8000]
    
    upper_val = 0.43
    lower_div = 10
    lower_val = upper_val / lower_div

    p_vals = sort(union([round(i, digits=6) for i in 0.0:0.1:0.9], [round(i, digits=6) for i in 0.4:0.005:0.5]))
    upper_epsilons = [upper_val for _ in 1:length(p_vals)]
    lower_epsilons = [lower_val for _ in 1:length(p_vals)]
    
    
    z_val = 1.45
    z_val_name = float_str(z_val)
    
    num_initial_conds = 3000
    num_init_conds_offset = 0
    initial_state_prob = 0.5

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
        sample_filepath = "stavskya_mc/data/time_rand_slidding_p/rho_per_epsilon/IC1/L$(L_val)/IC$(num_initial_conds)_L$(L_val)_epsilonu$(upper_epsilon_val_name)_epsilonl$(lower_epsilon_val_name)_pval$(p_val_name)_z$(z_val_name).csv"
        make_path_exist(sample_filepath)
        df = DataFrame("sample" => 1:num_initial_conds, "rho" => all_init_outputs)
        CSV.write(sample_filepath, df)
    end
end

end
