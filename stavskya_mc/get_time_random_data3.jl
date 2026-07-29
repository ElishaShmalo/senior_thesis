using Distributed
using SlurmClusterManager

# The way this works is we fix epsilon_prime range from 0 to 1
# Each timestep, epsilon = epsilon_prime +/- [-delta, delta]
# And we varry delta

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

    # L_vals = [8000, 10_000, 12_000, 14_000, 16_000, 18_000, 20_000]
    L_vals = [500, 1000, 2000, 4000, 8000]
    # epsilon_prime_vals = [round(0.2526 + 0.0001 * i, digits=4) for i in -30:30 if 0.253 <= round(0.2526 + 0.0001 * i, digits=4)]
    epsilon_prime_vals = [round(0.2878 + 0.0001 * i, digits=4) for i in -30:30]

    time_prefact = 200

    num_initial_conds = 1000
    initial_state_prob = 0.5

    delta_vals = [0.1]
end

@time begin
    
# get all the data
for L_val in L_vals
    println("L_val: $(L_val)")

    for delta_val in delta_vals
        println("L_val: $(L_val) | delta: $(delta_val)")
        delta_val_name = replace("$delta_val", "." => "p")
        for epsilon_prime in epsilon_prime_vals
            delta_val_to_use = min(epsilon_prime, delta_val)
            println("L_val: $(L_val) | delta: $(delta_val) | EpsilonPrime $(epsilon_prime)")
            all_init_outputs = [0.0 for _ in 1:num_initial_conds]

            epsilon_val_name = replace("$epsilon_prime", "." => "p")

            let delta_val_to_use=delta_val_to_use, epsilon_prime=epsilon_prime, L_val=L_val, num_initial_conds=num_initial_conds
                all_init_outputs = @distributed (vcat) for init_cond in 1:num_initial_conds
                    
                    state = make_rand_state(L_val, initial_state_prob)

                    evolved_state = time_random_delta_evolve_state(state, L_val*time_prefact, epsilon_prime, delta_val_to_use)
                    current_rho = calculate_avg_alive(evolved_state)

                    [current_rho]
                end
            end

            # Save init cond data as csv
            sample_filepath = "stavskya_mc/data/time_rand_delta/rho_per_epsilon/IC1/L$(L_val)/delta$(delta_val_name)/IC$(num_initial_conds)_L$(L_val)_epsilon$(epsilon_val_name).csv"
            make_path_exist(sample_filepath)
            df = DataFrame("sample" => 1:num_initial_conds, "rho" => all_init_outputs)
            CSV.write(sample_filepath, df)
        end
    end
end

end