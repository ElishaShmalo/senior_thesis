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
    L_vals = [4000, 12_000]
    epsilon_prime_vals = sort(union([round(0.005 * i, digits=4) for i in 0:175], [round(0.291 + 0.001 * i, digits=4) for i in 0:6]))

    time_prefact = 200

    num_initial_conds = 1000
    initial_state_prob = 0.5

    delta_vals = [0.001, 0.01, 0.1, 0.25]
end

collected_rhos = Dict{Int, Dict{Float64, Vector{Float64}}}()

@time begin
    
# get all the data
for L_val in L_vals
    println("L_val: $(L_val)")

    collected_rhos[L_val] = Dict{Float64, Vector{Float64}}()

    for delta_val in delta_vals
        println("L_val: $(L_val) | delta: $(delta_val)")
        delta_val_name = replace("$delta_val", "." => "p")
        for epsilon_prime in epsilon_prime_vals
            delta_val = min(epsilon_prime, delta_val)
            println("L_val: $(L_val) | delta: $(delta_val) | EpsilonPrime $(epsilon_prime)")
            all_init_outputs = [0.0 for _ in 1:num_initial_conds]

            epsilon_val_name = replace("$epsilon_prime", "." => "p")

            let delta_val=delta_val, epsilon_prime=epsilon_prime, L_val=L_val, num_initial_conds=num_initial_conds
                all_init_outputs = @distributed (vcat) for init_cond in 1:num_initial_conds
                    
                    state = make_rand_state(L_val, initial_state_prob)

                    epsilon_val = (epsilon_prime - delta_val) + 2*delta_val*rand()
                    evolved_state = evolve_state(state, L_val*time_prefact, epsilon_val)
                    current_rho = calculate_avg_alive(evolved_state)

                    [current_rho]
                end
            end

            collected_rhos[L_val][epsilon_prime] = all_init_outputs
            # Save init cond data as csv
            sample_filepath = "stavskya_mc/data/time_rand_delta/rho_per_epsilon/IC1/L$(L_val)/delta$(delta_val_name)/IC1_L$(L_val)_epsilon$(epsilon_val_name).csv"
            make_path_exist(sample_filepath)
            df = DataFrame("sample" => 1:num_initial_conds, "rho" => collected_rhos[L_val][epsilon_prime])
            CSV.write(sample_filepath, df)
        end
    end
end

end