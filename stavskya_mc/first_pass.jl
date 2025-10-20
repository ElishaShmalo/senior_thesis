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

    # L_vals = [8000, 10_000, 12_000, 14_000, 16_000, 18_000, 20_000]
    L_vals = [16_000]
    epsilon_vals = sort(union([round(0.001 * i, digits=4) for i in 0:350], [round(0.291 + 0.0001 * i, digits=4) for i in 0:60]))
    # epsilon_vals = [round(0.001 * i, digits=4) for i in 0:350]
    # epsilon_vals = [round(0.001 * i, digits=4) for i in 0:10]

    epsilon_vals = [ep for ep in epsilon_vals if 0.345 <= ep]

    time_prefact = 200

    num_initial_conds = 500
    initial_state_prob = 0.5
end

collected_rhos = Dict{Int, Dict{Float64, Vector{Float64}}}()

@time begin

# get all the data
for L_val in L_vals
    println("L_val: $(L_val)")

    collected_rhos[L_val] = Dict{Float64, Vector{Float64}}()

    for epsilon_val in epsilon_vals
        println("L_val: $(L_val) | Epsilon $(epsilon_val)")
        all_init_outputs = [0.0 for _ in 1:num_initial_conds]

        epsilon_val_name = replace("$epsilon_val", "." => "p")

        let epsilon_val=epsilon_val, L_val=L_val, num_initial_conds=num_initial_conds
            all_init_outputs = @distributed (vcat) for init_cond in 1:num_initial_conds
                
                state = make_rand_state(L_val, initial_state_prob)

                evolved_state = evolve_state(state, L_val*time_prefact, epsilon_val)
                current_rho = calculate_avg_alive(evolved_state)

                [current_rho]
            end
        end

        collected_rhos[L_val][epsilon_val] = all_init_outputs
        # Save init cond data as csv
        sample_filepath = "stavskya_mc/data/rho_per_epsilon/IC1/L$(L_val)/IC1_L$(L_val)_epsilon$(epsilon_val_name).csv"
        make_path_exist(sample_filepath)
        df = DataFrame("sample" => 1:num_initial_conds, "rho" => collected_rhos[L_val][epsilon_val])
        CSV.write(sample_filepath, df)
    end

end

end