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
    L_vals = [8000]
    start_epsilon = 0
    end_epsilon = 1/2
    rate = 0.002
    epsilon_prime_vals = [round(start_epsilon+i*rate, digits=6) for i in 0:Int(round((end_epsilon-start_epsilon)/rate))]
    
    z_val = 1.6
    z_val_name = replace("$(z_val)", "." => "p")
    
    num_initial_conds = 3000
    num_init_conds_offset = 0
    initial_state_prob = 0.5
end

@time begin
    
# get all the data
for L_val in L_vals

    T_f = Int(round(L_val^z_val))

    for epsilon_prime in epsilon_prime_vals
        println("L_val: $(L_val) | EpsilonPrime $(epsilon_prime)")
        all_init_outputs = [0.0 for _ in 1:num_initial_conds]

        epsilon_val_name = replace("$epsilon_prime", "." => "p")

        let epsilon_prime=epsilon_prime, L_val=L_val, num_initial_conds=num_initial_conds
            all_init_outputs = @distributed (vcat) for init_cond in 1:num_initial_conds
                
                state = make_rand_state(L_val, initial_state_prob)

                evolved_state = time_random_binary_evolve_state(state, T_f, epsilon_prime)
                current_rho = calculate_avg_alive(evolved_state)

                [current_rho]
            end
        end

        # Save init cond data as csv
        sample_filepath = "stavskya_mc/data/time_rand_binary/rho_per_epsilon/IC1/L$(L_val)/IC$(num_initial_conds)_L$(L_val)_epsilon$(epsilon_val_name)_z$(z_val_name).csv"
        make_path_exist(sample_filepath)
        df = DataFrame("sample" => 1:num_initial_conds, "rho" => all_init_outputs)
        CSV.write(sample_filepath, df)
    end
end

end
