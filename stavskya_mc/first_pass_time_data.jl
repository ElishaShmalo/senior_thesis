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
    L_vals = [20_000]
    epsilon_vals = [0.2943, 0.2944, 0.2945, 0.2946, 0.2947]
    time_prefact = 100
    
    num_initial_conds = 4000
    initial_state_prob = 0.5

    time_step = 50
end

@time begin
    
# get all the data
for L_val in L_vals
    println("L_val: $(L_val)")

    T_f = L_val*time_prefact

    println("L_val: $(L_val)")
    for epsilon in epsilon_vals
        println("L_val: $(L_val) | Epsilon $(epsilon)")
        all_init_outputs = [[0.0 for t in 1:Int(round(T_f/time_step))] for _ in 1:num_initial_conds]

        epsilon_val_name = replace("$epsilon", "." => "p")

        let epsilon=epsilon, L_val=L_val, num_initial_conds=num_initial_conds
            @distributed for init_cond in 1:num_initial_conds
                state = make_rand_state(L_val, initial_state_prob)
                current_rho = [0.0 for t in 1:Int(round(T_f/time_step))]

                for t in 1:Int(round(T_f/time_step))
                    state = evolve_state(state, time_step, epsilon)
                    current_rho[t] = calculate_avg_alive(state)
                end
                
                # Save init cond data as csv
                sample_filepath = "stavskya_mc/data/rho_per_time/IC1/L$(L_val)/IC1_L$(L_val)_epsilon$(epsilon_val_name)_timepref$(time_prefact)_sample$(init_cond).csv"
                make_path_exist(sample_filepath)
                df = DataFrame("time" => (1:length(current_rho))*time_step, "rho" => current_rho)
                CSV.write(sample_filepath, df)
            end
        end
    end
end

end
