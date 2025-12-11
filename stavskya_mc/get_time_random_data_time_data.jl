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
    L_vals = [20000]
    epsilon_prime_c = 0.287808
    epsilon_prime_vals = [round(epsilon_prime_c + 0.0001 * i, digits = 6) for i in -5:10]
    time_prefact = 100
    
    num_initial_conds = 2000
    initial_state_prob = 0.5

    delta_vals = [0.1]

    time_step = 10
end

@time begin
    
# get all the data
for L_val in L_vals
    println("L_val: $(L_val)")

    T_f = L_val*time_prefact

    for delta_val in delta_vals
        println("L_val: $(L_val) | delta: $(delta_val)")
        delta_val_name = replace("$delta_val", "." => "p")
        for epsilon_prime in epsilon_prime_vals
            delta_val_to_use = min(epsilon_prime, delta_val)
            println("L_val: $(L_val) | delta: $(delta_val) | EpsilonPrime $(epsilon_prime)")
            all_init_outputs = [[0.0 for t in 1:Int(round(T_f/time_step))] for _ in 1:num_initial_conds]

            epsilon_val_name = replace("$epsilon_prime", "." => "p")

            let delta_val_to_use=delta_val_to_use, epsilon_prime=epsilon_prime, L_val=L_val, num_initial_conds=num_initial_conds
                @sync begin
                    @distributed for init_cond in 1:num_initial_conds
                        
                        state = make_rand_state(L_val, initial_state_prob)
                        current_rho = [0.0 for t in 1:Int(round(T_f/time_step))]

                        for t in 1:Int(round(T_f/time_step))
                            state = time_random_delta_evolve_state(state, time_step, epsilon_prime, delta_val_to_use)
                            current_rho[t] = calculate_avg_alive(state)
                        end
                        
                        # Save init cond data as csv
                        sample_filepath = "stavskya_mc/data/time_rand_delta/rho_per_time/IC1/L$(L_val)/delta$(delta_val_name)/IC1_L$(L_val)_epsilon$(epsilon_val_name)_timepref$(time_prefact)_sample$(init_cond).csv"
                        make_path_exist(sample_filepath)
                        df = DataFrame("time" => (1:length(current_rho))*time_step, "rho" => current_rho)
                        CSV.write(sample_filepath, df)
                    end
                end
            end
        end
    end
end

end
