# In this file we numerically approximate the Lyapunov for diffrent a-vals and N (for the spiral) vals using the tech from Benettin
using Distributed
using SlurmClusterManager

println("We are adding $(SlurmManager()) workers")
addprocs(SlurmManager())

@everywhere begin
    println("Hello from worker $(myid()) on host $(gethostname())")
end

# Imports
@everywhere using Random, LinearAlgebra, DifferentialEquations, Serialization, Statistics, DelimitedFiles, SharedArrays, CSV, DataFrames

# Other files   
@everywhere include("../utils/make_spins.jl")
@everywhere include("../utils/general.jl")
@everywhere include("../utils/dynamics.jl")
@everywhere include("../utils/lyapunov.jl")
@everywhere include("../analytics/spin_diffrences.jl")

@time begin

# General Variables
# @everywhere num_unit_cells_vals = [8, 16, 32, 64]
# @everywhere num_unit_cells_vals = [128]
@everywhere num_unit_cells_vals = [32, 64]
@everywhere J = 1    # energy factor

# J vector with some randomness
@everywhere J_vec = J .* [1, 1, 1]

# Time to evolve until push back to S_A
@everywhere tau = 1 * J

# --- Trying to Replecate Results ---
@everywhere num_initial_conds = 1000 # We are avraging over x initial conditions
@everywhere init_cond_name_offset = 0
a_vals = [0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.7525, 0.755, 0.7563, 0.7575, 0.7588,  0.7594, 0.76, 0.7605, 0.761, 0.7615, 0.762, 0.7625, 0.763, 0.765, 0.7675, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82] # general a_vals
# a_vals = [0.7] # general a_vals

@everywhere epsilon = 0.1

@everywhere N_val = 4

z_val = 1.7
z_val_name = replace("$z_val", "." => "p")

# --- geting spin dists ---

for num_unit_cells in num_unit_cells_vals
    L = num_unit_cells * N_val
    println("L_val: $L")

    # number of pushes we are going to do
    n = Int(round(L^z_val))

    states_evolve_func = evolve_spins_to_time

    # Define s_naught to be used during control step
    S_NAUGHT = make_spiral_state(L, (2) / N_val)

    for a_val in a_vals
        println("L_val: $L | a_val: $a_val")
        a_val_name = replace("$a_val", "." => "p")

        # define the variables for the workers to use
        let a_val=a_val, L=L, n=n, S_NAUGHT=S_NAUGHT, num_initial_conds=num_initial_conds, states_evolve_func=states_evolve_func
            @sync @distributed for init_cond in 1:num_initial_conds
                println("L_val: $L | a_val: $a_val | IC: $init_cond / $num_initial_conds")

                J_vec = J .* [1, 1, 1]

                spin_chain_A = make_random_state(L) # our S_A

                # Making spin_chain_B to be spin_chain_A with the middle spin modified
                spin_chain_B = copy(spin_chain_A)
                new_mid_spin_val = spin_chain_B[div(length(spin_chain_B), 2)] + make_random_spin(epsilon)
                spin_chain_B[div(length(spin_chain_B), 2)] = normalize(new_mid_spin_val)
                
                # Will be used to calculate lyop exp
                current_spin_dists = zeros(n)
                current_sdiffs = zeros(n)

                # Do n pushes 
                for current_n in 1:n
                    # we need to change J_vec outside of the evolve func so that it is the same for S_A and S_B

                    # evolve both to time t' = t + tau with control
                    evolved_results = states_evolve_func(J_vec, spin_chain_A, spin_chain_B, a_val, tau, J, S_NAUGHT)
                    spin_chain_A = evolved_results[1][end]
                    spin_chain_B = evolved_results[2][end]

                    current_spin_dists[current_n] = calculate_spin_distence(spin_chain_A, spin_chain_B)
                    spin_chain_B = push_back(spin_chain_A, spin_chain_B, epsilon)

                    current_sdiffs[current_n] = weighted_spin_difference(spin_chain_A, S_NAUGHT)
                end

                sample_filepath = "data/non_trand/spin_dists_per_time/N$N_val/a$a_val_name/IC1/L$L/N$(N_val)_a$(a_val_name)_IC1_L$(L)_z$(z_val_name)_sample$(init_cond+init_cond_name_offset).csv"
                make_path_exist(sample_filepath)
                df = DataFrame("t" => 1:n, "lambda" => calculate_lambda_per_time(current_spin_dists, epsilon), "delta_s" => current_sdiffs)
                CSV.write(sample_filepath, df)
            end
        end
    end
end

end