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
@everywhere num_unit_cells_vals = [8, 16, 32]
# @everywhere num_unit_cells_vals = [128]
# @everywhere num_unit_cells_vals = [64]
@everywhere J = 1    # energy factor

# J vector with some randomness
@everywhere J_vec = J .* [1, 1, 1]

# Time to evolve until push back to S_A
@everywhere tau = 1 * J

# --- Trying to Replecate Results ---
@everywhere num_initial_conds = 1000 # We are avraging over x initial conditions
@everywhere init_cond_name_offset = 0
a_vals = [0.2, 0.4, 0.6, 0.7, 0.8] # general a_vals
# a_vals = [0.7] # general a_vals

@everywhere epsilon = 10^-5

@everywhere N_val = 4

z_val = 1.6
z_val_name = replace("$z_val", "." => "p")

# --- geting spin dists ---

for num_unit_cells in num_unit_cells_vals
    L = num_unit_cells * N_val
    println("L_val: $L")

    # Time we evolve to
    n = Int(round(L^z_val))

    states_evolve_func = random_global_control_evolve

    # Define s_naught to be used during control step
    S_NAUGHT = make_spiral_state(L, (2) / N_val)

    for a_val in a_vals
        println("L_val: $L | a_val: $a_val")
        a_val_name = replace("$a_val", "." => "p")

        # define the variables for the workers to use
        let a_val=a_val, L=L, n=n, S_NAUGHT=S_NAUGHT, num_initial_conds=num_initial_conds, states_evolve_func=states_evolve_func
            @sync @distributed for init_cond in 1:num_initial_conds
                println("L_val: $L | a_val: $a_val | IC: $init_cond / $num_initial_conds")

                spin_chain_A = S_NAUGHT + epsilon * make_random_state(L) # our S_A

                evolved_results = states_evolve_func(J_vec, spin_chain_A, a_val, n, J, S_NAUGHT)

                current_sdiffs = weighted_spin_difference_vs_time(spin_chain_A, S_NAUGHT)

                sample_filepath = "data/s_diff_per_time/N$N_val/a$a_val_name/IC1/L$L/N$(N_val)_a$(a_val_name)_IC1_L$(L)_z$(z_val_name)_sample$(init_cond+init_cond_name_offset).csv"
                make_path_exist(sample_filepath)
                df = DataFrame("t" => 1:n, "s_diff" => current_sdiffs)
                CSV.write(sample_filepath, df)
            end
        end
    end
end

end