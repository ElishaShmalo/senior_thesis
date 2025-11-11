# Looking for solitons in the hyzenberg spin chain

# Imports
using Random
using LinearAlgebra
using Plots
using DifferentialEquations
using StaticArrays
using Serialization
using Statistics

# Other files   
include("utils/make_spins.jl")
include("utils/general.jl")
include("utils/dynamics.jl")
include("analytics/spin_diffrences.jl")

# General Variables
global_L = 128  # number of spins
J = 1       # energy factor

# J vector
J_vec = J .* [1, 1, 1]

# Time step for evolution
Tau_F = 1 / J

# Evolve until
T = global_L

# --- Trying to Replecate Results ---
num_init_cond = 1 # We are avraging over x initial conditions
# a_vals = [round(0.5 + i*0.1, digits = 2) for i in 0:5] 
a_vals = [0.7616, 0.762] 

N_vals = [4]
z_val = 1

# Testing diffrent (random) initial conditions to see what they yeild
num_trials = 1
@time begin

for N_val in N_vals
    println("N: $N_val")

    L = global_L * N_val
    
    # Define s_naught as a constant
    S_NAUGHT = make_spiral_state(L, (2) / N_val)
    
    num_timesteps = L^z_val

    states_evolve_func = random_evolve_spins_to_time

    epsilon = 0.1

    for a_val in a_vals
        println("N: $N_val | a_val $a_val")
        
        for trial_num in 1:num_trials
            current_spin_delta = Vector{Vector{Float64}}([]) # []: Initial condition, Vec: Time, Vec: spin location, Float: δs
            OTOC_A_B = [Vector{Vector{Float64}}([]) for _ in 1:num_init_cond]

            spin_chain_A = make_random_state(L) # our S_A
            for i in 1:num_init_cond
                println("N: $N_val | a_val $a_val | IC $(i) | Trial $(trial_num)")

                

                # Making spin_chain_B to be spin_chain_A with the middle spin modified
                spin_chain_B = copy(spin_chain_A)
                new_mid_spin_val = spin_chain_B[div(length(spin_chain_B), 2)] + make_random_spin(epsilon)
                spin_chain_B[div(length(spin_chain_B), 2)] = normalize(new_mid_spin_val)

                returned_states = states_evolve_func(copy(J_vec), spin_chain_A, spin_chain_B, a_val, num_timesteps, J, S_NAUGHT)
                evolved_spin_chain_A = returned_states[1]
                evolved_spin_chain_B = returned_states[2]

                current_spin_delta = [get_delta_spin(evolved_spin_chain_A[t], S_NAUGHT) for t in 1:num_timesteps]
                OTOC_A_B[i] = [get_OTOC(evolved_spin_chain_B[t], evolved_spin_chain_A[t]) for t in 1:num_timesteps]
            end
            
            # saving avrage of δs for future ref
            aval_path = "$(replace("$a_val", "." => "p"))"
            
            results_file_path = "data/delta_evolved_spins/N$(N_val)/a$(aval_path)/IC$(num_init_cond)/L$L/"

            make_data_file(results_file_path * "N$(N_val)_a$(aval_path)_IC$(num_init_cond)_L$(L)_trial$(trial_num)_time_rand_deltaS.data", current_spin_delta)
            make_data_file(results_file_path * "N$(N_val)_a$(aval_path)_IC$(num_init_cond)_L$(L)_trial$(trial_num)_time_rand_OTOC.data", sum(OTOC_A_B)/num_init_cond)
        end
    end
end

end