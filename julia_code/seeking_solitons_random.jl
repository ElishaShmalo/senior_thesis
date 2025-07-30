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

# Set plotting theme
Plots.theme(:dark)
# General Variables
global_L = 4 * 64  # number of spins
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
a_vals = [0.9] 


# We will use an array to store the results of the simulation for each a_val. 
# First dimention represents the particular initial condition, Second is the time, third is the state at that time, 
# fourth is the particular spin value
# i.e. evolved_states[1, 2, 3, :] is the third spin at time t=2 for the first initial condition 
# i.e. evolved_states[1, 2, :, :] is an array represnting the state of t=2, first initial condition

# N_vals = [2, 4, 3, 6, 9, 10]
N_vals = [4, 10]

Js_rand = 0 # Js_rand ∈ {0, 1, 2}. 0: No random Js, 1: random J_x and J_y, 2: random J_x

# Testing diffrent (random) initial conditions to see what they yeild
num_trials = 10

for N_val in N_vals
    println("N: $N_val")

    L = get_nearest(N_val, global_L)
    
    # Define s_naught as a constant
    S_NAUGHT = make_spiral_state(L, (2) / N_val)
    
    num_timesteps = L

    state_evolve_func = global_control_evolve
    if N_val == 4 && Js_rand == 1 # Need to evolve with randomized J_vec for N=4
        state_evolve_func = random_global_control_evolve
    elseif N_val == 4 && Js_rand == 2
        state_evolve_func = semirand_global_control_evolve
    end

    J_vec = J .* [1, 1, 1]

    num_spins_off = N_val # Number of spins that will be diffrent to the spiral
    epsilon_off = 0.01 / num_spins_off

    for a_val in a_vals
        println("N: $N_val | a_val $a_val")
        current_spin_delta = [Vector{Vector{Float64}}([]) for _ in 1:num_init_cond] # []: Initial condition, Vec: Time, Vec: spin location, Float: δs

        for trial in 1:num_trials
            for i in 1:num_init_cond
                println("N: $N_val | a_val $a_val | IC $(i)")

                # It's going to be a spiral state with some spins a bit off
                original_random = make_spiral_state(L, (2) / N_val)
                shifted_index = div(length(original_random), 2) - div(num_spins_off, 2)
                for i in 1:num_spins_off
                    original_random[i+ shifted_index] = make_random_spin(epsilon_off)
                end

                returned_states = state_evolve_func(original_random, a_val, L*J, Tau_F, S_NAUGHT)

                current_spin_delta[i] = [get_delta_spin(state, S_NAUGHT) for state in returned_states]
            end
            
            # saving avrage of δs for future ref
            aval_path = "$(replace("$a_val", "." => "p"))"[1:3]
            
            results_file_path = "data/delta_evolved_spins/N$(N_val)/a$(aval_path)/IC$(num_init_cond)/L$L/N$(N_val)_a" * replace("$a_val", "." => "p") * "_IC$(num_init_cond)_L$(L)_rand$(Js_rand)_seksolNumOff$(num_spins_off)_EppOff$(replace("$epsilon_off", "." => "p"))_tri$(trial)_avg.dat"

            make_data_file(results_file_path, sum(current_spin_delta)/num_init_cond)
        end
    end
end
