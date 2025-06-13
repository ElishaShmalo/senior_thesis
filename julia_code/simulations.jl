# Hyzenberg Spin Chain Simulation in Julia
# Doing ferromagnet because it's cooler :)

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
include("utils/dynamics.jl")
include("analytics/spin_diffrences.jl")

# Set plotting theme
Plots.theme(:dark)
# General Variables
L = 64  # number of spins
J = 1       # energy factor

# J vector with some randomness
J_vec = J .* [rand([-1, 1]), rand([-1, 1]), 1]

# Time step for evolution
Tau_F = 1 / J

# Evolve until
T = L

# --- Trying to Replecate Results ---
num_init_cond = 200 # We are avraging over x initial conditions
a_vals = [0.6, 0.68, 0.7, 0.716, 0.734, 0.766, 0.8, 0.86, 0.9]

original_random = make_random_state()

# We will use an array to store the results of the simulation for each a_val. 
# First dimention represents the particular initial condition, Second is the time, third is the state at that time, 
# fourth is the particular spin value
# i.e. evolved_states[1, 2, 3, :] is the third spin at time t=2 for the first initial condition 
# i.e. evolved_states[1, 2, :, :] is an array represnting the state of t=2, first initial condition
num_timesteps = L

N_vals = [4, 6, 7, 8, 9, 10]

for N in N_vals
    println("N: $N")
    # Define s_naught as a constant
    S_NAUGHT = make_spiral_state(L, 2 * π / N)
    for a_val in a_vals
        println("N: $N | a_val $a_val")
        current_spin_delta = [Vector{Vector{Float64}}([]) for _ in 1:num_init_cond]

        for i in 1:num_init_cond
            returned_states = random_global_control_evolve(original_random, a_val, L*J, Tau_F, S_NAUGHT)

            current_spin_delta[i] = [get_delta_spin(state, S_NAUGHT) for state in returned_states]
            
        end

        # saving avrage of δs for future ref
        results_file_name = "N$(N)/N_$(N)_a_val_" * replace("$a_val", "." => "p") * "_IC$(num_init_cond)_L$L"

        open("data/delta_evolved_spins/" * results_file_name * "_avg.dat", "w") do io
            serialize(io, sum(current_spin_delta)/num_init_cond)
        end
    end
end

# --- Plotting the Dynamics of S_diif ---

for N in N_vals
    plt = plot()
    for a_val in a_vals
        results_file_name = "Nwd$(N)/N_$(N)_a_val_" * replace("$a_val", "." => "p") * "_IC$(num_init_cond)_L$L"

        delta_spins = open("data/delta_evolved_spins/" * results_file_name * "_avg.dat", "r") do io
            deserialize(io)
        end

        ts = [i * Tau_F for i in 0:size(delta_spins)[1]-1]

        S_diff = get_spin_diffrence_from_delta(delta_spins)

        plot!(ts, S_diff, label="a = $(a_val)")
    end

    xlabel!("time")
    ylabel!("S_diff")
    title!("Spin Dynamics for N $N")
    # display(plt)

    savefig("figs/N$N/s_diff_plot_diffrent_a_vals_N$(N)_IC$(num_init_cond)_L$(L).png")
end
