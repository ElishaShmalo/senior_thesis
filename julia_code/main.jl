# Hyzenberg Spin Chain Simulation in Julia
# Doing ferromagnet because it's cooler :)

# Imports
using Random
using LinearAlgebra
using Plots
using DifferentialEquations
using StaticArrays
using Serialization

# Other files   
include("utils/make_spins.jl")
include("utils/dynamics.jl")
include("analytics/spin_diffrences.jl")

# Set plotting theme
Plots.theme(:dark)

# General Variables
L = 4*64  # number of spins
J = 1       # energy factor

# J vector with some randomness
J_vec = J .* [rand([-1, 1]), rand([-1, 1]), 1]

# Time step for evolution
Tau_F = 1 / J

# Number of spins pushed each local control push
local_N_push = div(L, 10)
local_control_index = 1

# Evolve until
T = L

# Define s_naught as a constant
S_NAUGHT = make_spiral_state(L)


# --- Trying to Replecate Results ---
num_init_cond = 500 # We are avraging over x initial conditions
a_vals = [0.6, 0.68, 0.7, 0.716, 0.734, 0.766, 0.8, 0.86, 0.9]

original_random = make_random_state()

S_diffs = Dict{Float64, Vector{Float64}}()

# We will use an array to store the results of the simulation for each a_val. 
# First dimention represents the particular initial condition, Second is the time, third is the state at that time, 
# fourth is the particular spin value
# i.e. evolved_states[1, 2, 3, :] is the third spin at time t=2 for the first initial condition 
# i.e. evolved_states[1, 2, :, :] is an array represnting the state of t=2, first initial condition
num_timesteps = L

for a_val in a_vals

    current_S_diffs = [Vector{Float64}([]) for _ in 1:num_init_cond]

    evolved_states = Array{Float64}(undef, num_init_cond, num_timesteps, L, 3)

    for i in 1:num_init_cond
        returned_states = global_control_evolve(original_random, a_val, L*J, Tau_F, S_NAUGHT)

        current_S_diffs[i] = [weighted_spin_difference(state, S_NAUGHT) for state in returned_states]
        # Transfer returned_states to array
        for t in eachindex(returned_states)
            for l in eachindex(returned_states[t])
                for d in eachindex(returned_states[t][l])
                    evolved_states[i, t, l, d] = returned_states[t][l][d]
                end
            end
        end
    end

    S_diffs[a_val] = sum(current_S_diffs) / num_init_cond

    # Lets save the results such that we don't have to run the simulation every time
    results_file_name = "results_with_a_vals_" * replace("$a_val", "." => "p")

    open("data/evolved_spins/" * results_file_name * ".dat", "w") do io
        serialize(io, evolved_states)
    end
end

# --- Plotting the Dynamics of S_NAUGHT ---

ts = [i * Tau_F for i in 0:length(S_diffs[a_vals[1]])-1]

plt = plot()
for a_val in a_vals
    plot!(ts, S_diffs[a_val], label="a = $(a_val)")
end

xlabel!("time")
ylabel!("S_diff")
title!("Spin Dynamics")
display(plt)

savefig("figs/s_diff_plot_diffrent_a_vals_IC$(num_init_cond)_L$(L).png")

# --- Analyzing the avraged results ---

results_file_name = "results_with_a_vals_" * replace(join(["$(a_val)" for a_val in a_vals], "_"), "." => "p")

avraged_evolved_states = open("data/evolved_spins/" * results_file_name * ".dat", "r") do io
    deserialize(io)
end

# Examining the delta spin evolution of the avrage trajectory

# delta_spins = Dict{Float64, Vector{Vector{Float64}}}()

# for a_val in keys(avraged_evolved_states)
#     for t in avraged_evolved_states
#     println(delta_spin(avraged_evolved_states[a_val][1], S_NAUGHT))
# end