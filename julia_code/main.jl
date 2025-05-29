# Hyzenberg Spin Chain Simulation in Julia
# Doing ferromagnet because it's cooler :)

# Imports
using Random
using LinearAlgebra
using Plots
using DifferentialEquations
using StaticArrays

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
num_init_cond = 100 # We are avraging over x initial conditions
a_vals = [0.6, 0.68, 0.7, 0.716, 0.734, 0.766, 0.8, 0.86, 0.9]

original_random = make_random_state()

S_diffs = Dict{Float64, Vector{Float64}}()


for a_val in a_vals

    current_S_diffs = [Float64[] for _ in 1:num_init_cond]

    for i in 1:num_init_cond
        returned_states = global_control_evolve(original_random, a_val, L*J, Tau_F, S_NAUGHT)

        current_S_diffs[i] = [weighted_spin_difference(state, S_NAUGHT) for state in returned_states]
    end

    S_diffs[a_val] = sum(current_S_diffs) / num_init_cond
end

# --- Plotting the Dynamics ---

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

