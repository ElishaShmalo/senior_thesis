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


# --- Lets check that S_NAUGHT is actually stable ---
num_init_cond_spiral = 1
spiral_angle = pi / 2
original_spiral = make_spiral_state(L, spiral_angle)

S_diffs_spiral_per_ic = [Float64[] for _ in 1:num_init_cond_spiral]

for i in 1:num_init_cond_spiral
    current_returned_states = no_control_evolve(original_spiral, L*J, Tau_F)
    
    S_diffs_spiral_per_ic[i] = [weighted_spin_difference(state, S_NAUGHT) for state in current_returned_states]
end

S_diffs_spiral = sum(S_diffs_spiral_per_ic) / num_init_cond_spiral

plot([i for i in 1:length(S_diffs_spiral)], S_diffs_spiral, xlabel="Time", ylabel="S_diff", title="spiral_state: $spiral_angle")

savefig("figs/tests/s_diff_spiral_plot_$(replace(string(round(spiral_angle, digits=3)), "." => "p")).png")

# --- Test with a = x ---
num_init_cond_test = 10
a_val_test = 0.766
original_random = make_random_state(L)

S_diffs_test_per_ic = [Float64[] for _ in 1:num_init_cond_test]

for i in 1:num_init_cond_test
    current_returned_states = global_control_evolve(original_random, a_val_test, L*J, Tau_F, S_NAUGHT)

    S_diffs_test_per_ic[i] = [weighted_spin_difference(state, S_NAUGHT) for state in current_returned_states]
end

S_diffs_test = sum(S_diffs_test_per_ic) / num_init_cond_test

plot([i for i in 1:length(S_diffs_test)], S_diffs_test, xlabel="Time", ylabel="S_diff", title="a = $a_val_test")

savefig("figs/tests/s_diff_plot_$(replace(string(a_val_test), "." => "p")).png")

