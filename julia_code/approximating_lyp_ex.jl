# In this file we numerically approximate the Lyapunov for diffrent a-vals and N (for the spiral) vals using the tech from Benettin

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
L = 4*64  # number of spins
J = 1       # energy factor

# J vector with some randomness
J_vec = J .* [rand([-1, 1]), rand([-1, 1]), 1]

# Time to evolve until push back to S_A
tau = J

# Number of spins pushed each local control push
local_N_push = div(L, 10)
local_control_index = 1

# number of pushes we are going to do
n = 20

# function for calculating d_i
function calculate_spin_distence(S_A::Vector{Vector{Float64}}, S_B::Vector{Vector{Float64}})
    dotted = map(dot, S_A, S_B)
    return sqrt(sum(2 .* (1 .- dotted)))
end

function push_back(S_A::Vector{Vector{Float64}}, S_B::Vector{Vector{Float64}}, epsilon_val)
    L = length(S_A)

    thing_to_add = (epsilon_val / L) .* ((S_B .- S_A) / map(norm, S_B .- S_A))

    return S_A .+ thing_to_add
end

function calculate_lambda(spin_dists, tau_val, epsilon_val)
    n_val = length(spin_dists)
    return (1/(n_val * tau_val)) * sum(map(log, spin_dists ./ epsilon_val))
end

# --- Trying to Replecate Results ---
num_init_cond = 500 # We are avraging over x initial conditions
a_vals = [0.55, 0.6, 0.65, 0.68, 0.7, 0.716, 0.734, 0.766, 0.8, 0.86, 0.9]
N_vals = [4]

epsilon = 10^(-5)

num_initial_conds = 10

collected_lambdas = Dict{Int, Dict{Float64, Vector{Float64}}}() # Int: N_val, Float64: a_val, Vec{Float64}: lambda for each initilal cond

for N_val in N_vals

    # Define s_naught to be used during control step
    S_NAUGHT = make_spiral_state(L, (2 * pi) / N)

    collected_lambdas[N_val] = Dict(a => zeros(num_initial_conds) for a in a_vals)

    for a_val in a_vals
        for init_cond in 1:num_init_cond

            spin_chain_A = make_random_state() # Essentially our S_A
            spin_chain_B = spin_chain_A .+ (epsilon .* make_random_state())
            
            current_n = 1
            t = 0

            current_spin_dists = zeros(n)

            # Do n pushes 
            while current_n < n
                # evolve both to time t' = t + tau with control
                spin_chain_A = global_control_evolve(spin_chain_A, a_val, tau, tau, S_NAUGHT)
                spin_chain_B = global_control_evolve(spin_chain_B, a_val, tau, tau, S_NAUGHT)

                t += tau

                d_abs = calculate_spin_distence(spin_chain_A, spin_chain_B)

                spin_chain_B = push_back(spin_chain_A, spin_chain_B, epsilon)

                current_n += 1
            end
        end
    end

end
