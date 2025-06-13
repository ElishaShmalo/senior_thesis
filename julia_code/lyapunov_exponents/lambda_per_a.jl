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
include("../utils/make_spins.jl")
include("../utils/dynamics.jl")
include("../utils/lyapunov.jl")
include("../analytics/spin_diffrences.jl")

# Set plotting theme
Plots.theme(:dark)

# General Variables
L = 4*64  # number of spins
J = 1       # energy factor

# J vector with some randomness
J_vec = J .* [rand([-1, 1]), rand([-1, 1]), 1]

# Time to evolve until push back to S_A
tau = 3 * J

# number of pushes we are going to do
n = L

# --- Trying to Replecate Results ---
num_initial_conds = 50 # We are avraging over x initial conditions
a_vals = [0.1, 0.6, 0.68, 0.7, 0.716, 0.734, 0.766, 0.8, 0.86]
# N_vals = [4, 6, 7, 8, 9, 10]
N_vals = [6]

epsilon = 0.1

# --- Calculating Lambdas ---

collected_lambdas = Dict{Int, Dict{Float64, Float64}}() # Int: N_val, Float64: a_val, Float64: avrg lambda by each initilal cond

for N_val in N_vals
    println("N_val: $N_val")

    state_evolve_func = global_control_evolve
    if N_val == 4 # Need to evolve with randomized J_vec for N=4
        state_evolve_func = random_global_control_evolve
    end

    # Define s_naught to be used during control step
    S_NAUGHT = make_spiral_state(L, (2 * pi) / N_val)

    collected_lambdas[N_val] = Dict(a => 0 for a in a_vals)

    for a_val in a_vals
        println("N_val: $N_val | a_val: $a_val")
        current_lambdas = zeros(num_initial_conds)

        for init_cond in 1:num_initial_conds
            println("N_val: $N_val | a_val: $a_val | IC: $init_cond / $num_initial_conds")

            spin_chain_A = make_random_state(L) # Essentially our S_A

            # Making spin_chain_B to be spin_chain_A with the middle spin modified
            to_add = make_uniform_state(L, 0)
            to_add[div(length(to_add), 2)] = make_random_spin(epsilon)
            spin_chain_B = spin_chain_A .+ to_add
            spin_chain_B = spin_chain_B ./ map(norm, spin_chain_B)

            current_spin_dists = zeros(n)

            # Do n pushes 
            for current_n in 1:n
                # evolve both to time t' = t + tau with control
                spin_chain_A = state_evolve_func(spin_chain_A, a_val, tau, J, S_NAUGHT)[end]
                spin_chain_B = state_evolve_func(spin_chain_B, a_val, tau, J, S_NAUGHT)[end]

                d_abs = calculate_spin_distence(spin_chain_A ./ map(norm, spin_chain_A), spin_chain_B ./ map(norm, spin_chain_B))
                spin_chain_B = push_back(spin_chain_A, spin_chain_B, epsilon)

                current_spin_dists[current_n] = d_abs
            end
            current_lambdas[init_cond] = mean(calculate_lambda(current_spin_dists, tau, epsilon))
        end

        collected_lambdas[N_val][a_val] = mean(current_lambdas)
    end

    filename = "N$(replace("$N_val", "." => "p"))/" * "N$(N_val)_IC$(num_initial_conds)"

    open("data/spin_chain_lambdas/" * filename * ".dat", "w") do io
        serialize(io, collected_lambdas[N_val])
        println("Saved file $filename")
    end
end

plt = plot()

for N_val in N_vals
    filename = "N$(replace("$N_val", "." => "p"))/" * "N$(N_val)_IC$(num_initial_conds)"
    collected_lambdas[N_val] = open("data/spin_chain_lambdas/" * filename * ".dat", "r") do io
        deserialize(io)
    end
    plot!(sort(a_vals), [val for val in values(sort(collected_lambdas[N_val]))], label="N=$N_val")
end

xlabel!("a")
ylabel!("λ")
display(plt)
