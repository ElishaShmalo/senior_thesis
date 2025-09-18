# In this file we examine how the local (in time) lambda_i evolve when we use the tech from Benettin
# We do this to see which lambda_i we should keep in the Benettin sum for lambda

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
L = 32 * 8 # number of spins
J = 1       # energy factor

# J vector with some randomness
J_vec = J .* [rand([-1, 1]), rand([-1, 1]), 1]

# Time to evolve until push back to S_A
tau = J

# number of pushes we are going to do
n = max(div(L,tau), 50) # This way the total time evolved is at least L

# --- Trying to Replecate Results ---
num_initial_conds = 10 # We are avraging over x initial conditions
a_vals = [round(0.6 + 0.02*i, digits=2) for i in 1:10]
N_vals = [4, 10]

epsilon = 0.1

# --- Lambda(t) (We are checking how long it takes for our lambda's to stabalize to a coherent value) ---   
lambda_of_times = Dict{Int, Dict{Float64, Vector{Float64}}}() # Int: N_val, Float64: a_val, Vec{Float64}: avg lambda of time

for N_val in N_vals
    println("N_val: $N_val")

    # Define s_naught to be used during control step
    S_NAUGHT = make_spiral_state(L, (2 * pi) / N_val)

    lambda_of_times[N_val] = Dict(a => zeros(n) for a in a_vals)

    filename = "N$(N_val)_L$(L)_IC$(num_initial_conds)"
    plt = plot()

    for a_val in a_vals
        println("N_val: $N_val | a_val: $a_val")

        avrage_spin_dists = zeros(n)

        for init_cond in 1:num_initial_conds
            println("N_val: $N_val | a_val: $a_val | IC: $init_cond / $num_initial_conds")

            spin_chain_A = make_random_state(L) # our S_A

            # Making spin_chain_B to be spin_chain_A with the middle spin modified
            spin_chain_B = copy(spin_chain_A)
            spin_chain_B[div(length(spin_chain_B), 2)] += make_random_spin(epsilon)

            current_spin_dists = zeros(n)

            # Do n pushes 
            for current_n in 1:n
                if N_val == 4
                    J_vec[1] *= (rand() > 0.5) ? -1 : 1 # Randomly choosing signs for Jx and Jy to remove solitons
                    J_vec[2] *= (rand() > 0.5) ? -1 : 1
                end

                spin_chain_A = global_control_evolve(J_vec, spin_chain_A, a_val, tau, J, S_NAUGHT)[end]
                spin_chain_B = global_control_evolve(J_vec, spin_chain_B, a_val, tau, J, S_NAUGHT)[end]

                d_abs = calculate_spin_distence(spin_chain_A, spin_chain_B)
                spin_chain_B = push_back(spin_chain_A, spin_chain_B, epsilon)

                current_spin_dists[current_n] = d_abs
            end
            avrage_spin_dists += current_spin_dists
        end

        avrage_spin_dists ./= num_initial_conds

        lambda_of_times[N_val][a_val] = calculate_lambda_per_time(avrage_spin_dists, epsilon)

        plot!(lambda_of_times[N_val][a_val], label="a = $(a_val)")
    end
    xlabel!("time")
    ylabel!("λ")
    title!("λ(t) for N = $N_val")

    savefig("figs/N$N_val/lambda_per_time_" * filename * ".png")

    open("data/lambda_per_time/N$N_val/" * filename * ".dat", "w") do io
            serialize(io, lambda_of_times[N_val])
        end
end


for N_val in N_vals
    x_min = 50
    filename = "N$(N_val)_L$(L)_IC$(num_initial_conds)_xmin$x_min"
    plt = plot()
    for a_val in a_vals
        plot!([i for i in x_min:length(lambda_of_times[N_val][a_val])], lambda_of_times[N_val][a_val][x_min:end] , label="a = $(a_val)")
    end
    xlabel!("time")
    ylabel!("λ")
    title!("λ(t) for N = $N_val")

    savefig("figs/N$N_val/lambda_per_time_" * filename * ".png")
end

# By looking at the graphs I see that I should be using the values of lambda after 75 (for L = 256)
