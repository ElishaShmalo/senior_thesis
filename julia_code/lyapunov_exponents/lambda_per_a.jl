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
tau = 1 * J

# number of pushes we are going to do
n = L
num_skip = 75 # we skip 75 of the first n pushes to get a stable result (we chose 75 from our results in "lambda_per_time.js)

# --- Trying to Replecate Results ---
num_initial_conds = 5 # We are avraging over x initial conditions
a_vals = [0.01, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.68, 0.7, 0.71, 0.725, 0.75, 0.78, 0.8, 0.85, 0.875, 0.9]
# N_vals = [3, 4, 6]
N_vals = [4]

epsilon = 0.01

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

            spin_chain_A = make_random_state(L) # our S_A

            # Making spin_chain_B to be spin_chain_A with the middle spin modified
            spin_chain_B = copy(spin_chain_A)
            new_mid_spin_val = spin_chain_B[div(length(spin_chain_B), 2)] + make_random_spin(epsilon)
            spin_chain_B[div(length(spin_chain_B), 2)] = map(normalize, new_mid_spin_val)

            current_spin_dists = zeros(n)

            # Do n pushes 
            for current_n in 1:n
                # evolve both to time t' = t + tau with control
                spin_chain_A = state_evolve_func(spin_chain_A, a_val, tau, J, S_NAUGHT)[end]
                spin_chain_B = state_evolve_func(spin_chain_B, a_val, tau, J, S_NAUGHT)[end]

                d_abs = calculate_spin_distence(spin_chain_A, spin_chain_B)
                spin_chain_B = push_back(spin_chain_A, spin_chain_B, epsilon)

                current_spin_dists[current_n] = d_abs
            end
            current_lambdas[init_cond] = calculate_lambda(current_spin_dists[num_skip:end], tau, epsilon, n - num_skip)
        end

        collected_lambdas[N_val][a_val] = mean(current_lambdas)
    end

    filename = "N$(replace("$N_val", "." => "p"))/" * "N$(N_val)_IC$(num_initial_conds)_L$(L)"

    open("data/spin_chain_lambdas/" * filename * ".dat", "w") do io
        serialize(io, collected_lambdas[N_val])
        println("Saved file $filename")
    end
end

plt = plot()
plot_name = "lambda_per_a_Ns$(join(N_vals))_IC$(num_initial_conds)_L$L"
for N_val in N_vals
    filename = "N$(replace("$N_val", "." => "p"))/" * "N$(N_val)_IC$(num_initial_conds)_L$(L)"
    collected_lambdas[N_val] = open("data/spin_chain_lambdas/" * filename * ".dat", "r") do io
        deserialize(io)
    end
    plot!(sort(a_vals), [val for val in values(sort(collected_lambdas[N_val]))], marker = :circle, label="N=$N_val")
end

x_vals = range(0.01, stop = 1, length = 1000)

plot!(plt, x_vals, log.(x_vals), linestyle = :dash, label = "ln(a)")

xlabel!("a")
ylabel!("λ")
display(plt)

savefig("figs/" * plot_name * ".png")
