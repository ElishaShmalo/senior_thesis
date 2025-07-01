# In this file we numerically approximate the Lyapunov for diffrent a-vals and N (for the spiral) vals using the tech from Benettin

# Imports
using Random
using LinearAlgebra
using Plots
using DifferentialEquations
using StaticArrays
using Serialization
using Statistics
using DelimitedFiles

# Other files   
include("../utils/make_spins.jl")
include("../utils/general.jl")
include("../utils/dynamics.jl")
include("../utils/lyapunov.jl")
include("../analytics/spin_diffrences.jl")

# Making necessary folders if they dont exist
folder_names_in_order = ["data", "data/spin_chain_lambdas", "figs", "figs/lambda_per_time", "figs/lambda_per_a"]
for folder in folder_names_in_order
    if !isdir(folder)
        mkdir(folder)
    end
end

# Set plotting theme
Plots.theme(:dark)

# General Variables
L = 128  # number of spins
J = 1    # energy factor

# J vector with some randomness
J_vec = J .* [1, 1, 1]

# Time to evolve until push back to S_A
tau = 1 * J

# number of pushes we are going to do
n = L
num_skip = Int((7 * L) / 8) # we only keep the last L/8 time samples so that the initial condition is properly lost

# --- Trying to Replecate Results ---
num_initial_conds = 500 # We are avraging over x initial conditions
a_vals = [0.6 + i*0.02 for i in 0:20] # 0.6, 0.62, 0.64, 0.66, 0.68, 0.7,
trans_a_vals = [0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8]
a_vals = unique(sort(vcat(trans_a_vals, a_vals)))

# N_vals = [2, 3, 6, 9]
N_vals = [4]
# Making individual folders for N_vals
for N_val in N_vals
    if !isdir("data/spin_chain_lambdas/N$N_val")
        mkdir("data/spin_chain_lambdas/N$N_val")
    end
    if !isdir("figs/lambda_per_time/N$N_val")
        mkdir("figs/lambda_per_time/N$N_val")
    end
    if !isdir("figs/lambda_per_a/N$N_val")
        mkdir("figs/lambda_per_a/N$N_val")
    end
end

epsilon = 0.01

# --- Calculating Lambdas ---

# Our dict for recording results
collected_lambdas = Dict{Int, Dict{Float64, Float64}}() # Int: N_val, Float64: a_val, Float64: avrg lambda
collected_lambda_SEMs = Dict{Int, Dict{Float64, Float64}}() # Int: N_val, Float64: a_val, Float64: standard error on the mean for lambda

for N_val in N_vals
    println("N_val: $N_val")

    # Define s_naught to be used during control step
    S_NAUGHT = make_spiral_state(get_nearest(N_val, L), (2) / N_val)

    # Initializes results for this N_val
    collected_lambdas[N_val] = Dict(a => 0 for a in a_vals)
    collected_lambda_SEMs[N_val] = Dict(a => 0 for a in a_vals)

    if N_val != 4
        J_vec = J .* [1, 1, 1]
    end

    for a_val in a_vals
        println("N_val: $N_val | a_val: $a_val")
        
        # We will avrage over this later
        current_lambdas = zeros(num_initial_conds)

        for init_cond in 1:num_initial_conds
            println("N_val: $N_val | a_val: $a_val | IC: $init_cond / $num_initial_conds")

            spin_chain_A = make_random_state(get_nearest(N_val, L)) # our S_A

            # Making spin_chain_B to be spin_chain_A with the middle spin modified
            spin_chain_B = copy(spin_chain_A)
            new_mid_spin_val = spin_chain_B[div(length(spin_chain_B), 2)] + make_random_spin(epsilon)
            spin_chain_B[div(length(spin_chain_B), 2)] = normalize(new_mid_spin_val)
            
            # Will be used to calculate lyop exp
            current_spin_dists = zeros(n)

            # Do n pushes 
            for current_n in 1:n
                # we need to change J_vec outside of the evolve func so that it is the same for S_A and S_B

                # evolve both to time t' = t + tau with control
                evolved_results = random_evolve_spins_to_time(spin_chain_A, spin_chain_B, a_val, tau, J, S_NAUGHT)
                spin_chain_A = evolved_results[1][end]
                spin_chain_B = evolved_results[2][end]

                d_abs = calculate_spin_distence(spin_chain_A, spin_chain_B)
                spin_chain_B = push_back(spin_chain_A, spin_chain_B, epsilon)

                current_spin_dists[current_n] = d_abs
            end
            current_lambdas[init_cond] = calculate_lambda(current_spin_dists[num_skip:end], tau, epsilon, n - num_skip)
        end

        collected_lambdas[N_val][a_val] = mean(current_lambdas)
        collected_lambda_SEMs[N_val][a_val] = std(current_lambdas)/sqrt(length(current_lambdas))

    end

    # Save the results for each N_val sepratly
    filename = "N$N_val/" * "N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(get_nearest(N_val, L))"

    open("data/spin_chain_lambdas/" * filename * ".dat", "w") do io
        serialize(io, collected_lambdas[N_val])
        println("Saved file $filename")
    end
    open("data/spin_chain_lambdas/" * filename * "sems.dat", "w") do io
        serialize(io, collected_lambda_SEMs[N_val])
        println("Saved file $(filename)sems")
    end

    # Make .csv file
    # Extract and sort keys and values
    lambda_dict = collected_lambdas[N_val]
    sems_dict = collected_lambda_SEMs[N_val]
    dict_keys = sort(collect(keys(lambda_dict)))

    # Prepare rows: each row is [aval, lambda, lambda_sem]
    rows = [[aval, lambda_dict[aval], sems_dict[aval]] for aval in dict_keys]

    # Make output CSV path
    csv_path = "data/spin_chain_lambdas/" * filename * ".csv"

    # Write to CSV with header
    open(csv_path, "w") do io
        writedlm(io, [["aval", "lambda", "lambda_sem"]], ',')  # Header
        writedlm(io, rows, ',')                                # Data rows
    end
end

# Save the plot (if you want all the N_vals on one plot)
plt = plot()
plot_name = "lambda_per_a_Ns$(join(N_vals))_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$L"
for N_val in N_vals
    filename = "N$N_val/" * "N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(get_nearest(N_val, L))"
    collected_lambdas[N_val] = open("data/spin_chain_lambdas/" * filename * ".dat", "r") do io
        deserialize(io)
    end
    collected_lambda_SEMs[N_val] = open("data/spin_chain_lambdas/" * filename * "sems.dat", "r") do io
        deserialize(io)
    end
    plot!(
        sort(a_vals), 
        [val for val in values(sort(collected_lambdas[N_val]))], 
        yerror=[val for val in values(sort(collected_lambda_SEMs[N_val]))], 
        marker = :circle, label="N=$N_val")
end

x_vals = range(minimum(a_vals) - 0.005, stop = 1, length = 1000)

plot!(plt, x_vals, log.(x_vals), linestyle = :dash, label = "ln(a)", title="λ(a) for L=$L")

xlabel!("a")
ylabel!("λ")
display(plt)

savefig("figs/lambda_per_a/" * plot_name * ".png")


# Make .csv file
# for N_val in N_vals
#     # Construct file paths
#     filename = "N$(replace("$N_val", "." => "p"))/N$(N_val)_IC$(num_initial_conds)_L$(get_nearest(N_val, L))"
#     fullpath = "data/spin_chain_lambdas/" * filename * ".dat"

#     # Load the lambda vals
#     collected_lambdas[N_val] = open(fullpath, "r") do io
#         deserialize(io)
#     end

#     # Load the lambda SEMs
#     collected_lambda_SEMs[N_val] = open("data/spin_chain_lambdas/" * filename * "sems.dat", "r") do io
#         deserialize(io)
#     end

#     # Extract and sort keys and values
#     lambda_dict = collected_lambdas[N_val]
#     sems_dict = collected_lambda_SEMs[N_val]
#     dict_keys = sort(collect(keys(lambda_dict)))

#     # Prepare rows: each row is [aval, lambda, lambda_sem]
#     rows = [[aval, lambda_dict[aval], sems_dict[aval]] for aval in dict_keys]

#     # Make output CSV path
#     csv_path = "data/spin_chain_lambdas/" * filename * ".csv"

#     # Write to CSV with header
#     open(csv_path, "w") do io
#         writedlm(io, [["aval", "lambda", "lambda_sem"]], ',')  # Header
#         writedlm(io, rows, ',')                                # Data rows
#     end
# end

