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
using Distributed

# Other files   
include("../utils/make_spins.jl")
include("../utils/general.jl")
include("../utils/dynamics.jl")
include("../utils/lyapunov.jl")
include("../analytics/spin_diffrences.jl")

# Set plotting theme
Plots.theme(:dark)

# General Variables
@everywhere global_L = 256  # number of spins
@everywhere J = 1    # energy factor

# J vector with some randomness
@everywhere J_vec = J .* [1, 1, 1]

# Time to evolve until push back to S_A
@everywhere tau = 1 * J

# number of pushes we are going to do
@everywhere n = global_L

# --- Trying to Replecate Results ---
@everywhere num_initial_conds = 10 # We are avraging over x initial conditions
# a_vals = [round(0.6 + i*0.02, digits=2) for i in 0:20] # 0.6, 0.62, 0.64, 0.66, 0.68, 0.7,
a_vals = [0.5, 0.6, 0.8] # 0.6, 0.62, 0.64, 0.66, 0.68, 0.7,

N_vals = [4, 10]
# N_vals = [2, 3, 4, 6, 9, 10]
# Making individual folders for N_vals

@everywhere epsilon = 0.1

# --- Calculating Lambdas ---

# Our dict for recording results
collected_lambdas = Dict{Int, Dict{Float64, Float64}}() # Int: N_val, Float64: a_val, Float64: avrg lambda
collected_lambda_SEMs = Dict{Int, Dict{Float64, Float64}}() # Int: N_val, Float64: a_val, Float64: standard error on the mean for lambda

for N_val in N_vals
    println("N_val: $N_val")

    @everywhere L = get_nearest(N_val, global_L)

    @everywhere states_evolve_func = evolve_spins_to_time


    @everywhere num_skip = Int((7 * L) / 8) # we only keep the last L/8 time samples so that the initial condition is properly lost

    # Define s_naught to be used during control step
    S_NAUGHT = make_spiral_state(L, (2) / N_val)

    # Initializes results for this N_val
    collected_lambdas[N_val] = Dict(a => 0 for a in a_vals)
    collected_lambda_SEMs[N_val] = Dict(a => 0 for a in a_vals)

    if N_val != 4
        J_vec = J .* [1, 1, 1]
    end

    for a_val in a_vals
        println("N_val: $N_val | a_val: $a_val")
        
        # We will avrage over this later
        @everywhere current_lambdas = zeros(num_initial_conds)

        @sync @distributed for init_cond in 1:num_initial_conds
            local_N_val = $N_val
            println("N_val: $local_N_val | a_val: $a_val | IC: $init_cond / $num_initial_conds")

            spin_chain_A = make_random_state(L) # our S_A

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
                evolved_results = states_evolve_func(spin_chain_A, spin_chain_B, a_val, tau, J, S_NAUGHT)
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
    filepath = "N$N_val/SeveralAs/IC$num_initial_conds/L$L/" * "N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L)"

    open("data/spin_chain_lambdas/" * filepath * ".dat", "w") do io
        serialize(io, collected_lambdas[N_val])
        println("Saved file $filepath")
    end
    open("data/spin_chain_lambdas/" * filepath * "sems.dat", "w") do io
        serialize(io, collected_lambda_SEMs[N_val])
        println("Saved file $(filepath)sems")
    end

    # Make .csv file
    # Extract and sort keys and values
    lambda_dict = collected_lambdas[N_val]
    sems_dict = collected_lambda_SEMs[N_val]
    dict_keys = sort(collect(keys(lambda_dict)))

    # Prepare rows: each row is [aval, lambda, lambda_sem]
    rows = [[aval, lambda_dict[aval], sems_dict[aval]] for aval in dict_keys]

    # Make output CSV path
    csv_path = "data/spin_chain_lambdas/" * filepath * ".csv"

    # Write to CSV with header
    open(csv_path, "w") do io
        writedlm(io, [["aval", "lambda", "lambda_sem"]], ',')  # Header
        writedlm(io, rows, ',')                                # Data rows
    end
end

# Save the plot (if you want all the N_vals on one plot)
plt = plot()
plot_path = "SeveralNs/SeveralAs/IC$num_initial_conds/L$global_L/lambda_per_a_Ns$(join(N_vals))_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$L"
for N_val in N_vals

    L = get_nearest(N_val, global_L)

    filepath = "N$N_val/SeveralAs/IC$num_initial_conds/L$L/" * "N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L)"
    collected_lambdas[N_val] = open("data/spin_chain_lambdas/" * filepath * ".dat", "r") do io
        deserialize(io)
    end
    collected_lambda_SEMs[N_val] = open("data/spin_chain_lambdas/" * filepath * "sems.dat", "r") do io
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
#     filepath = "N$(replace("$N_val", "." => "p"))/N$(N_val)_IC$(num_initial_conds)_L$(get_nearest(N_val, L))"
#     fullpath = "data/spin_chain_lambdas/" * filepath * ".dat"

#     # Load the lambda vals
#     collected_lambdas[N_val] = open(fullpath, "r") do io
#         deserialize(io)
#     end

#     # Load the lambda SEMs
#     collected_lambda_SEMs[N_val] = open("data/spin_chain_lambdas/" * filepath * "sems.dat", "r") do io
#         deserialize(io)
#     end

#     # Extract and sort keys and values
#     lambda_dict = collected_lambdas[N_val]
#     sems_dict = collected_lambda_SEMs[N_val]
#     dict_keys = sort(collect(keys(lambda_dict)))

#     # Prepare rows: each row is [aval, lambda, lambda_sem]
#     rows = [[aval, lambda_dict[aval], sems_dict[aval]] for aval in dict_keys]

#     # Make output CSV path
#     csv_path = "data/spin_chain_lambdas/" * filepath * ".csv"

#     # Write to CSV with header
#     open(csv_path, "w") do io
#         writedlm(io, [["aval", "lambda", "lambda_sem"]], ',')  # Header
#         writedlm(io, rows, ',')                                # Data rows
#     end
# end

