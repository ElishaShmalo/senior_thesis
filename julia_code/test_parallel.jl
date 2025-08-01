# In this file we numerically approximate the Lyapunov for diffrent a-vals and N (for the spiral) vals using the tech from Benettin
using Distributed

# Imports
@everywhere using Random, LinearAlgebra, Plots, DifferentialEquations, StaticArrays, Serialization, Statistics, DelimitedFiles, SharedArrays, CSV, DataFrames

# Other files   
@everywhere include("utils/make_spins.jl")
@everywhere include("utils/general.jl")
@everywhere include("utils/dynamics.jl")
@everywhere include("utils/lyapunov.jl")
@everywhere include("analytics/spin_diffrences.jl")

# Set plotting theme
Plots.theme(:dark)

# General Variables
@everywhere L_vals = [32, 64]  # number of spins
@everywhere J = 1    # energy factor

# J vector with some randomness
@everywhere J_vec = J .* [1, 1, 1]

# Time to evolve until push back to S_A
@everywhere tau = 1 * J

# --- Trying to Replecate Results ---
@everywhere num_initial_conds = 50 # We are avraging over x initial conditions
# a_vals = [round(0.6 + i*0.01, digits=2) for i in 0:20] # 0.6, 0.62, 0.64, 0.66, 0.68, 0.7,
# trans_a_vals = [0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8]
# a_vals = sort(union(a_vals, trans_a_vals))
a_vals = [0.5, 0.7, 0.8]

@everywhere epsilon = 0.1

# --- Calculating Lambdas ---

# Our dict for recording results
collected_lambdas = Dict{Int, Dict{Float64, Float64}}() # Int: L_val, Float64: a_val, Float64: avrg lambda
collected_lambda_SEMs = Dict{Int, Dict{Float64, Float64}}() # Int: L_val, Float64: a_val, Float64: standard error on the mean for lambda

for L in L_vals
    println("L_val: $L")

    # number of pushes we are going to do
    n = L

    states_evolve_func = random_evolve_spins_to_time

    num_skip = Int(round((7 * L) / 8)) # we only keep the last L/8 time samples so that the initial condition is properly lost

    # Define s_naught to be used during control step
    S_NAUGHT = make_spiral_state(L, (2) / 4)

    # Initializes results for this N_val
    collected_lambdas[L] = Dict(a => 0 for a in a_vals)
    collected_lambda_SEMs[L] = Dict(a => 0 for a in a_vals)

    for a_val in a_vals
        println("L_val: $L | a_val: $a_val")
        
        # We will avrage over this later
        current_lambdas = SharedArray{Float64}(num_initial_conds)

        # define the variables for the workers to use
        let a_val=a_val, L=L, n=n, S_NAUGHT=S_NAUGHT, num_initial_conds=num_initial_conds, states_evolve_func=states_evolve_func, num_skip=num_skip
            @sync @distributed for init_cond in 1:num_initial_conds

                println("L_val: $L | a_val: $a_val | IC: $init_cond / $num_initial_conds")

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
                    evolved_results = states_evolve_func(copy(J_vec), spin_chain_A, spin_chain_B, a_val, tau, J, S_NAUGHT)
                    spin_chain_A = evolved_results[1][end]
                    spin_chain_B = evolved_results[2][end]

                    d_abs = calculate_spin_distence(spin_chain_A, spin_chain_B)
                    spin_chain_B = push_back(spin_chain_A, spin_chain_B, epsilon)

                    current_spin_dists[current_n] = d_abs
                end
                current_lambdas[init_cond] = calculate_lambda(current_spin_dists[num_skip:end], tau, epsilon, n - num_skip)
            end
        end

        collected_lambdas[L][a_val] = mean(current_lambdas)
        collected_lambda_SEMs[L][a_val] = std(current_lambdas)/sqrt(length(current_lambdas))

    end

    # Save the results for each L_val sepratly
    filepath = "N4/SeveralAs/IC$num_initial_conds/L$L/" * "N4_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L)"


    make_data_file("data/spin_chain_lambdas/" * filepath * ".dat", collected_lambdas[L])
    make_data_file("data/spin_chain_lambdas/" * filepath * "sems.dat", collected_lambda_SEMs[L])

    # Make .csv file
    # Extract and sort keys and values
    lambda_dict = collected_lambdas[L]
    sems_dict = collected_lambda_SEMs[L]
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

# Save the plot
println("Making Plot")
plt = plot()
plot_path = "N4/SeveralAs/IC$num_initial_conds/SeveralLs/lambda_per_a_N4_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(L_vals))"

for L in L_vals

    filepath = "N4/SeveralAs/IC$num_initial_conds/L$L/" * "N4_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L)"
    collected_lambdas[L] = open("data/spin_chain_lambdas/" * filepath * ".dat", "r") do io
        deserialize(io)
    end
    collected_lambda_SEMs[L] = open("data/spin_chain_lambdas/" * filepath * "sems.dat", "r") do io
        deserialize(io)
    end
    plot!(
        sort(a_vals), 
        [val for val in values(sort(collected_lambdas[L]))], 
        yerror=[val for val in values(sort(collected_lambda_SEMs[L]))], 
        marker = :circle, label="L=$L")
end

x_vals = range(minimum(a_vals) - 0.005, stop = 1, length = 1000)

plot!(plt, x_vals, log.(x_vals), linestyle = :dash, label = "ln(a)", title="λ(a) for N=4")

xlabel!("a")
ylabel!("λ")
display(plt)

mkpath(dirname("figs/lambda_per_a/" * plot_path))
savefig("figs/lambda_per_a/" * plot_path * ".png")
println("Saved Plot: $("figs/lambda_per_a/" * plot_path * ".png")")

# Save CSV
print("Saving CSVs")
for L in L_vals

    filepath = "N4/SeveralAs/IC$num_initial_conds/L$L/" * "N4_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L)"
    collected_lambdas[L] = open("data/spin_chain_lambdas/" * filepath * ".dat", "r") do io
        deserialize(io)
    end
    collected_lambda_SEMs[L] = open("data/spin_chain_lambdas/" * filepath * "sems.dat", "r") do io
        deserialize(io)
    end
end

# Sort a_vals to ensure correct row order
sorted_a_vals = sort(a_vals)

# Prepare the header
col_names = ["a_val"]
for L in L_vals
    push!(col_names, "lambda_L=$L")
    push!(col_names, "SEM_L=$L")
end

# Create the data rows
cols = Vector{Vector{Union{Missing, Float64}}}()
push!(cols, sorted_a_vals)
    
for L in L_vals
    push!(cols, [collected_lambdas[L][k] for k in sorted_a_vals])
    push!(cols, [collected_lambda_SEMs[L][k] for k in sorted_a_vals])
end 

# Convert to DataFrame and save
df = DataFrame(cols, Symbol.(col_names))
csv_path = "data/spin_chain_lambdas/SeveralNs/SeveralAs/IC$num_initial_conds/SeveralLs/lambda_per_a_N4_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_$(join(L_vals)).csv"
mkpath(dirname(csv_path))
CSV.write(csv_path, df)
println("Saved CsV: $csv_path")