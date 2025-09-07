# In this file we numerically approximate the Lyapunov for diffrent a-vals and N (for the spiral) vals using the tech from Benettin
using Distributed
using SlurmClusterManager

println("We are adding $(SlurmManager()) workers")
addprocs(SlurmManager())

@everywhere begin
    println("Hello from worker $(myid()) on host $(gethostname())")
end

# Imports
@everywhere using Random, LinearAlgebra, Plots, DifferentialEquations, Serialization, Statistics, DelimitedFiles, SharedArrays, CSV, DataFrames

# Other files   
@everywhere include("../utils/make_spins.jl")
@everywhere include("../utils/general.jl")
@everywhere include("../utils/dynamics.jl")
@everywhere include("../utils/lyapunov.jl")
@everywhere include("../analytics/spin_diffrences.jl")

@time begin

# Set plotting theme
Plots.theme(:dark)

# General Variables
# @everywhere num_unit_cells_vals = [8, 16, 32, 64]
@everywhere num_unit_cells_vals = [128]
# @everywhere num_unit_cells_vals = [8]
@everywhere J = 1    # energy factor

# J vector with some randomness
@everywhere J_vec = J .* [1, 1, 1]

# Time to evolve until push back to S_A
@everywhere tau = 1 * J

# --- Trying to Replecate Results ---
@everywhere num_initial_conds = 1000 # We are avraging over x initial conditions
a_vals = sort(union([round(0.6 + i*0.01, digits=2) for i in 0:9])) # general a_vals
# a_vals = [0.6, 0.7, 0.8] # 0.6, 0.62, 0.64, 0.66, 0.68, 0.7,

@everywhere epsilon = 0.1

@everywhere N_val = 4

z_val = 1.6
z_val_name = replace("$z_val", "." => "p")

# --- Calculating Lambdas ---

# Our dict for recording results
collected_lambdas = Dict{Int, Dict{Float64, Float64}}() # Int: L_val, Float64: a_val, Float64: avrg lambda
collected_lambda_SEMs = Dict{Int, Dict{Float64, Float64}}() # Int: L_val, Float64: a_val, Float64: standard error on the mean for lambda

for num_unit_cells in num_unit_cells_vals
    L = num_unit_cells * N_val
    println("L_val: $L")

    # number of pushes we are going to do
    n = Int(round(L^z_val))

    states_evolve_func = random_evolve_spins_to_time

    num_skip = Int(round((7 * n) / 8)) # we only keep the last L/8 time samples so that the initial condition is properly lost

    # Define s_naught to be used during control step
    S_NAUGHT = make_spiral_state(L, (2) / N_val)

    # Initializes results for this N_val
    collected_lambdas[L] = Dict(a => 0 for a in a_vals)
    collected_lambda_SEMs[L] = Dict(a => 0 for a in a_vals)

    for a_val in a_vals
        println("L_val: $L | a_val: $a_val")
        a_val_name = replace("$a_val", "." => "p")
        # We will avrage over this later
        current_lambdas = zeros(Float64, num_initial_conds)

        # define the variables for the workers to use
        let a_val=a_val, L=L, n=n, S_NAUGHT=S_NAUGHT, num_initial_conds=num_initial_conds, states_evolve_func=states_evolve_func, num_skip=num_skip
            current_lambdas = @distributed (vcat) for init_cond in 1:num_initial_conds
                println("L_val: $L | a_val: $a_val | IC: $init_cond / $num_initial_conds")

                J_vec = J .* [1, 1, 1]

                spin_chain_A = make_random_state(L) # our S_A

                # Making spin_chain_B to be spin_chain_A with the middle spin modified
                spin_chain_B = copy(spin_chain_A)
                new_mid_spin_val = spin_chain_B[div(length(spin_chain_B), 2)] + make_random_spin(epsilon)
                spin_chain_B[div(length(spin_chain_B), 2)] = normalize(new_mid_spin_val)
                
                # Will be used to calculate lyop exp
                current_spin_dists = zeros(n)
                current_sdiffs = zeros(n)

                # Do n pushes 
                for current_n in 1:n
                    # we need to change J_vec outside of the evolve func so that it is the same for S_A and S_B

                    # evolve both to time t' = t + tau with control
                    evolved_results = states_evolve_func(J_vec, spin_chain_A, spin_chain_B, a_val, tau, J, S_NAUGHT)
                    spin_chain_A = evolved_results[1][end]
                    spin_chain_B = evolved_results[2][end]

                    current_spin_dists[current_n] = calculate_spin_distence(spin_chain_A, spin_chain_B)
                    spin_chain_B = push_back(spin_chain_A, spin_chain_B, epsilon)

                    current_sdiffs[current_n] = weighted_spin_difference(spin_chain_A, S_NAUGHT)
                end
                lambda = calculate_lambda(current_spin_dists[num_skip:end], tau, epsilon, n - num_skip)
                
                sample_filepath = "data/spin_dists_per_time/N$N_val/a$a_val_name/IC1/L$L/N$(N_val)_a$(a_val_name)_IC1_L$(L)_z$(z_val_name)_sample$(init_cond)"
                make_path_exist(sample_filepath)
                df = DataFrame("t" => 1:n, "lambda" => calculate_lambda_per_time(current_spin_dists, epsilon), "delta_s" => current_sdiffs)
                CSV.write(sample_filepath, df)

                # return the calculated lambda
                [lambda]
            end
        end

        collected_lambdas[L][a_val] = mean(current_lambdas)
        collected_lambda_SEMs[L][a_val] = std(current_lambdas)/sqrt(length(current_lambdas))

    end

    # Save the results for each L_val sepratly
    filepath = "N$N_val/SeveralAs/IC$num_initial_conds/L$L/" * "N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L)_z$(z_val_name)"


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

# --- Save CSV ---
print("Saving CSVs")
for L in num_unit_cells_vals * N_val
    L = Int(L)
    filepath = "N$(N_val)/SeveralAs/IC$num_initial_conds/L$L/" * "N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L)_z$(z_val_name)"
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
for L in num_unit_cells_vals * N_val
    push!(col_names, "lambda_L=$L")
    push!(col_names, "SEM_L=$L")
end

# Create the data rows
cols = Vector{Vector{Union{Missing, Float64}}}()
push!(cols, sorted_a_vals)
    
for L in num_unit_cells_vals * N_val
    L = Int(L)
    push!(cols, [collected_lambdas[L][k] for k in sorted_a_vals])
    push!(cols, [collected_lambda_SEMs[L][k] for k in sorted_a_vals])
end 

# Convert to DataFrame and save
df = DataFrame(cols, Symbol.(col_names))
csv_path = "data/spin_chain_lambdas/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/lambda_per_a_N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_$(join(num_unit_cells_vals * N_val))_z$(z_val_name).csv"
mkpath(dirname(csv_path))
CSV.write(csv_path, df)
println("Saved CsV: $csv_path")

end