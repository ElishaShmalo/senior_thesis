
using CSV, DataFrames

include("utils/general.jl")

N_vals = [2, 3, 4, 6, 9, 10]
a_vals = [round(0.6 + i*0.02, digits=2) for i in 0:20]

num_initial_conds = 500
global_L = 256

collected_lambdas = Dict{Int, Dict{Float64, Float64}}() # Int: N_val, Float64: a_val, Float64: avrg lambda
collected_lambda_SEMs = Dict{Int, Dict{Float64, Float64}}() # Int: N_val, Float64: a_val, Float64: standard error on the mean for lambda

for N_val in N_vals

    L = get_nearest(N_val, global_L)

    filepath = "N$N_val/SeveralAs/IC$num_initial_conds/L$L/" * "N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L)"
    collected_lambdas[N_val] = open("data/spin_chain_lambdas/" * filepath * ".dat", "r") do io
        deserialize(io)
    end
    collected_lambda_SEMs[N_val] = open("data/spin_chain_lambdas/" * filepath * "sems.dat", "r") do io
        deserialize(io)
    end
end

# Sort a_vals to ensure correct row order
sorted_a_vals = sort(a_vals)

# Prepare the header
col_names = ["a_val"]
for N in N_vals
    push!(col_names, "lambda_N=$N")
    push!(col_names, "SEM_N=$N")
end

println(length(col_names))

# Create the data rows
cols = Vector{Vector{Union{Missing, Float64}}}()
push!(cols, sorted_a_vals)

    
for N in N_vals
    push!(cols, [collected_lambdas[N][k] for k in sorted_a_vals])
    push!(cols, [collected_lambda_SEMs[N][k] for k in sorted_a_vals])
end 

# Convert to DataFrame and save
df = DataFrame(cols, Symbol.(col_names))
mkpath(dirname("data/spin_chain_lambdas/SeveralNs/SeveralAs/IC$num_initial_conds/L$global_L/lambda_per_a_Ns$(join(N_vals))_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$global_L.csv"))
CSV.write("data/spin_chain_lambdas/SeveralNs/SeveralAs/IC$num_initial_conds/L$global_L/lambda_per_a_Ns$(join(N_vals))_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$global_L.csv", df)

using Plots


# Load the CSV you just saved
filename = "data/spin_chain_lambdas/SeveralNs/SeveralAs/IC$num_initial_conds/L$global_L/lambda_per_a_Ns$(join(N_vals))_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$global_L.csv"
df = CSV.read(filename, DataFrame)

# Plot setup
plt = plot(title = "a_val vs Lambda for different N",
     xlabel = "a_val",
     ylabel = "Lambda",
     legend=:topright)

# Plot one curve per N_val
for N in N_vals
    if N == 4
        continue
    end
    col_name = Symbol("lambda_N=$N")
    # Extract x and y
    x = df.a_val
    y = df[!, col_name]
    plot!(x, y, label = "N = $N")
end

display(plt)
