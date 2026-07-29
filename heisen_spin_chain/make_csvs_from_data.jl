
using CSV, DataFrames

include("utils/general.jl")

N_vals = [2, 3, 4, 6, 9, 10]
a_vals = [round(0.6 + i*0.02, digits=2) for i in 0:20]

num_initial_conds = 500
global_L = 256

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
