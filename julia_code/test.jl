# Hyzenberg Spin Chain Simulation in Julia
# Doing ferromagnet because it's cooler :)

# Imports
using CSV
using DataFrames

# Replace "path/to/file.csv" with the actual file path
df = CSV.read("path/to/file.csv", DataFrame)

collected_lambdas = Dict{Int, Dict{Float64, Float64}}() # Int: N_val, Float64: a_val, Float64: avrg lambda
collected_lambda_SEMs = Dict{Int, Dict{Float64, Float64}}() # Int: N_val, Float64: a_val, Float64: standard error on the mean for lambda

L = 256

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

