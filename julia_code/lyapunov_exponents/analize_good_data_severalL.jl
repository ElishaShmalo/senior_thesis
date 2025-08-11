
# Imports
using Random, LinearAlgebra, Plots, DifferentialEquations, Serialization, Statistics, DelimitedFiles, SharedArrays, CSV, DataFrames

# Other files   
include("../utils/make_spins.jl")
include("../utils/general.jl")
include("../utils/dynamics.jl")
include("../utils/lyapunov.jl")
include("../analytics/spin_diffrences.jl")


# Set plotting theme
Plots.theme(:dark)

# General Variables
# @everywhere num_unit_cells_vals = [8, 16, 32, 64, 128]
# @everywhere num_unit_cells_vals = [8, 16, 32]
num_unit_cells_vals = [8, 16, 32, 64]

# --- Trying to Replecate Results ---
num_initial_conds = 1000 # We are avraging over x initial conditions
a_vals = [round(0.6 + i*0.01, digits=2) for i in 0:25] # general a_vals
# a_vals = [0.6, 0.7, 0.8] # 0.6, 0.62, 0.64, 0.66, 0.68, 0.7,
# a_vals = [0.75, 0.7525, 0.755, 0.7575, 0.76, 0.7625, 0.765, 0.7675, 0.77] # trans a_vals

epsilon = 0.1

N_val = 4

# Our dict for recording results
collected_lambdas = Dict{Int, Dict{Float64, Float64}}() # Int: L_val, Float64: a_val, Float64: avrg lambda
collected_lambda_SEMs = Dict{Int, Dict{Float64, Float64}}() # Int: L_val, Float64: a_val, Float64: standard error on the mean for lambda

# Load in avraged results
for L in num_unit_cells_vals * N_val
    L = Int(L)
    filepath = "N$(N_val)/SeveralAs/IC$num_initial_conds/L$L/" * "N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L)"
    collected_lambdas[L] = open("data/spin_chain_lambdas/" * filepath * ".dat", "r") do io
        deserialize(io)
    end
    collected_lambda_SEMs[L] = open("data/spin_chain_lambdas/" * filepath * "sems.dat", "r") do io
        deserialize(io)
    end
end

# Colapsing varience plot
# Create plot
plt = plot(
    title="Colapssing λ(a) for N=$N_val",
    xlabel="a",
    ylabel="Scaled Var(λ)"
)
a_crit = 0.76
nu = 1.1
# Plot data for each L
for L in num_unit_cells_vals * N_val
    L = Int(L)
    plot!(plt, (a_vals .- a_crit) .* L^(1/nu), [val for val in values(sort(collected_lambda_SEMs[L]))] * sqrt(num_initial_conds-1),
          label="L=$L",
          linestyle=:solid,
          markersize=5,
          linewidth=1,
          marker = :circle)
end

var_plot_path = "figs/lambda_per_a/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/lambda_per_a_N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(N_val .* num_unit_cells_vals))_VarsCollapsed.png"
make_path_exist(var_plot_path)
savefig(var_plot_path)
println("Saved Plot: $(var_plot_path)")
