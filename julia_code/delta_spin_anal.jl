# Imports
using Random
using LinearAlgebra
using Plots
using DifferentialEquations
using StaticArrays
using Serialization
using Statistics

# Other files   
include("utils/make_spins.jl")
include("utils/dynamics.jl")
include("analytics/spin_diffrences.jl")

# Set plotting theme
Plots.theme(:dark)
# General Variables
L = 4*64  # number of spins
N = 4
num_init_cond = 1

# Heat map of delta_spins
plt = plot()
a_val = 0.7
results_file_name = "N$N/a_val_" * replace("$a_val", "." => "p") * "_IC$(num_init_cond)" * "_L$L"

delta_spins = open("data/delta_evolved_spins/" * results_file_name * "_avg.dat", "r") do io
    deserialize(io)
end

if typeof(delta_spins) == Vector{Vector{Float64}}
    delta_spins = hcat(delta_spins...)'
end

heatmap!(delta_spins[1:256, 1:256], colorbar_title="δS")

xlabel!("x")
ylabel!("t")
title!("a = $a_val")
display(plt)
