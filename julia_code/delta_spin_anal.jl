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
include("utils/general.jl")
include("utils/dynamics.jl")
include("analytics/spin_diffrences.jl")

# Set plotting theme
Plots.theme(:dark)
# General Variables
L = 4*64  # number of spins
N_val = 4
num_init_cond = 1

rand_j = 2
if N_val != 4 
    rand_j = 0
end

# Heat map of delta_spins
plt = plot()
a_val = 0.76
results_file_name = "N$N_val/N_$(N_val)_a_val_" * replace("$a_val", "." => "p") * "_IC$(num_init_cond)" * "_L$(get_nearest(N_val, L))"

if rand_j == 0 && N_val == 4
    results_file_name = results_file_name * "_nonrand"
elseif rand_j == 2 && N_val == 4
    results_file_name = results_file_name * "_semirand"
end

delta_spins = open("data/delta_evolved_spins/" * results_file_name * "_avg.dat", "r") do io
    deserialize(io)
end

if typeof(delta_spins) == Vector{Vector{Float64}}
    delta_spins = hcat(delta_spins...)'
end

heatmap!(delta_spins, colorbar_title="δS", c=:thermal)

xlabel!("x")
ylabel!("t")
title!("N = $N_val | a = $a_val | IC = $num_init_cond | L = $(get_nearest(N_val, L))")
display(plt)








savefig("figs/delta_spin_heatmaps/$results_file_name.png")
