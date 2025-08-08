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
N_val = 10
num_init_cond = 1

Js_rand = 0

# Heat map of delta_spins
plt = plot()
a_val = 1.0
aval_path = "$(replace("$a_val", "." => "p"))"[1:3]

# results_file_name = "N$(N_val)/a$(aval_path)/IC$(num_init_cond)/L$L/N$(N_val)_a" * replace("$a_val", "." => "p") * "_IC$(num_init_cond)_L$(L)_rand$Js_rand"
results_file_path = "N4/a0p7/IC1/L256/N4_a0p72_IC1_L256_rand0_seksolNumOff40_EppOff0p01_true_rand_avg.dat" # 

delta_spins = open("data/delta_evolved_spins/" * results_file_path, "r") do io
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




# prep_save_plot("figs/delta_spin_heatmaps/$(results_file_path).png")
# savefig("figs/delta_spin_heatmaps/$(results_file_path).png")

# println(readdir("data/delta_evolved_spins/"*"N$(N_val)/a$(aval_path)/IC$(num_init_cond)/L$(get_nearest(N_val, L))/"))
