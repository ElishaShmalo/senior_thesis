# Hyzenberg Spin Chain Simulation in Julia
# Doing ferromagnet because it's cooler :)

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

J_vec = [1, 1, 1]


L = get_nearest(N, 256)
S_0 = make_spiral_state(L, 2 * π / N)

# println(get_diffrential_test(S_0))
N = 3
V_1 = [0, cos((2 * π / N)* 2), sin((2 * π / N)* 2)]
V_2 = [0, cos((2 * π / N)* 1), sin((2 * π / N)* 1)] .+ [0, cos((2 * π / N)* 3), sin((2 * π / N)* 3)]

println(cross(V_1, V_2))

println(typeof(V_1)," ", typeof(V_2)," ", typeof(cross(V_1, V_2))," ", typeof((2 * π / N)* 2))