
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
include("utils/lyapunov.jl")
include("analytics/spin_diffrences.jl")

# General Variables
L = 4*64  # number of spins

epsilon = 0.1

spin_chain_A = make_random_state(L) # Essentially our S_A

# Making spin_chain_B to be spin_chain_A with the middle spin modified
to_add = make_uniform_state(L, 0)
to_add[div(length(to_add), 2)] = make_random_spin(epsilon)
spin_chain_B = spin_chain_A .+ to_add

println(map(norm, spin_chain_B - spin_chain_A)[div(length(to_add), 2)])

X = [[1, 0], [0, 2]]
Y = [[2, 1], [3, 2]]

map(dot, X, Y)

