
using LinearAlgebra

num_t = 5
num_x = 4
num_u = 2

X = Array{Float64}(undef, num_t, num_x, num_u)
println(X[1, :, :])
println(X[1, :, :][1,:])