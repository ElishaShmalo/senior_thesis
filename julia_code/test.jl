
using LinearAlgebra

num_t = 5
num_x = 4
num_u = 2
X = Float64[j for i in 1:4, j in 1:4]

println(mean(X, dims=2))