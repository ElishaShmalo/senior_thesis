
using LinearAlgebra

X = [1, 2, 3, 2 * ℯ]

println(sum(map(log, X ./ 2)))
println(map(log, X ./ 2))
