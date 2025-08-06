using LinearAlgebra

include("analytics/spin_diffrences.jl")

A = [[1., 1., 0.], [1., 0., 0.]]
B = [[1., 0., 1.], [0., -1., 0.]]
@time begin
    println(weighted_spin_difference(A, B))
end

@time begin
   println(weighted_spin_difference2(A, B)) 
end

