
using LinearAlgebra

N = 10

X = [i for i in 1:100]

function mod_v(v)
    temp = X[length(X)-(N-1): length(X)]
    X[N+1:length(X)] = X[1:length(X)-(N)]
    X[1:N] = temp
end

println(X)
mod_v(X)
println(X)

Y = [1, 2, 3, 4]
Y[1:2] = Y[3:4]
