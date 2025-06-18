
X = [[1, 2, 3], [2, 3, 4]]
Y = [1]

println(map(normalize, X))
println(X ./ map(norm, X))
