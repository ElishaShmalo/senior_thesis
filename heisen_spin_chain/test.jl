L = 32
x = [i for i in 1:L^1.7]

n = Int(round(L^1.7))
aw = 1/32

num_skip = Int(round(n*(1-aw)))

println(x[num_skip+1:n])
