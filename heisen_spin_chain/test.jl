L = 32
x = [[i i+1] for i in 1:L]

print(sum(x))
print(reduce(+, x))
