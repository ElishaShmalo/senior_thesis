using Plots

# Define the range
x_min = 0
x_max = 256
x_center = (x_min + x_max) / 2

# Define offset so that center of tanh(x - offset) is at midpoint
offset = x_center

# Define the x values
x = range(x_min, x_max, length=1000)

# Define the functions
tanh_curve = tanh.(x .- offset)
sech_curve = 1 ./ cosh.(x .- offset)

# Plot
plot(x, tanh_curve, label="tanh(j): s", lw=2)
plot!(x, sech_curve, label="sech(j): phi", lw=2)
xlabel!("j")
ylabel!("Value")
title!("AttemptedInitialCond")
savefig("attempted_initial_cond.png")