# Hyzenberg Spin Chain Simulation in Julia
# Doing ferromagnet because it's cooler :)

# Imports
using Random
using LinearAlgebra
using Plots

prefs_dir = "/Users/elishashmalo/.julia/prefs/"
mkpath(prefs_dir)
write(joinpath(prefs_dir, "GR.toml"), "use_jll_binary = false\n")

using PyPlot

# Set the interactive backend for matplotlib
matplotlib.use("TkAgg")

x = 1:10
y = rand(10)

plot(x, y)
xlabel("x-axis")
ylabel("y-axis")
title("My first PyPlot!")

show()

# Set plotting theme
theme(:dark)

# General Variables
L = 4 * 60  # number of spins
J = 1       # energy factor

# J vector with some randomness
J_vec = J .* [rand([-1, 1]), rand([-1, 1]), 1]

# Time step for evolution
Tau_F = 1 / J

# Evolve until
t = L

# --- Defining Initial States ---

# Random spin state
function make_random_state(n::Int=L)
    state = [normalize(rand(3)) for _ in 1:n]
    return reduce(hcat, state)'  # make it an n x 3 array
end

# Uniform spin state along z
function make_uniform_state(n::Int=L, z_dir::Int=1)
    state = [[0.0, 0.0, z_dir] for _ in 1:n]
    return reduce(hcat, state)'
end

# Spiral spin state
function make_spiral_state(n::Int=L, spiral_angle::Float64=π/4, phi::Float64=0.0)
    state = [[0.0, cos(i * spiral_angle + phi), sin(i * spiral_angle + phi)] for i in 0:(n-1)]
    return reduce(hcat, state)'
end

# --- Equation of Motion ---

function differential_s(state::Matrix{Float64})
    n = size(state, 1)
    dstate_dt = [cross(-(J * state[mod1(i-1, n), :] + J * state[mod1(i+1, n), :]), state[i, :]) for i in 1:n]
    return reduce(hcat, dstate_dt)'
end

function evolve_differential(state::Matrix{Float64}, dstate_dt::Matrix{Float64}, dt::Float64)
    return state .+ dstate_dt .* dt
end

# --- Testing differential_s ---

s_naught = make_spiral_state(L)

println(differential_s(s_naught))

# Define s_naught as a constant
const S_NAUGHT = make_spiral_state(L)

# --- Control Push ---

function control_push(spin::Matrix{Float64}, a::Float64)
    numerator = (1-a) .* S_NAUGHT .+ a .* spin
    denominator = sqrt.(sum(numerator.^2, dims=2))
    return numerator ./ denominator
end

# --- Test Control Push ---

test_spin = make_random_state(L)

println(test_spin[1, :], norm(test_spin[1, :]))
controlled_test = control_push(test_spin, 0.0)
println(controlled_test[1, :], norm(controlled_test[1, :]))

# --- Weighted Spin Difference ---

function weighted_spin_difference(spin_chain::Matrix{Float64}, s_0::Matrix{Float64})
    delta_S = sqrt.(sum((spin_chain .- s_0).^2, dims=2))
    return sum(delta_S) / length(delta_S)
end

# --- Test Weighted Spin Difference ---

println("No control: ", weighted_spin_difference(test_spin, S_NAUGHT))
println("Push of 1/2: ", weighted_spin_difference(control_push(test_spin, 0.5), S_NAUGHT))
println("Push of 0: ", weighted_spin_difference(control_push(test_spin, 0.0), S_NAUGHT))

# --- First Tests of Dynamics ---

a_vals = [0.6, 0.7, 0.716]

original_random = make_random_state()

S_diffs = Dict{Float64, Vector{Float64}}()

for a_val in a_vals
    t = 0.0
    current_S_diffs = Float64[]

    push!(current_S_diffs, weighted_spin_difference(original_random, S_NAUGHT))

    current_state = deepcopy(original_random)

    while t < L-1
        differential = differential_s(current_state)
        current_state = evolve_differential(current_state, differential, Tau_F)
        current_state = control_push(current_state, a_val)
        push!(current_S_diffs, weighted_spin_difference(current_state, S_NAUGHT))

        t += Tau_F
    end
    S_diffs[a_val] = current_S_diffs
end

# --- Plotting the Dynamics ---

ts = [i * Tau_F for i in 0:length(S_diffs[a_vals[1]])-1]

plt = plot()
for a_val in a_vals
    plot!(ts, S_diffs[a_val], label="a = $(a_val)")
end

xlabel!("time")
ylabel!("S_diff")
title!("Spin Dynamics")
display(plt)
