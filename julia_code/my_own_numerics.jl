# Hyzenberg Spin Chain Simulation in Julia
# Doing ferromagnet because it's cooler :)

# Imports
using Random
using LinearAlgebra
using Plots

# Set plotting theme
Plots.theme(:dark)

# General Variables
L = 4 * 60  # number of spins
J = 1       # energy factor

# J vector with some randomness
J_vec = J .* normalize([rand([-1, 1]), rand([-1, 1]), 1])

# Time step for evolution
Tau_F = 1 / J

# Number of spins pushed each local control push
local_N_push = div(L, 10)
local_control_index = 1

# Evolve until
t = L

# --- Defining functions to make Initial States ---

# Random spin state
function make_random_state(n::Int=L)
    return [normalize(rand(3)) for _ in 1:n]
end

# Uniform spin state along z
function make_uniform_state(n::Int=L, z_dir::Int=1)
    return [[0.0, 0.0, z_dir] for _ in 1:n]
end

# Spiral spin state
function make_spiral_state(n::Int=L, spiral_angle::Float64=π/4, phi::Float64=0.0)
    return [[0.0, cos(i * spiral_angle + phi), sin(i * spiral_angle + phi)] for i in 0:(n-1)]
end

# --- Equation of Motion ---

function differential_s(state::Vector{Vector{Float64}}, periodic=false)
    n = size(state, 1)

    to_cross = [[0., 0., 0.] for _ in 1:n]

    for i in 1:n
        prev = (i == 1)    ? (periodic ? state[end] : [0.,0.,0.]) : state[i - 1]
        next = (i == n)    ? (periodic ? state[1]  : [0.,0.,0.]) : state[i + 1]
        to_cross[i] = -((prev .* J_vec) + (next .* J_vec))
    end

    differential = [cross(to_cross[i], state[i]) for i in 1:n] 

    return differential
end

function evolve_differential(state::Matrix{Float64}, dstate_dt::Matrix{Float64}, dt::Float64)
    return state .+ dstate_dt .* dt
end

# --- Testing differential_s ---
# Define s_naught as a constant
S_NAUGHT = make_spiral_state(L)

# --- Control Push ---
function global_control_push(state::Vector{Vector{Float64}}, a::Float64)
    numerator = ((1-a) .* S_NAUGHT) .+ (a .* state)
    denominator = map(norm, numerator)
    return numerator ./ denominator
end

function local_control_push(state::Vector{Vector{Float64}}, a::Float64, N::Int64=N_push, start_index::Int64=1)
    # Returns a new state with a control push done to the N spins after (and including) start index in "state"
    local_indexes = [((start_index+i-1) % length(S_NAUGHT)) + 1 for i in 0:N_push-1]
    
    numerator = ((1-a) .* S_NAUGHT[local_indexes]) .+ (a .* state[local_indexes])
    denominator = map(norm, numerator)

    local_pushed_state = copy(state)
    local_pushed_state[local_indexes] = numerator ./ denominator

    return local_pushed_state
end

# --- Test Control Push ---

test_spin = make_random_state(L)

println(test_spin[1, :], norm(test_spin[1, :]))

controlled_test = global_control_push(test_spin, 0.0)
println(controlled_test, norm(test_spin[1, :]))

println(local_control_push(test_spin, 0., local_N_push, 10) - test_spin)

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
