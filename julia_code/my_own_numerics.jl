# Hyzenberg Spin Chain Simulation in Julia
# Doing ferromagnet because it's cooler :)

# Imports
using Random
using LinearAlgebra
using Plots
using DifferentialEquations
using StaticArrays

# Set plotting theme
Plots.theme(:dark)

# General Variables
L = 250  # number of spins
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

# --- Equation of Motion and Utils for Runge Kutta ---

# Unroll state into a flat vector
# We need to do this to use runge kutta
function flatten_state(state::Vector{Vector{Float64}})
    return vcat(state...)
end

# Turn the flattened state (vec) back into an actual Vec{Vec{3, Float64}}
function unflatten_state(vec::Vector{Float64})
    return [vec[3i-2:3i] for i in 1:div(length(vec), 3)]
end

function evolve_spin(u::Vector{Float64}, t_span, periodic=false) 
    # Expects and returns a flattened version of the state at the end of t_span
    
    # Define the ODE
    function spin_ode!(du, u, p, t)
        state = unflatten_state(u)
        n = length(state)
        to_cross = Vector{SVector{3, Float64}}(undef, n)

        zero_vec = SVector{3, Float64}(0.0, 0.0, 0.0)

        @inbounds for i in 1:n
            prev = (i == 1)  ? (periodic ? state[end] : zero_vec) : state[i - 1]
            next = (i == n)  ? (periodic ? state[1]  : zero_vec) : state[i + 1]
            to_cross[i] = -J_vec .* (prev + next)
        end

        dstate = [cross(to_cross[i], state[i]) for i in 1:n]
        du .= flatten_state(dstate)
    end

    prob = ODEProblem(spin_ode!, u, t_span)
    sol = solve(prob, RK4(), dt=0.01)

    return sol.u[end]
end

# --- Testing differential_s ---
# Define s_naught as a constant
S_NAUGHT = make_spiral_state(L)
FLATTENED_S_NAUGHT = flatten_state(S_NAUGHT)

# --- Control Push ---
function global_control_push(state, a::Float64)
    numerator = ((1-a) .* S_NAUGHT) .+ (a .* state)
    denominator = map(norm, numerator)
    return numerator ./ denominator
end

function local_control_push(state, a::Float64, N::Int64=local_N_push, start_index::Int64=1)
    # Returns a new state with a control push done to the N spins after (and including) start index in "state"
    local_indexes = [((start_index+i-1) % length(S_NAUGHT)) + 1 for i in 0:N-1]
    
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

println(local_control_push(test_spin, 0., local_N_push, L+10) - test_spin)

# --- Weighted Spin Difference ---

function weighted_spin_difference(spin_chain::Vector{Vector{Float64}}, s_0::Vector{Vector{Float64}})
    delta_spin_chain = norm(spin_chain .- s_0, 1)
    return delta_spin_chain / length(spin_chain)
end

# --- Test Weighted Spin Difference ---

println("No control: ", weighted_spin_difference(test_spin, S_NAUGHT))
println("Push of a=1/2: ", weighted_spin_difference(global_control_push(test_spin, 0.5), S_NAUGHT))
println("Push of a=0: ", weighted_spin_difference(global_control_push(test_spin, 0.0), S_NAUGHT))

# --- First Tests of Dynamics ---

# Evolve to time T with global_control_push
function global_control_evolve(original_state, a_val, T, t_step)
    t = 0.0
    us_of_time = Vector{Vector{Float64}}([])

    current_u = flatten_state(original_state)
    push!(us_of_time, current_u)

    while t < T

        J_vec[1] *= (rand() > 0.5) ? -1 : 1 # Randomly choosing signs for Jx and Jy to remove solitons
        J_vec[2] *= (rand() > 0.5) ? -1 : 1

        current_u = evolve_spin(current_u, (0, t_step))
        current_u = flatten_state(global_control_push(unflatten_state(current_u), a_val))
        push!(us_of_time, current_u)
    
        t += t_step
    end
    return [unflatten_state(u) for u in us_of_time]
end

# Evolve to time T with local_control_push
function local_control_evolve(original_state, a_val, T, t_step)
    t = 0.0
    us_of_time = Vector{Vector{Float64}}([])
    local_control_index = 1

    current_u = flatten_state(original_state)
    push!(us_of_time, current_u)

    while t < T

        J_vec[1] *= (rand() > 0.5) ? -1 : 1 # Randomly choosing signs for Jx and Jy to remove solitons
        J_vec[2] *= (rand() > 0.5) ? -1 : 1

        current_u = evolve_spin(current_u, (0, t_step))
        current_u = local_control_push(unflatten_state(current_u), a_val, local_N_push, local_control_index)
        local_control_index += local_N_push
        local_control_index = ((local_control_index-1) % L) + 1
        push!(us_of_time, current_u)

        t += t_step
    end
    return [unflatten_state(u) for u in us_of_time]
end

# Test with a = 0.6
original_random = make_random_state(L)
returned_states = global_control_evolve(original_random, 0.6, L*J, Tau_F)

S_diffs_of_time = [weighted_spin_difference(state, S_NAUGHT) for state in returned_states]

plot([i for i in 1:length(S_diffs_of_time)], S_diffs_of_time, xlabel="Time", ylabel="S_diff", title="a = 0.6")

savefig("s_diff_plot_a_0p6.png")

# --- Trying to Replecate Results ---
a_vals = [0.6, 0.68, 0.7, 0.716, 0.734, 0.766]

original_random = make_random_state()

S_diffs = Dict{Float64, Vector{Float64}}()

for a_val in a_vals
    returned_states = global_control_evolve(original_random, a_val, L*J, Tau_F)

    S_diffs[a_val] = [weighted_spin_difference(state, S_NAUGHT) for state in returned_states]
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

savefig("s_diff_plot_diffrent_a_vals.jpeg")
