
function flatten_state(state::Vector{Vector{Float64}})
    return vcat(state...)
end

# Turn the flattened state (vec) back into an actual Vec{Vec{3, Float64}}
function unflatten_state(vec::Vector{Float64})
    return [vec[3i-2:3i] for i in 1:div(length(vec), 3)]
end

periodic = false

# Define the ODE
function dstate(u)
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
    return flatten_state(dstate)
end

function make_spiral_state(n::Int=L, spiral_angle::Float64=π/2, phi::Float64=0.0)
    return [[0.0, cos(i * spiral_angle + phi), sin(i * spiral_angle + phi)] for i in 0:(n-1)]
end

s0 = make_spiral_state(250)

println(dstate(flatten_state(s0)))