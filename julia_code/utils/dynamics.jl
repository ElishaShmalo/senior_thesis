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

# --- Control Push ---

function global_control_push(state, a::Float64, s_0::Vector{Vector{Float64}})
    numerator = ((1-a) .* s_0) .+ (a .* state)
    denominator = map(norm, numerator)
    return numerator ./ denominator
end

function local_control_push(state, a::Float64, N::Int64, start_index::Int64, s_0::Vector{Vector{Float64}})
    # Returns a new state with a control push done to the N spins after (and including) start index in "state"
    local_indexes = [((start_index+i-1) % length(s_0)) + 1 for i in 0:N-1]
    
    numerator = ((1-a) .* s_0[local_indexes]) .+ (a .* state[local_indexes])
    denominator = map(norm, numerator)

    local_pushed_state = copy(state)
    local_pushed_state[local_indexes] = numerator ./ denominator

    return local_pushed_state
end

# Define the ODE
function spin_ode!(du, u, p, t)
    state = unflatten_state(u)

    n = length(state)
    
    to_cross = Vector{SVector{3, Float64}}(undef, n)

    # Handle periodic B.C.
    to_cross[1] = -J_vec .* (state[end] + state[2])
    to_cross[end] = -J_vec .* (state[end-1] + state[1])

    @inbounds for i in 2:n-1
        prev = state[i - 1]
        next = state[i + 1]
        to_cross[i] = -J_vec .* (prev + next)
    end

    dstate = [cross(to_cross[i], state[i]) for i in 1:n]
    du .= flatten_state(dstate)
end

function evolve_spin(u_::Vector{Float64}, t_span) 
    # Expects and returns a flattened version of the state at the end of t_span

    prob = ODEProblem(spin_ode!, u_, t_span)
    sol = solve(prob, RK4(), dt=0.001)

    return sol.u[end]
end

# --- Deffrent Evolutions ---
# Evolve to time T, no control push
function no_control_evolve(original_state, T, t_step)
    t = 0.0
    us_of_time = Vector{Vector{Float64}}([flatten_state(original_state)])

    current_u = flatten_state(original_state)
    push!(us_of_time, current_u)

    while t < T - 2

        J_vec[1] *= (rand() > 0.5) ? -1 : 1 # Randomly choosing signs for Jx and Jy to remove solitons
        J_vec[2] *= (rand() > 0.5) ? -1 : 1

        current_u = evolve_spin(current_u, (t, t_step+t))
        push!(us_of_time, current_u)
    
        t += t_step
    end
    return [unflatten_state(u) for u in us_of_time]
end

# Evolve to time T with global_control_push
function random_global_control_evolve(original_state, a_val, T, t_step, s_0)
    t = 0
    us_of_time = Vector{Vector{Float64}}([flatten_state(original_state)])

    current_u = flatten_state(original_state)
    push!(us_of_time, current_u)

    while t < T - 2
        J_vec[1] *= (rand() > 0.5) ? -1 : 1 # Randomly choosing signs for Jx and Jy to remove solitons
        J_vec[2] *= (rand() > 0.5) ? -1 : 1

        current_u = evolve_spin(current_u, (t, t_step+t))
        current_u = flatten_state(global_control_push(unflatten_state(current_u), a_val, s_0))
        push!(us_of_time, current_u)
    
        t += t_step
    end
    return [unflatten_state(u) for u in us_of_time]
end

# Evolve to time T with global_control_push
function global_control_evolve(original_state, a_val, T, t_step, s_0)
    t = 0
    us_of_time = Vector{Vector{Float64}}([flatten_state(original_state)])

    current_u = flatten_state(original_state)
    push!(us_of_time, current_u)

    while t < T - 2

        current_u = evolve_spin(current_u, (t, t_step+t))
        current_u = flatten_state(global_control_push(unflatten_state(current_u), a_val, s_0))
        push!(us_of_time, current_u)
    
        t += t_step
    end
    return [unflatten_state(u) for u in us_of_time]
end

# Evolve to time T with local_control_push
function local_control_evolve(original_state, a_val, T, t_step, s_0)
    t = 0.0
    us_of_time = Vector{Vector{Float64}}([flatten_state(original_state)])
    local_control_index = 1

    current_u = flatten_state(original_state)
    push!(us_of_time, current_u)

    while t < T -2

        J_vec[1] *= (rand() > 0.5) ? -1 : 1 # Randomly choosing signs for Jx and Jy to remove solitons
        J_vec[2] *= (rand() > 0.5) ? -1 : 1

        current_u = evolve_spin(current_u, (t, t_step+t))
        current_u = flatten_state(local_control_push(unflatten_state(current_u), a_val, local_N_push, local_control_index, s_0))
        local_control_index += local_N_push
        local_control_index = ((local_control_index-1) % L) + 1
        push!(us_of_time, current_u)

        t += t_step
    end
    return [unflatten_state(u) for u in us_of_time]
end
