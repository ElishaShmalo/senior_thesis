# --- Equation of Motion and Utils for Runge Kutta ---
using StaticArrays

# Unroll state into a flat vector
# We need to do this to use runge kutta
function flatten_state(state::Vector{Vector{Float64}})
    return vcat(state...)
end

# Turn the flattened state (vec) back into an actual Vec{Vec{3, Float64}}
function unflatten_state(vec::Vector{Float64})
    return [vec[3i-2:3i] for i in 1:div(length(vec), 3)]
end

function get_diffrential_test(L_J_vec, state)
    n = length(state)
    to_cross = Vector{SVector{3, Float64}}(undef, n)

    # Handle periodic B.C.
    to_cross[1] = -L_J_vec .* (state[end] + state[2])
    to_cross[end] = -L_J_vec .* (state[end-1] + state[1])

    @inbounds for i in 2:n-1
        prev = state[i - 1]
        next = state[i + 1]
        to_cross[i] = -L_J_vec .* (prev + next)
    end

    dstate = [cross(to_cross[i], state[i]) for i in 1:n]
    return dstate
end

# --- Control Push ---

function global_control_push(state::Vector{Vector{Float64}}, a::Float64, s_0::Vector{Vector{Float64}})
    numerator = ((1-a) .* s_0) .+ (a .* state)
    return map(normalize, numerator)
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

    L_J_vec = p

    n = length(state)
    
    to_cross = Vector{SVector{3, Float64}}(undef, n)

    # Handle periodic B.C.
    to_cross[1] = -L_J_vec .* (state[end] + state[2])
    to_cross[end] = -L_J_vec .* (state[end-1] + state[1])

    @inbounds for i in 2:n-1
        prev = state[i - 1]
        next = state[i + 1]
        to_cross[i] = -L_J_vec .* (prev + next)
    end

    dstate = [cross(to_cross[i], state[i]) for i in 1:n]
    du .= flatten_state(dstate)
end

function evolve_spin(L_J_vec, u_::Vector{Float64}, t_span) 
    # Expects and returns a flattened version of the state at the end of t_span

    prob = ODEProblem(spin_ode!, u_, t_span, L_J_vec)
    sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-10)

    return sol.u[end]
end

# --- Deffrent Evolutions ---
# Evolve to time T, no control push
function no_control_evolve(L_J_vec, original_state, T, t_step)
    t = t_step
    
    current_u = flatten_state(original_state)
    us_of_time = Vector{Vector{Float64}}([zeros(length(current_u)) for _ in 0:div(T, t_step)])

    while t < T + t_step

        L_J_vec[1] *= (rand() > 0.5) ? -1 : 1 # Randomly choosing signs for Jx and Jy to remove solitons
        L_J_vec[2] *= (rand() > 0.5) ? -1 : 1

        current_u = evolve_spin(L_J_vec, current_u, (t, t_step+t))
        push!(us_of_time, current_u)
    
        t += t_step
    end
    return [unflatten_state(u) for u in us_of_time]
end

# Evolve to time T with global_control_push
function global_control_evolve(L_J_vec, original_state, a_val, T, t_step, s_0)
        current_u = flatten_state(original_state)
    us_of_time = Vector{Vector{Float64}}([zeros(length(current_u)) for _ in 0:div(T, t_step)])

    us_of_time[1] = current_u
    t = t_step
    while t < T + t_step

        current_u = evolve_spin(L_J_vec, current_u, (t, t+t_step))
        current_u = flatten_state(global_control_push(unflatten_state(current_u), a_val, s_0))
        
        t += t_step
        us_of_time[Int(div(t, t_step))] = current_u 
    end
    return [unflatten_state(u) for u in us_of_time]
end

# Evolve to time T with global_control_push and random J_x J_y
"""
    random_global_control_evolve(L_J_vec, original_state, a_val, T, t_step, s_0)

Returns the whole trajectory as a vector of states: element k is the state at time
(k-1)*t_step, for k = 1, ..., div(T, t_step)+1. Element 1 is the initial state, so element k
is NOT time k (review 4.1.2).

Every state is stored, so memory grows like L*T. For S_diff time series use
`random_global_control_sdiff` below, which keeps O(L) memory and records true times.

2026-09 change (review 4.1.4): `L_J_vec` is copied. Before, the random signs were written
into the caller's vector (the global J_vec), so each call started from the signs the
previous one left behind. That was statistically harmless but fragile. For a fixed seed
and a fresh J_vec the trajectory is unchanged.
"""
function random_global_control_evolve(L_J_vec, original_state, a_val, T, t_step, s_0)
    J_signs = copy(L_J_vec)
    current_u = flatten_state(original_state)
    us_of_time = Vector{Vector{Float64}}([zeros(length(current_u)) for _ in 0:div(T, t_step)])
    us_of_time[1] = current_u
    t = t_step
    while t < T + t_step
        J_signs[1] *= (rand() > 0.5) ? -1 : 1 # Randomly choosing signs for Jx and Jy to remove solitons
        J_signs[2] *= (rand() > 0.5) ? -1 : 1

        current_u = evolve_spin(J_signs, current_u, (t, t+t_step))
        current_u = flatten_state(global_control_push(unflatten_state(current_u), a_val, s_0))
        
        t += t_step
        us_of_time[Int(div(t, t_step))] = current_u 
    end
    return [unflatten_state(u) for u in us_of_time]
end

"""
    random_global_control_sdiff(L_J_vec, original_state, a_val, T, t_step, s_0; record_every=1)

Same dynamics as `random_global_control_evolve`: the same random numbers are drawn in the
same order, so a fixed seed gives an identical trajectory. The difference is that only the
current state is kept, and S_diff (`weighted_spin_difference`, analytics/spin_diffrences.jl)
is measured on the fly.

Returns `(times, s_diff)` at the true times t = 0, record_every*t_step, 2*record_every*t_step,
..., up to div(T, t_step)*t_step. t = 0 is the initial state. Memory is O(L) instead of
O(L*T). `L_J_vec` is copied, so the caller's vector is never modified.
(2026-09, review 4.1.2-4.1.4)
"""
function random_global_control_sdiff(L_J_vec, original_state, a_val, T, t_step, s_0;
                                     record_every::Integer=1)
    record_every >= 1 || throw(ArgumentError("record_every must be >= 1, got $record_every"))
    J_signs = copy(L_J_vec)
    current_u = flatten_state(original_state)
    n_steps = div(T, t_step)            # same number of steps as random_global_control_evolve
    times = Float64[0.0]
    s_diff = Float64[weighted_spin_difference(unflatten_state(current_u), s_0)]
    t = t_step
    for k in 1:n_steps
        J_signs[1] *= (rand() > 0.5) ? -1 : 1   # random signs of J_x, J_y (removes solitons)
        J_signs[2] *= (rand() > 0.5) ? -1 : 1
        current_u = evolve_spin(J_signs, current_u, (t, t + t_step))
        current_u = flatten_state(global_control_push(unflatten_state(current_u), a_val, s_0))
        t += t_step
        if k % record_every == 0
            push!(times, k * t_step)
            push!(s_diff, weighted_spin_difference(unflatten_state(current_u), s_0))
        end
    end
    return times, s_diff
end

# Evolve to time T with global_control_push and random J_x 
function semirand_global_control_evolve(L_J_vec, original_state, a_val, T, t_step, s_0)
    current_u = flatten_state(original_state)
    us_of_time = Vector{Vector{Float64}}([zeros(length(current_u)) for _ in 0:div(T, t_step)])

    us_of_time[1] = current_u
    t = t_step
    while t < T + t_step
        L_J_vec[1] *= (rand() > 0.5) ? -1 : 1 # Randomly choosing signs for Jx to remove solitons

        current_u = evolve_spin(L_J_vec, current_u, (t, t+t_step))
        current_u = flatten_state(global_control_push(unflatten_state(current_u), a_val, s_0))
        
        t += t_step
        us_of_time[Int(div(t, t_step))] = current_u 
    end
    return [unflatten_state(u) for u in us_of_time]
end



# Used for calculating lyapunov exponents
function random_evolve_spins_to_time(L_J_vec, Sa, Sb, a_val, T, t_step, s_0)
    current_us = [flatten_state(Sa), flatten_state(Sb)]
    us_of_time = [Vector{Vector{Float64}}([zeros(length(current_us)) for _ in 0:div(T, t_step)]), Vector{Vector{Float64}}([zeros(length(current_us)) for _ in 0:div(T, t_step)])]

    us_of_time[1][1] = current_us[1]
    us_of_time[2][1] = current_us[2]

    t = t_step
    while t < T + t_step
        # 2026-09: keep |J_x|, |J_y| and only randomize the sign (before, the entries were set to
        # +-1, which silently dropped the magnitude if J != 1). The caller's vector is still
        # modified here; benettin_lambda_sdiff (utils/lyapunov.jl) passes it a private copy.
        L_J_vec[1] = ((rand() > 0.5) ? -1 : 1) * abs(L_J_vec[1]) # Randomly choosing signs for Jx and Jy to remove solitons
        L_J_vec[2] = ((rand() > 0.5) ? -1 : 1) * abs(L_J_vec[2])

        current_us = [evolve_spin(L_J_vec, current_us[1], (t, t_step+t)), evolve_spin(L_J_vec, current_us[2], (t, t_step+t))]
        current_us[1] = flatten_state(global_control_push(unflatten_state(current_us[1]), a_val, s_0))
        current_us[2] = flatten_state(global_control_push(unflatten_state(current_us[2]), a_val, s_0))
        
        t += t_step
        us_of_time[1][Int(div(t, t_step))] = current_us[1]
        us_of_time[2][Int(div(t, t_step))] = current_us[2]
    end
    return [[unflatten_state(u) for u in us_of_time[1]], [unflatten_state(u) for u in us_of_time[2]]]
end

# Used for calculating lyapunov exponents
function evolve_spins_to_time(L_J_vec, Sa, Sb, a_val, T, t_step, s_0)
    current_us = [flatten_state(Sa), flatten_state(Sb)]
    us_of_time = [Vector{Vector{Float64}}([zeros(length(current_us)) for _ in 0:div(T, t_step)]), Vector{Vector{Float64}}([zeros(length(current_us)) for _ in 0:div(T, t_step)])]

    us_of_time[1][1] = current_us[1]
    us_of_time[2][1] = current_us[2]

    t = t_step
    while t < T + t_step
        current_us = [evolve_spin(L_J_vec, current_us[1], (t, t_step+t)), evolve_spin(L_J_vec, current_us[2], (t, t_step+t))]
        current_us[1] = flatten_state(global_control_push(unflatten_state(current_us[1]), a_val, s_0))
        current_us[2] = flatten_state(global_control_push(unflatten_state(current_us[2]), a_val, s_0))
        
        t += t_step
        us_of_time[1][Int(div(t, t_step))] = current_us[1]
        us_of_time[2][Int(div(t, t_step))] = current_us[2]
    end
    return [[unflatten_state(u) for u in us_of_time[1]], [unflatten_state(u) for u in us_of_time[2]]]
end

# Used for calculating lyapunov exponents
function semirand_evolve_spins_to_time(L_J_vec, Sa, Sb, a_val, T, t_step, s_0)
    current_us = [flatten_state(Sa), flatten_state(Sb)]
    us_of_time = [Vector{Vector{Float64}}([zeros(length(current_us)) for _ in 0:div(T, t_step)]), Vector{Vector{Float64}}([zeros(length(current_us)) for _ in 0:div(T, t_step)])]

    us_of_time[1][1] = current_us[1]
    us_of_time[2][1] = current_us[2]

    t = t_step
    while t < T + t_step
        L_J_vec[1] = (rand() > 0.5) ? -1 : 1 # Randomly choosing signs for Jx and Jy to remove solitons

        current_us = [evolve_spin(L_J_vec, current_us[1], (t, t_step+t)), evolve_spin(L_J_vec, current_us[2], (t, t_step+t))]
        current_us[1] = flatten_state(global_control_push(unflatten_state(current_us[1]), a_val, s_0))
        current_us[2] = flatten_state(global_control_push(unflatten_state(current_us[2]), a_val, s_0))
        
        t += t_step
        us_of_time[1][Int(div(t, t_step))] = current_us[1]
        us_of_time[2][Int(div(t, t_step))] = current_us[2]
    end
    return [[unflatten_state(u) for u in us_of_time[1]], [unflatten_state(u) for u in us_of_time[2]]]
end








# Evolve to time T with local_control_push
function local_control_evolve(original_state, a_val, T, t_step, s_0)
    t = t_step
    local_control_index = 1

    current_u = flatten_state(original_state)
    us_of_time = Vector{Vector{Float64}}([zeros(length(current_u)) for _ in 0:div(T, t_step)])

    while t < T + t_step

        L_J_vec[1] *= (rand() > 0.5) ? -1 : 1 # Randomly choosing signs for Jx and Jy to remove solitons
        L_J_vec[2] *= (rand() > 0.5) ? -1 : 1

        current_u = evolve_spin(L_J_vec, current_u, (t, t_step+t))
        current_u = flatten_state(local_control_push(unflatten_state(current_u), a_val, local_N_push, local_control_index, s_0))
        local_control_index += local_N_push
        local_control_index = ((local_control_index-1) % L) + 1
        push!(us_of_time, current_u)

        t += t_step
    end
    return [unflatten_state(u) for u in us_of_time]
end
