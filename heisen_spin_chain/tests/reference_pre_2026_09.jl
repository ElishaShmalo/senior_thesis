# Frozen copies of two functions from heisen_spin_chain/utils/dynamics.jl as they were BEFORE
# the 2026-09 changes (renamed with an _old suffix), plus the old inline Benettin loop of
# get_good_data_severalL*.jl. Used only by test_spin_refactor.jl. Do not edit.

function random_global_control_evolve_old(L_J_vec, original_state, a_val, T, t_step, s_0)
    current_u = flatten_state(original_state)
    us_of_time = Vector{Vector{Float64}}([zeros(length(current_u)) for _ in 0:div(T, t_step)])
    us_of_time[1] = current_u
    t = t_step
    while t < T + t_step
        L_J_vec[1] *= (rand() > 0.5) ? -1 : 1 # Randomly choosing signs for Jx and Jy to remove solitons
        L_J_vec[2] *= (rand() > 0.5) ? -1 : 1

        current_u = evolve_spin(L_J_vec, current_u, (t, t+t_step))
        current_u = flatten_state(global_control_push(unflatten_state(current_u), a_val, s_0))
        
        t += t_step
        us_of_time[Int(div(t, t_step))] = current_u 
    end
    return [unflatten_state(u) for u in us_of_time]
end

function random_evolve_spins_to_time_old(L_J_vec, Sa, Sb, a_val, T, t_step, s_0)
    current_us = [flatten_state(Sa), flatten_state(Sb)]
    us_of_time = [Vector{Vector{Float64}}([zeros(length(current_us)) for _ in 0:div(T, t_step)]), Vector{Vector{Float64}}([zeros(length(current_us)) for _ in 0:div(T, t_step)])]

    us_of_time[1][1] = current_us[1]
    us_of_time[2][1] = current_us[2]

    t = t_step
    while t < T + t_step
        L_J_vec[1] = (rand() > 0.5) ? -1 : 1 # Randomly choosing signs for Jx and Jy to remove solitons
        L_J_vec[2] = (rand() > 0.5) ? -1 : 1

        current_us = [evolve_spin(L_J_vec, current_us[1], (t, t_step+t)), evolve_spin(L_J_vec, current_us[2], (t, t_step+t))]
        current_us[1] = flatten_state(global_control_push(unflatten_state(current_us[1]), a_val, s_0))
        current_us[2] = flatten_state(global_control_push(unflatten_state(current_us[2]), a_val, s_0))
        
        t += t_step
        us_of_time[1][Int(div(t, t_step))] = current_us[1]
        us_of_time[2][Int(div(t, t_step))] = current_us[2]
    end
    return [[unflatten_state(u) for u in us_of_time[1]], [unflatten_state(u) for u in us_of_time[2]]]
end

# The loop that get_good_data_severalL*.jl ran inline before 2026-09 (lines 74-102), unchanged
# apart from being wrapped in a function and calling the _old evolve function.
function old_inline_benettin(J_vec, spin_chain_A, a_val, n, tau, J, S_NAUGHT, epsilon)
    spin_chain_B = copy(spin_chain_A)
    new_mid_spin_val = spin_chain_B[div(length(spin_chain_B), 2)] + make_random_spin(epsilon)
    spin_chain_B[div(length(spin_chain_B), 2)] = normalize(new_mid_spin_val)
    current_spin_dists = zeros(n)
    current_sdiffs = zeros(n)
    for current_n in 1:n
        evolved_results = random_evolve_spins_to_time_old(J_vec, spin_chain_A, spin_chain_B, a_val, tau, J, S_NAUGHT)
        spin_chain_A = evolved_results[1][end]
        spin_chain_B = evolved_results[2][end]
        current_spin_dists[current_n] = calculate_spin_distence(spin_chain_A, spin_chain_B)
        spin_chain_B = push_back(spin_chain_A, spin_chain_B, epsilon)
        current_sdiffs[current_n] = weighted_spin_difference(spin_chain_A, S_NAUGHT)
    end
    return collect(1:n), calculate_lambda_per_time(current_spin_dists, epsilon), current_sdiffs
end
