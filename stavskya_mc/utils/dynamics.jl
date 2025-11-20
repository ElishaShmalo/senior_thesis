include("general.jl")


function evolve_state(state, time_steps, epsilon)
    current_state = copy(state)
    L = length(current_state)
    choose_rule = Vector{Int}(undef, L) # Preallocate choose rule array
    
    new_state   = [0.0 for _ in 1:L]     # preallocate new state

    for t in 1:time_steps
        choose_rule .= Int.(rand(L) .< epsilon)

        new_state[1] = choose_rule[1] + (1 - choose_rule[1]) * (current_state[L] * current_state[1])
        @inbounds @simd for i in 2:L
            prev = current_state[i-1]
            # new_state[i] = choose_rule[i] == 1 ? 1 : prev * current_state[i]
            new_state[i] = choose_rule[i] + (1 - choose_rule[i]) * (prev * current_state[i])
        end

        # swap references instead of copying
        current_state, new_state = new_state, current_state
    end

    return current_state
end

function time_random_delta_evolve_state(state, time_steps, epsilon_prime, delta)
    current_state = copy(state)
    L = length(current_state)
    choose_rule = Vector{Int}(undef, L)
    
    new_state   = [0.0 for _ in 1:L]

    for t in 1:time_steps
        epsilon = (epsilon_prime - delta) + 2*delta*rand()
        choose_rule .= Int.(rand(L) .< epsilon)



        new_state[1] = choose_rule[1] + (1 - choose_rule[1]) * (current_state[L] * current_state[1])
        @inbounds @simd for i in 2:L
            prev = current_state[i-1]
            # new_state[i] = choose_rule[i] == 1 ? 1 : prev * current_state[i]
            new_state[i] = choose_rule[i] + (1 - choose_rule[i]) * (prev * current_state[i])
        end

        # swap references instead of copying
        current_state, new_state = new_state, current_state
    end

    return current_state
end
