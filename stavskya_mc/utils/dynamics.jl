include("general.jl")


function evolve_state(state, time_steps, epsilon)
    current_state = copy(state)
    L = length(current_state)
    choose_rule = make_rand_state(L, epsilon)     # preallocate rule array
    
    new_state   = [0.0 for _ in 1:L]     # preallocate new state

    for t in 1:time_steps
        
        @inbounds @simd for i in eachindex(choose_rule)
            choose_rule[i] = Int(choose_probs[i] < epsilon)
        end
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