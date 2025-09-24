include("general.jl")


function evolve_state(state, time_steps, epsilon)
    current_state = copy(state)
    L = length(current_state)
    choose_rule = make_rand_state(L, epsilon)     # preallocate rule array
    
    new_state   = [0.0 for _ in 1:L]     # preallocate new state

    for t in 1:time_steps
        choose_probs = rand(Uniform(0,1),L)
        @inbounds @simd for i in eachindex(choose_rule)
            choose_rule[i] = choose_probs[i] < epsilon ? 1 : 0
        end
        @inbounds @simd for i in 1:L
            prev = current_state[ i == 1 ? L : i-1 ] # wrap around
            new_state[i] = choose_rule[i] == 1 ? 1 : prev * current_state[i]
        end

        # swap references instead of copying
        current_state, new_state = new_state, current_state
    end

    return current_state
end