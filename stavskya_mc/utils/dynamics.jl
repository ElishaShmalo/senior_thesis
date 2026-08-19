include("general.jl")
using Random

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
    
    new_state = [0.0 for _ in 1:L]

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

function time_random_n_evolve_state(state, time_steps, epsilon_prime, a)
    current_state = copy(state)
    L = length(current_state)
    choose_rule = Vector{Int}(undef, L)

    new_state   = [0.0 for _ in 1:L]

    n = (a/epsilon_prime) - 1
    
    for t in 1:time_steps
        epsilon = a*rand()^n
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

@inline gaussian(mean::T, std::T) where {T<:AbstractFloat} =
    mean + std * randn(T)

function time_random_gauss_evolve_state(state, time_steps, epsilon_prime, sig)
    current_state = copy(state)
    L = length(current_state)
    choose_rule = Vector{Int}(undef, L)

    new_state = [0.0 for _ in 1:L]
    
    for t in 1:time_steps
        epsilon = gaussian(epsilon_prime, sig)

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

function time_random_binary_evolve_state(state, time_steps, epsilon_prime)
    current_state = copy(state)
    L = length(current_state)
    choose_rule = Vector{Int}(undef, L)

    new_state = [0.0 for _ in 1:L]
    for t in 1:time_steps
        epsilon = 2 * round(rand()) * epsilon_prime

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

function time_random_p_evolve_state(state, time_steps, upper_ep, lower_ep, p_val)
    current_state = copy(state)
    L = length(current_state)
    choose_rule = Vector{Int}(undef, L)

    new_state = [0.0 for _ in 1:L]
    for t in 1:time_steps
        choice = Int(rand() > p_val)
        epsilon = [upper_ep, lower_ep][choice+1]

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
