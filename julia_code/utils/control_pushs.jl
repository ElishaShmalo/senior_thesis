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
