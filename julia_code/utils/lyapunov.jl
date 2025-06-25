# function for calculating d_i
function calculate_spin_distence(S_A::Vector{Vector{Float64}}, S_B::Vector{Vector{Float64}})
    diff = S_A .- S_B
    dotted = map(dot, diff, diff)
    return sqrt(sum(dotted))
end

# Bring S_B closer to S_A with fixed vector norm ε across the entire chain
function push_back(S_A::Vector{Vector{Float64}}, S_B::Vector{Vector{Float64}}, epsilon_val::Float64)
    # Flatten both spin chains into one long vector
    flat_A = reduce(vcat, S_A)
    flat_B = reduce(vcat, S_B)

    # Compute difference and normalize it
    diff_vec = flat_B - flat_A
    diff_norm = norm(diff_vec)

    # Rescale difference to have norm epsilon_val
    pushed_vec = flat_A + (epsilon_val / diff_norm) * diff_vec

    # Reshape back into Vector{Vector{Float64}}
    return [pushed_vec[3i-2:3i] for i in 1:length(S_A)]
end

# Calculates Lyapunov val given list of spin distences
function calculate_lambda(spin_dists, tau_val, epsilon_val, n_val)
    return sum(map(log, spin_dists ./ epsilon_val)) / (n_val * tau_val) 
end

function calculate_lambda_per_time(spin_dists, tau_val, epsilon_val, n_val)
    return map(log, spin_dists ./ epsilon_val) / (n_val * tau_val)
end

## I've chosen to keep this *WRONG* implementation of Benettin for nostalgic pourpouses
## Note that this is the wrong way to do Benettin (though it may seem legit).
## If you want to hear why this is incorrect, feel free to contact me (Elisha Shmalo)
# Bring S_B closer to S_A with factor epsilon_val
# function wrong_push_back(S_A::Vector{Vector{Float64}}, S_B::Vector{Vector{Float64}}, epsilon_val)
#     L = length(S_A)
#     spin_diff = S_B .- S_A
#     thing_to_add = (epsilon_val / sqrt(L)) .* (map(normalize, spin_diff))

#     return S_A .+ thing_to_add
# end
