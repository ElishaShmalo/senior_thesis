# function for calculating d_i
function calculate_spin_distence(S_A::Vector{Vector{Float64}}, S_B::Vector{Vector{Float64}})
    dotted = map(dot, S_A, S_B)
    return sqrt(2 * sum(1 .- dotted))
end

# Bring S_B closer to S_A with factor epsilon_val
function push_back(S_A::Vector{Vector{Float64}}, S_B::Vector{Vector{Float64}}, epsilon_val)
    spin_diff = S_B .- S_A
    thing_to_add = (epsilon_val) .* spin_diff ./ map(norm, spin_diff)

    return S_A .+ thing_to_add
end

# Calculates Lyapunov val given list of spin distences
function calculate_lambda(spin_dists, tau_val, epsilon_val, n_val)
    return (1/(n_val * tau_val)) * sum(map(log, spin_dists ./ epsilon_val))
end

function calculate_lambda_per_time(spin_dists, tau_val, epsilon_val, n_val)
    return (1/(n_val * tau_val)) .* map(log, spin_dists ./ epsilon_val)
end
