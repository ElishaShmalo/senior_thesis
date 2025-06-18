# function for calculating d_i
function calculate_spin_distence(S_A::Vector{Vector{Float64}}, S_B::Vector{Vector{Float64}})
    diff = S_A .- S_B
    dotted = map(dot, diff, diff)
    return sqrt(sum(dotted))
end

# Bring S_B closer to S_A with factor epsilon_val
function push_back(S_A::Vector{Vector{Float64}}, S_B::Vector{Vector{Float64}}, epsilon_val)
    L = length(S_A)
    spin_diff = S_B .- S_A
    thing_to_add = (epsilon_val) .* (map(normalize, spin_diff))

    return S_A .+ thing_to_add
end

# Calculates Lyapunov val given list of spin distences
function calculate_lambda(spin_dists, tau_val, epsilon_val, n_val)
    return sum(map(log, spin_dists ./ epsilon_val)) / (n_val * tau_val) 
end

function calculate_lambda_per_time(spin_dists, tau_val, epsilon_val, n_val)
    return map(log, spin_dists ./ epsilon_val) / (n_val * tau_val)
end
