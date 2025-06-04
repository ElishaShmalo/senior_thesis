# --- Weighted Spin Difference ---

using Statistics

function weighted_spin_difference(spin_chain::Vector{Vector{Float64}}, s_0::Vector{Vector{Float64}})
    delta_spin_chain = norm(spin_chain .- s_0, 1)
    return delta_spin_chain / length(spin_chain)
end

function get_delta_spin(spin_chain1::Vector{Vector{Float64}}, spin_chain2::Vector{Vector{Float64}})
    return map(norm, spin_chain1-spin_chain2)
end

function get_spin_diffrence_from_delta(delta_chain::Vector{Vector{Float64}})
    return map(mean, delta_chain)
end

# This functions handles the case wherin the deltas are in Matrix form
function get_spin_diffrence_from_delta(delta_chain::Matrix{Float64})
    return mean(delta_chain, dims=2)
end
