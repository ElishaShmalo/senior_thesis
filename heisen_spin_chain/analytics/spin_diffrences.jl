# --- Weighted Spin Difference ---

using Statistics

function weighted_spin_difference(spin_chain::Vector{Vector{Float64}}, s_0::Vector{Vector{Float64}})
    return mean(map(norm, spin_chain .- s_0))
end

function weighted_spin_difference_vs_time(spin_chain_vs_time::Vector{Vector{Vector{Float64}}}, s_0::Vector{Vector{Float64}})
    return map(sc -> weighted_spin_difference(sc, s_0), spin_chain_vs_time)
end

function get_delta_spin(spin_chain1::Vector{Vector{Float64}}, spin_chain2::Vector{Vector{Float64}})
    return map(norm, spin_chain1-spin_chain2)
end

function get_OTOC(spin_chain1::Vector{Vector{Float64}}, spin_chain2::Vector{Vector{Float64}})
    return 1 .- map(dot, spin_chain1, spin_chain2)
end

function get_spin_diffrence_from_delta(delta_chain::Vector{Vector{Float64}})
    return map(mean, delta_chain)
end

# This functions handles the case wherin the deltas are in Matrix form
function get_spin_diffrence_from_delta(delta_chain::Matrix{Float64})
    return mean(delta_chain, dims=2)
end
