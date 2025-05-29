# --- Weighted Spin Difference ---

function weighted_spin_difference(spin_chain::Vector{Vector{Float64}}, s_0::Vector{Vector{Float64}})
    delta_spin_chain = norm(spin_chain .- s_0, 1)
    return delta_spin_chain / length(spin_chain)
end