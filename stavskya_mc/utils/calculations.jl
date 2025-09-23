include("dynamics.jl")


function calculate_avg_alive(state)
    L = length(state)
    return (1/L) * sum(state)
end