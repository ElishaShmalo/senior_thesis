
function get_nearest(N, L)
    """Returns the nearest integer to L that is a multiple of N"""
    return round(Int, L/N) * N
end

function get_theoretical_a_crit(N)
    return ℯ^(-(1-cos((2 * pi) / N))*sqrt(cos((2 * pi) / N)))
end