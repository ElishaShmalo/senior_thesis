# --- Defining functions to make Initial States ---

# The states have the form state = [[S_1x, S_1y, S_1z], [S_2x, S_2y, S_2z],... ]

# Random spin state
function make_random_state(n::Int=L)
    return [normalize(rand(3)) for _ in 1:n]
end

# Uniform spin state along z
function make_uniform_state(n::Int=L, z_dir::Int=1)
    return [[0.0, 0.0, z_dir] for _ in 1:n]
end

# Spiral spin state
function make_spiral_state(n::Int=L, spiral_angle::Float64=π/2, phi::Float64=0.0)
    return [[0.0, cos(i * spiral_angle + phi), sin(i * spiral_angle + phi)] for i in 0:(n-1)]
end

function make_random_spin(size = 1)
    return size * normalize(rand(3))
end
