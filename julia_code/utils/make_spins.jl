# --- Defining functions to make Initial States ---
# The states have the form state = [[S_1x, S_1y, S_1z], [S_2x, S_2y, S_2z],... ]

using SymPy

# Random spin state
function make_random_state(n::Int=L)
    return [normalize(rand(3)) for _ in 1:n]
end

# Uniform spin state along z
function make_uniform_state(n::Int=L, z_dir::Int=1)
    return [[0.0, 0.0, z_dir] for _ in 1:n]
end

function a_cospi(x)
    return N(simplify(SymPy.cos(Sym(x) * SymPy.pi)))
end
function a_sinpi(x)
    return N(simplify(SymPy.sin(Sym(x) * SymPy.pi)))
end

# Spiral spin state
function make_spiral_state(n::Int=L, spiral_angle_coff::Float64=1/2)
    return [[0.0, a_cospi(i * spiral_angle_coff), a_sinpi(i * spiral_angle_coff)] for i in 0:(n-1)]
end

# function make_spiral_state(n::Int=L, spiral_angle_coff::Float64=1/2)
#     return [[0.0, cospi(i * spiral_angle_coff), sinpi(i * spiral_angle_coff)] for i in 0:(n-1)]
# end

function make_random_spin(size = 1)
    return size * normalize(rand(3))
end

