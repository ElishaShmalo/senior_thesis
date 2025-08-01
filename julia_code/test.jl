include("utils/general.jl")
include("utils/dynamics.jl")
include("utils/make_spins.jl")

println(evolve_spins_to_time([1, 1, 1], make_random_state(4), make_random_state(4), 0.5, 10, 1, make_spiral_state(4, 1/2))[1])