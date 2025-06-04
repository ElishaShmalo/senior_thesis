# This is a file with a few helpful functions for saving and loading data (since we have so much)

function convert_evolved_states_to_float32(evolved_states::Dict{Float64, Array{Float64,4}})
    new_dict = Dict{Float64, Array{Float32,4}}()
    for (a_val, arr) in evolved_states
        new_dict[a_val] = Float32.(arr)
    end
    return new_dict
end

