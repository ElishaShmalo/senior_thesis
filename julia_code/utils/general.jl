using Distributed

function get_nearest(N, L)
    """Returns the nearest integer to L that is a multiple of N"""
    return round(Int, L/N) * N
end

function get_theoretical_a_crit(N)
    return ℯ^(-(1-cos((2 * pi) / N))*sqrt(cos((2 * pi) / N)))
end

function shift_arr(arr::Vector, shift_amount::Int)
    n = length(arr)
    if n == 0
        return arr
    end
    shift_amount = mod(shift_amount, n)  # wrap shift to range [0, n-1]
    return vcat(arr[end - shift_amount + 1:end], arr[1:end - shift_amount])
end

function rand_float(a, b)
    return a + (b - a) * rand()  
end

function make_data_file(path, content)
    # Extract the parent directory
    parent_dir = dirname(path)

    # Ensure the parent directories exist
    mkpath(parent_dir)

    open(path, "w") do io
        serialize(io, content)
        println("Saved file $path")
    end
end

function make_path_exist(path)
    # Extract the parent directory
    parent_dir = dirname(path)

    # Ensure the parent directories exist
    mkpath(parent_dir)
end


