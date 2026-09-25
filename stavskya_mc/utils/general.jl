using Distributions

function make_rand_state(L, prob)
    Int.(rand(L) .< prob)
end

function make_path_exist(path)
    # Extract the parent directory
    parent_dir = dirname(path)

    # Ensure the parent directories exist
    mkpath(parent_dir)
end

function save_simple_dict_to_csv(dict::Dict{Float64, Float64}, filepath::String)
    df = DataFrame("a" => collect(keys(dict)), "observations" => collect(values(dict)))
    make_path_exist(filepath)
    CSV.write(filepath, df)
    println("Wrote csv $(filepath)")
end


"""
    make_log_times(t_max; points_per_decade=20)

Sorted, unique integer output times: 0 (the initial state), then about
`points_per_decade` roughly log-spaced times per decade from 1 up to and including
`t_max`. Added 2026-09 for the log-time generators in block_disorder/.
"""
function make_log_times(t_max::Integer; points_per_decade::Integer=20)
    t_max >= 1 || throw(ArgumentError("t_max must be >= 1, got $t_max"))
    points_per_decade >= 1 || throw(ArgumentError("points_per_decade must be >= 1"))
    n = ceil(Int, points_per_decade * log10(t_max)) + 1
    ts = round.(Int, exp10.(range(0, log10(t_max); length=max(n, 2))))
    return unique(vcat(0, ts, t_max))
end
