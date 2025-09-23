using Distributions

function make_rand_state(L, prob)
    [val < prob ? 1 : 0 for val in rand(Uniform(0,1),L)]
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
