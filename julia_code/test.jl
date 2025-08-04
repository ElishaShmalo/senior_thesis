using CSV
using DataFrames

function save_to_csv(data1::Vector{Float64}, data2::Vector{Float64}, filename::String)

    df = DataFrame(Index = 1:length(data1), Data1 = data1, Data2 = data2)
    CSV.write(filename, df)
end

# Example usage:
data1 = [1.0, 2.0, 3.0]
data2 = [4.0, 5.0, 6.0]

save_to_csv(data1, data2, "test.csv")
