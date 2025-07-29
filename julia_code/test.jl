include("utils/general.jl")

# Define the full file path
file_path = "parent1/parent2/test.txt"

# Extract the parent directory
parent_dir = dirname(file_path)

println(dirname("data/spin_chain_lambdas/data/spin_chain_lambdas/N2/SeveralAs/IC10/L256/N2_ar0p5_0p8_IC10_L256.dat.dat"))

# Ensure the parent directories exist
mkpath(parent_dir)