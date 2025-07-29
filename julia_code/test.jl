include("utils/general.jl")

# Define the full file path
file_path = "parent1/parent2/test.txt"

# Extract the parent directory
parent_dir = dirname(file_path)

# Ensure the parent directories exist
mkpath(parent_dir)