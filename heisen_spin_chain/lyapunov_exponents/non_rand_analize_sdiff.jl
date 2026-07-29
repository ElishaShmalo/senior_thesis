# --- S_diff analysis ---

# Imports
using Random, LinearAlgebra, Plots, DifferentialEquations, Serialization, Statistics, DelimitedFiles, SharedArrays, CSV, DataFrames, GLM, LaTeXStrings
using Colors

# Other files   
include("../utils/make_spins.jl")
include("../utils/general.jl")
include("../utils/dynamics.jl")
include("../utils/lyapunov.jl")
include("../analytics/spin_diffrences.jl")

default(
    xlabelfont = 14,   # font size for x-axis label
    ylabelfont = 14,   # font size for y-axis label
    guidefont = 14     # alternative, some backends use 'guidefont'
)


J = 1    # energy factor

# J vector with some randomness
J_vec = J .* [1, 1, 1]

# Time to evolve until push back to S_A
tau = 1 * J

# General Variables
num_unit_cells_vals = [8, 16]

# Create a gradient of blues — light to dark
num_uc = length(num_unit_cells_vals)
blue_palette = blue_palette = cgrad([RGB(0.55, 0.75, 0.85), RGB(0.2, 0.35, 0.9)], num_uc, categorical=true) # from light to dark blue
# num_unit_cells_vals = [8, 16, 32, 64]
# num_unit_cells_vals = [8]

# --- Trying to Replecate Results ---
num_initial_conds = 1000 # We are avraging over x initial conditions
trans_a_vals = [0.7525, 0.755, 0.7575, 0.76, 0.7625, 0.765, 0.7675, 0.77]
post_a_vals = [round(0.8 + i * 0.02, digits=2) for i in 0:5]
a_vals = sort(union([round(0.6 + i*0.01, digits=2) for i in 0:20], [0.7525, 0.755, 0.7575, 0.7625, 0.765, 0.7675], [0.763], post_a_vals)) # general a_vals
# a_vals = sort(union([round(0.7 + i*0.01, digits=2) for i in 0:12], trans_a_vals)) # 0.6, 0.62, 0.64, 0.66, 0.68, 0.7,
# a_vals = [0.68, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.7525, 0.755, 0.7575, 0.76, 0.7605, 0.761, 0.7615, 0.7625, 0.763, 0.765, 0.7675, 0.77, 0.78, 0.79, 0.8]
a_vals = [0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.7525, 0.755, 0.7563, 0.7575, 0.7588,  0.7594, 0.76, 0.7605, 0.761, 0.7615, 0.762, 0.7625, 0.763, 0.765, 0.7675, 0.77, 0.78, 0.79, 0.8]

epsilon = 0.1

N_val = 4

z_val = 1.7
z_val_name = replace("$z_val", "." => "p")

z_fit_val = 1.7
z_fit_name = replace("$z_fit_val", "." => "p")

collected_S_diffs = Dict{Int, Dict{Float64, Vector{Float64}}}() # Int: L_val, Float64: a_val, Vec{Float64}: avrg S_diff(t)
collected_S_diff_SEMs = Dict{Int, Dict{Float64, Vector{Float64}}}() # Int: L_val, Float64: a_val, Vec{Float64}: avrg S_diff(t)

# --- Load in and take avrages from all samples ---
for num_unit_cells in num_unit_cells_vals
    L = num_unit_cells * N_val
    println("L_val: $L")

    # number of pushes we are going to do
    n = Int(round(L^z_fit_val))

    # Define s_naught to be used during control step
    S_NAUGHT = make_spiral_state(L, (2) / N_val)

    # # Initializes results for this N_val
    collected_S_diffs[L] = Dict(a_val => zeros(n) for a_val in a_vals)
    collected_S_diff_SEMs[L] = Dict(a_val => zeros(n) for a_val in a_vals)

    for a_val in a_vals
        println("L_val: $L | a_val: $a_val")
        a_val_name = replace("$a_val", "." => "p")
        # We will avrage over this later
        current_S_diffs = [zeros(Float64, n) for _ in 1:num_initial_conds]

        for init_cond in 1:num_initial_conds

            sample_filepath = "data/non_trand/spin_dists_per_time/N$N_val/a$a_val_name/IC1/L$L/N$(N_val)_a$(a_val_name)_IC1_L$(L)_z$(z_val_name)_sample$(init_cond).csv"
            df = CSV.read(sample_filepath, DataFrame)

            current_S_diffs[init_cond] = df[!, "delta_s"][1:n]
        end

        collected_S_diffs[L][a_val] = mean(current_S_diffs)
        collected_S_diff_SEMs[L][a_val] = std(current_S_diffs) / sqrt(num_initial_conds)
    end
end

# --- Making S_diff Plots ---
a_vals_to_plot = [0.7]
num_unit_cell_to_plot = num_unit_cells_vals[end]
L_val_to_plot = Int(round(num_unit_cell_to_plot * N_val))

# Create plot
plt = plot(
    title=L"$S_{Diff} for N=%$(N_val)$ | L = %$(L_val_to_plot)",
    xlabel=L"t",
    ylabel=L"$S_{Diff}$"
)

# Plot data for each a_val
for (i, a_val) in enumerate(a_vals_to_plot)
    # pick the color Plots.jl will use for series i
    c = Plots.palette(:auto)[i]

    plot!(plt, collected_S_diffs[L_val_to_plot][a_val][1:round(Int, L_val_to_plot^z_fit_val)],
        yerr=collected_S_diff_SEMs[L_val_to_plot][a_val][1:round(Int, L_val_to_plot^z_fit_val)],
        label="a = $(a_val)",
        linestyle=:solid,
        linewidth=1,
        color = c,          # sets line color
        seriescolor = c,    # ensures error bars match
        markerstrokecolor = c # (optional) markers match too
        )
end

s_diff_plot_path = "figs/delta_evolved_spins/N$(N_val)/SeveralAs/IC$num_initial_conds/L$(L_val_to_plot)/S_diff_N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L_val_to_plot)_z$(z_val_name).png"
make_path_exist(s_diff_plot_path)
# savefig(s_diff_plot_path)
println("Saved Plot: $(s_diff_plot_path)")
display(plt)

# --- Making S_diff(t=L^z) as function of a ---
z_val = 1.65
z_val_name = replace("$(z_val)", "." => "p")
plt = plot(
    title=L"$S_{Diff}(t=L^{ %$(z_val) })$ for N=%$(N_val) as Function of a",
    xlabel=L"a",
    ylabel=L"S_{Diff}(t=L^{ %$(z_val) })"
)
plot!(plt, [NaN], [NaN], label = "L =", linecolor = RGBA(0,0,0,0))
L_vals_to_plot = Int.(round.(num_unit_cells_vals * N_val))
# Plot data for each a_val
for (i, L_val) in enumerate(L_vals_to_plot)
    # pick the color Plots.jl will use for series i
    c = blue_palette[i]


    plot!(plt, [a_val for a_val in sort(a_vals)], [collected_S_diffs[L_val][a_val][round(Int, L_val^z_val)] for a_val in sort(a_vals)],
        yerr=[collected_S_diff_SEMs[L_val][a_val][round(Int, L_val^z_val)] for a_val in sort(a_vals)],
        label="$(L_val)",
        linestyle=:solid,
        linewidth=1,
        color = c,          # sets line color
        seriescolor = c,    # ensures error bars match
        markerstrokecolor = c, # (optional) markers match too
        markersize = 4,
        marker = :circle, 
        )
end

s_diff_per_val_plot_path = "figs/delta_evolved_spins/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/S_diff_per_aval_N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(num_unit_cells_vals .* N_val))_z$(z_val_name).png"
make_path_exist(s_diff_per_val_plot_path)
# savefig(s_diff_per_val_plot_path)
println("Saved Plot: $(s_diff_per_val_plot_path)")
display(plt)

# Zoomed S_diff
zoomed_a_vals = [a_val for a_val in sort(a_vals) if 0.75 <= a_val <= 0.77]
plt = plot(
    title=L"$S_{Diff}(t=L^%$(z_val))$ for N=$N_val as Function of a",
    xlabel=L"a",
    ylabel=L"$S_{Diff}(t=L^%$(z_val))$"
)

L_vals_to_plot = Int.(round.(num_unit_cells_vals * N_val))

# Plot data for each a_val
for (i, L_val) in enumerate(L_vals_to_plot)
    # pick the color Plots.jl will use for series i
    c = Plots.palette(:auto)[i]


    plot!(plt, [a_val for a_val in sort(zoomed_a_vals)], [collected_S_diffs[L_val][a_val][round(Int, L_val^z_val)] for a_val in sort(zoomed_a_vals)],
        yerr=[collected_S_diff_SEMs[L_val][a_val][round(Int, L_val^z_val)] for a_val in sort(a_vals)],
        label="L = $(L_val)",
        linestyle=:dash,
        linewidth=1,
        color = c,          # sets line color
        seriescolor = c,    # ensures error bars match
        markerstrokecolor = c # (optional) markers match too
        )
end

zoomed_s_diff_per_val_plot_path = "figs/delta_evolved_spins/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/Zoomed_S_diff_per_aval_N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(num_unit_cells_vals .* N_val))_z$(z_val_name).png"
make_path_exist(zoomed_s_diff_per_val_plot_path)
# savefig(zoomed_s_diff_per_val_plot_path)
println("Saved Plot: $(zoomed_s_diff_per_val_plot_path)")
display(plt)

