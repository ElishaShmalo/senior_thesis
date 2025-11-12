
# Imports
using Random, LinearAlgebra, Plots, DifferentialEquations, Serialization, Statistics, DelimitedFiles, SharedArrays, CSV, DataFrames, GLM, LaTeXStrings
using Colors
using Measures

# Other files   
include("../utils/make_spins.jl")
include("../utils/general.jl")
include("../utils/dynamics.jl")
include("../utils/lyapunov.jl")
include("../analytics/spin_diffrences.jl")

default(
    guidefont = 24,     # alternative, some backends use 'guidefont'
    tickfont = 14,      # font size for axis tick marks
    legendfont = 10, 
    margin = 3mm
)

J = 1

# Time to evolve until push back to S_A
tau = 1 * J

# General Variables
num_unit_cells_vals = [64]

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
a_vals = [0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.7525, 0.755, 0.7575, 0.76, 0.7605, 0.761, 0.7615, 0.7625, 0.763, 0.765, 0.7675, 0.77, 0.78, 0.79, 0.8]
# a_vals = [0.62, 0.64, 0.66, 0.69, 0.68, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.7525, 0.755, 0.7563, 0.7575, 0.7588,  0.7594, 0.76]

epsilon = 0.1

N_val = 4

z_val = 1.7
z_val_name = replace("$z_val", "." => "p")

z_fit_val = 1.6
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

        if a_val < 0.7 && (a_val * 100) % 2 == 1
            z_val = 1.6
            z_val_name = replace("$z_val", "." => "p")
        else
            z_val = 1.7
            z_val_name = replace("$z_val", "." => "p")
        end

        for init_cond in 1:num_initial_conds

            sample_filepath = "data/spin_dists_per_time/N$N_val/a$a_val_name/IC1/L$L/N$(N_val)_a$(a_val_name)_IC1_L$(L)_z$(z_val_name)_sample$(init_cond).csv"
            df = CSV.read(sample_filepath, DataFrame)

            current_S_diffs[init_cond] = df[!, "delta_s"][1:n]
        end

        collected_S_diffs[L][a_val] = mean(current_S_diffs)
        collected_S_diff_SEMs[L][a_val] = std(current_S_diffs) / sqrt(num_initial_conds)
    end
end

# First we calculate the error on the log(s_diff)
collected_log_S_diff_errs = Dict{Int, Dict{Float64, Vector{Float64}}}() # Int: L_val, Float64: a_val, Vec{Float64}: avrg S_diff(t)
for L_val in num_unit_cells_vals * N_val
    println("L_val $(L_val)")
    collected_log_S_diff_errs[L_val] = Dict{Float64, Vector{Float64}}()
    for a_val in a_vals
        collected_log_S_diff_errs[L_val][a_val] = (1 ./ (collected_S_diffs[L_val][a_val])) .* collected_S_diff_SEMs[L_val][a_val]
    end
end


times_to_fit = Dict{Float64, Dict{Int, Dict{Float64, Vector{Int}}}}() # z_val, L_val, a_val, (min, max)

# Now we do for z = 1.6
times_to_fit[1.6] = Dict{Int, Dict{Float64, Vector{Int}}}()
times_to_fit[1.6][512] = Dict{Float64, Vector{Int}}(0.67 => [5, 80],
                                               0.68 => [5, 90],
                                               0.69 => [5, 100],
                                               0.70 => [5, 140],
                                               0.71 => [5, 160],
                                               0.72 => [20, 256],
                                               0.73 => [40, 410],
                                               0.74 => [60, 900],
                                               0.75 => [80, 3150],
                                               0.7525 => [175, 4350],
                                               0.755 => [80, 1.250 * 10^4],
                                               0.7575 => [400, Int(round(512^1.6))],
                                               0.76 => [400, Int(round(512^1.6))],
                                               0.7625 => [400, Int(round(512^1.6))],
                                               0.763 => [400, Int(round(512^1.6))])
times_to_fit[1.6][256] = Dict{Float64, Vector{Int}}(0.67 => [5, 70],
                                               0.68 => [5, 80], 
                                               0.69 => [5, 100], 
                                               0.7 => [5, 120], 
                                               0.71 => [10, 200], 
                                               0.72 => [10, 325], 
                                               0.73 => [100, 650], 
                                               0.74 => [200, 1400], 
                                               0.75 => [200, 3500], 
                                               0.7525 => [400, 5200], 
                                               0.755 => [300, 6500], 
                                               0.7575 => [200, Int(round(256^1.6))], 
                                               0.76 => [200, Int(round(256^1.6))], 
                                               0.7625 => [200, Int(round(256^1.6))], 
                                               0.763 => [200, Int(round(256^1.6))])
times_to_fit[1.6][128] = Dict{Float64, Vector{Int}}(0.67 => [5, 60],
                                               0.68 => [5, 84], 
                                               0.69 => [5, 96], 
                                               0.7 => [5, 120], 
                                               0.71 => [5, 180], 
                                               0.72 => [5, 250], 
                                               0.73 => [5, 500], 
                                               0.74 => [100, 1350], 
                                               0.75 => [150, Int(round(128^1.6))-50], 
                                               0.7525 => [150, Int(round(128^1.6))], 
                                               0.755 => [150, Int(round(128^1.6))], 
                                               0.7575 => [150, Int(round(128^1.6))], 
                                               0.76 => [200, Int(round(128^1.6))], 
                                               0.7625 => [200, Int(round(128^1.6))], 
                                               0.763 => [200, Int(round(128^1.6))])
times_to_fit[1.6][64] = Dict{Float64, Vector{Int}}(0.67 => [1, 65],
                                               0.68 => [1, 90], 
                                               0.69 => [1, 100], 
                                               0.7 => [1, 140], 
                                               0.71 => [1, 220], 
                                               0.72 => [1, 280], 
                                               0.73 => [1, 340], 
                                               0.74 => [1, 600], 
                                               0.75 => [1, Int(round(64^1.6))], 
                                               0.7525 => [1, Int(round(64^1.6))], 
                                               0.755 => [1, Int(round(64^1.6))], 
                                               0.7575 => [1, Int(round(64^1.6))], 
                                               0.76 => [1, Int(round(64^1.6))], 
                                               0.7625 => [1, Int(round(64^1.6))], 
                                               0.763=> [1, Int(round(64^1.6))],)

times_to_fit[1.6][32] = Dict{Float64, Vector{Int}}(0.67 => [1, 60],
                                               0.68 => [1, 70], 
                                               0.69 => [1, 90], 
                                               0.7 => [1, 120], 
                                               0.71 => [1, 168], 
                                               0.72 => [1, 240], 
                                               0.73 => [1, Int(round(32^1.6))], 
                                               0.74 => [1, Int(round(32^1.6))], 
                                               0.75 => [1, Int(round(32^1.6))], 
                                               0.7525 => [1, Int(round(32^1.6))], 
                                               0.755 => [1, Int(round(32^1.6))], 
                                               0.7575 => [1, Int(round(32^1.6))], 
                                               0.76 => [1, Int(round(32^1.6))], 
                                               0.7625 => [1, Int(round(32^1.6))], 
                                               0.763 => [1, Int(round(32^1.6))])



z_fit_val = 1.6
z_fit_name = replace("$(z_fit_name)", "." => "p")

decay_a_vals = [val for val in a_vals if 0.68 <= val <= 0.7605]
# decay_a_vals = [0.69]

# Calculating the fits
all_log_s_diff_slopes = Dict{Int, Dict{Float64, Float64}}()
all_log_s_diff_offsets = Dict{Int, Dict{Float64, Float64}}()
all_log_s_diff_slope_errs = Dict{Int, Dict{Float64, Float64}}()

for L_val in N_val .* num_unit_cells_vals
    all_log_s_diff_slopes[L_val] = Dict{Float64, Float64}()
    all_log_s_diff_offsets[L_val] = Dict{Float64, Float64}()
    all_log_s_diff_slope_errs[L_val] = Dict{Float64, Float64}()

    for a_val in decay_a_vals
        if a_val > 0.76
            continue
        end
        println("L: $L_val | a_val: $a_val")
        log_s_diff_to_fit = [val for val in log.(collected_S_diffs[L_val][a_val][times_to_fit[z_fit_val][L_val][a_val][1]:times_to_fit[z_fit_val][L_val][a_val][2]])]

        xs = 1:length(log_s_diff_to_fit)

        df = DataFrame(x=xs, y=log_s_diff_to_fit)
        model = lm(@formula(y ~ x), df)

        all_log_s_diff_offsets[L_val][a_val] = coef(model)[1]
        all_log_s_diff_slopes[L_val][a_val] = coef(model)[2]
        all_log_s_diff_slope_errs[L_val][a_val] = stderror(model)[2]
    end
end

                                              
# --- Ploting Log S_diff PLot again ---
num_unit_cell_to_plot = 64
L_val_to_plot = Int(round(num_unit_cell_to_plot * N_val))
a_vals_to_plot = [0.68, 0.69, 0.7, 0.71, 0.72, 0.74, 0.76]
fitted_a_vals = [0.69]
sholder_a_vals = Dict(0.7 => 76)

# Zoomed in plot
y_lims = (-15, 0)
x_lims = (0, 300)
# Create plot
plt = plot(
    title=L"$\log(S_{\mathrm{diff}})$ for N=%$N_val | L = %$(L_val_to_plot)",
    xlabel=L"t",
    ylabel=L"\log(S_{\mathrm{diff}})",
    ylims=y_lims,
    xlims=x_lims,
    legend=:bottomleft
)

# Plot data for each a_val
for (i,a_val) in enumerate(a_vals_to_plot)
    c = Plots.palette(:auto)[i]

    plot!(plt, log.(collected_S_diffs[L_val_to_plot][a_val][1:round(Int, 350)]),
    yerr = collected_log_S_diff_errs[L_val_to_plot][a_val][1:round(Int, 350)],
        label="a = $(a_val)",
        linestyle=:solid,
        linewidth=1,
        color = c,          # sets line color
        seriescolor = c,    # ensures error bars match
        markerstrokecolor = c)

    if a_val in fitted_a_vals

        xs = 1:x_lims[2]

        plot!(plt, xs, 
            all_log_s_diff_offsets[L_val_to_plot][a_val] .+ (all_log_s_diff_slopes[L_val_to_plot][a_val] .* (xs .- 5)),
            linestyle=:dash,
            label=nothing,
            linewidth=3,
            color = c,          # sets line color
            seriescolor = c,    # ensures error bars match
            markerstrokecolor = c
            )
    end

    if a_val in keys(sholder_a_vals)

        xs = 1:x_lims[2]
        log_a_val = log(a_val)
        offset = sholder_a_vals[a_val]
        plot!(plt, xs, 
            xs .* log_a_val .+ offset,
            linestyle=:dashdotdot,
            linewidth=3,
            label=nothing,
            color = c,          # sets line color
            seriescolor = c,    # ensures error bars match
            markerstrokecolor = c
            )
    end
end

log_s_diff_plot_path = "figs/log_delta_evolved_spins/N$(N_val)/SeveralAs/IC$num_initial_conds/L$(L_val_to_plot)/Zoomed_Log_S_diff$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L_val_to_plot)_z$(z_fit_name).png"
make_path_exist(log_s_diff_plot_path)
savefig(log_s_diff_plot_path)
println("Saved Plot: $(log_s_diff_plot_path)")
display(plt)

