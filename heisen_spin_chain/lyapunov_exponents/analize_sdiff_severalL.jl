# --- S_diff analysis ---

# Imports
using Random, LinearAlgebra, Plots, DifferentialEquations, Serialization, Statistics, DelimitedFiles, SharedArrays, CSV, DataFrames, GLM

# Other files   
include("../utils/make_spins.jl")
include("../utils/general.jl")
include("../utils/dynamics.jl")
include("../utils/lyapunov.jl")
include("../analytics/spin_diffrences.jl")


# Set plotting theme
Plots.theme(:dark)

J = 1    # energy factor

# J vector with some randomness
J_vec = J .* [1, 1, 1]

# Time to evolve until push back to S_A
tau = 1 * J

# General Variables
num_unit_cells_vals = [8, 16, 32, 64, 128]
# num_unit_cells_vals = [8, 16, 32, 64]
# num_unit_cells_vals = [8]

# --- Trying to Replecate Results ---
num_initial_conds = 1000 # We are avraging over x initial conditions
trans_a_vals = [0.7525, 0.755, 0.7575, 0.76, 0.7625, 0.765, 0.7675, 0.77]
post_a_vals = [round(0.8 + i * 0.02, digits=2) for i in 0:5]
a_vals = sort(union([round(0.6 + i*0.01, digits=2) for i in 0:20], [0.7525, 0.755, 0.7575, 0.7625, 0.765, 0.7675], [0.763], post_a_vals)) # general a_vals
# a_vals = sort(union([round(0.7 + i*0.01, digits=2) for i in 0:12], trans_a_vals)) # 0.6, 0.62, 0.64, 0.66, 0.68, 0.7,
a_vals = [0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.7525, 0.755, 0.7575, 0.76, 0.7625, 0.763, 0.765, 0.7675, 0.77, 0.78, 0.79]

epsilon = 0.1

N_val = 4

z_val = 1.6
z_val_name = replace("$z_val", "." => "p")

collected_S_diffs = Dict{Int, Dict{Float64, Vector{Float64}}}() # Int: L_val, Float64: a_val, Vec{Float64}: avrg S_diff(t)
collected_S_diff_SEMs = Dict{Int, Dict{Float64, Vector{Float64}}}() # Int: L_val, Float64: a_val, Vec{Float64}: avrg S_diff(t)

# --- Load in and take avrages from all samples ---
for num_unit_cells in num_unit_cells_vals
    L = num_unit_cells * N_val
    println("L_val: $L")

    # number of pushes we are going to do
    n = Int(round(L^1.6))

    # Define s_naught to be used during control step
    S_NAUGHT = make_spiral_state(L, (2) / N_val)

    # Initializes results for this N_val
    collected_S_diffs[L] = Dict(a_val => zeros(n) for a_val in a_vals)
    collected_S_diff_SEMs[L] = Dict(a_val => zeros(n) for a_val in a_vals)

    for a_val in a_vals
        println("L_val: $L | a_val: $a_val")
        a_val_name = replace("$a_val", "." => "p")
        # We will avrage over this later
        current_S_diffs = [zeros(Float64, n) for _ in 1:num_initial_conds]

        for init_cond in 1:num_initial_conds

            sample_filepath = "data/spin_dists_per_time/N$N_val/a$a_val_name/IC1/L$L/N$(N_val)_a$(a_val_name)_IC1_L$(L)_z$(z_val_name)_sample$(init_cond).csv"
            df = CSV.read(sample_filepath, DataFrame)

            current_S_diffs[init_cond] = df[!, "delta_s"]
        end

        collected_S_diffs[L][a_val] = mean(current_S_diffs)
        collected_S_diff_SEMs[L][a_val] = std(current_S_diffs) / sqrt(num_initial_conds)
    end
end

# --- Making S_diff Plots ---
a_vals_to_plot = [0.755, 0.7575, 0.76, 0.7625, 0.765]
num_unit_cell_to_plot = num_unit_cells_vals[end]
L_val_to_plot = Int(round(num_unit_cell_to_plot * N_val))

# Create plot
plt = plot(
    title="SDiff for N=$N_val | L = $(L_val_to_plot)",
    xlabel="t",
    ylabel="S_Diff"
)

# Plot data for each a_val
for (i, a_val) in enumerate(a_vals_to_plot)
    # pick the color Plots.jl will use for series i
    c = Plots.palette(:auto)[i]

    plot!(plt, collected_S_diffs[L_val_to_plot][a_val][1:round(Int, L_val_to_plot^z_val)],
        yerr=collected_S_diff_SEMs[L_val_to_plot][a_val][1:round(Int, L_val_to_plot^z_val)],
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
savefig(s_diff_plot_path)
println("Saved Plot: $(s_diff_plot_path)")
display(plt)

# --- Making S_diff(t=L^z) as function of a ---
z_val = 1.6
z_val_name = replace("$(z_val)", "." => "p")

plt = plot(
    title="S_Diff(t=L^$(z_val)) for N=$N_val as Function of a",
    xlabel="a",
    ylabel="S_Diff(t=L^$(z_val))"
)

L_vals_to_plot = Int.(round.(num_unit_cells_vals * N_val))

# Plot data for each a_val
for (i, L_val) in enumerate(L_vals_to_plot)
    # pick the color Plots.jl will use for series i
    c = Plots.palette(:auto)[i]


    plot!(plt, [a_val for a_val in sort(a_vals)], [collected_S_diffs[L_val][a_val][round(Int, L_val^z_val)] for a_val in sort(a_vals)],
        yerr=[collected_S_diff_SEMs[L_val][a_val][round(Int, L_val^z_val)] for a_val in sort(a_vals)],
        label="L = $(L_val)",
        linestyle=:solid,
        linewidth=1,
        color = c,          # sets line color
        seriescolor = c,    # ensures error bars match
        markerstrokecolor = c # (optional) markers match too
        )
end

s_diff_per_val_plot_path = "figs/delta_evolved_spins/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/S_diff_per_aval_N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(num_unit_cells_vals .* N_val))_z$(z_val_name).png"
make_path_exist(s_diff_per_val_plot_path)
savefig(s_diff_per_val_plot_path)
println("Saved Plot: $(s_diff_per_val_plot_path)")
display(plt)

# --- Collapsing S_Diff as a function of a ---

a_vals_to_collapse = [a_val for a_val in a_vals if 0.75 <= a_val <=0.77]
a_vals_to_collapse = [a_val for a_val in a_vals]

z_val = 1.6
z_val_name = replace("$(z_val)", "." => "p")

a_crit = 0.76088130 # pm 1.6626e-04
nu = 1.52703636 #pm 0.04538342	
beta = 0.32002792 #pm 0.02029257

a_crit = 0.76128506 # pm 5.4926e-05
nu = 1.69861984 #pm 0.02605228
beta = 0.18004203 #pm 0.01141406

plt = plot(
    title="FSS S_Diff(t=L^$(z_val)) | N=$N_val, β=$(round(beta, digits=3)), ν=$(round(nu, digits=3)), a_crit=$(round(a_crit, digits = 3))",
    xlabel="(a-a_c)L^{1/ν}",
    ylabel="S_Diff(t=L^$(z_val)) L^{β/ν}"
)

L_vals_to_plot = Int.(round.(num_unit_cells_vals * N_val))

# Plot data for each a_val
for (i, L_val) in enumerate(L_vals_to_plot)
    # pick the color Plots.jl will use for series i
    c = Plots.palette(:auto)[i]

    xs = [a_val - a_crit for a_val in sort(a_vals_to_collapse)] .* (L_val ^ (1/nu))
    ys = [collected_S_diffs[L_val][a_val][round(Int, L_val^z_val)] for a_val in sort(a_vals_to_collapse)] .* (L_val^(beta / nu))
    y_errs = sqrt.([collected_S_diff_SEMs[L_val][a_val][round(Int, L_val^z_val)] for a_val in sort(a_vals_to_collapse)].^2 .+ (log(L_val) * delta_z/(L_val^(beta/nu)))^2)
    plot!(plt, xs, ys,
        # yerr=y_errs,
        label="L = $(L_val)",
        linestyle=:solid,
        linewidth=1,
        color = c,          # sets line color
        seriescolor = c,    # ensures error bars match
        markerstrokecolor = c, # (optional) markers match too
        marker = :circle, markersize=2,
        )
end

fss_s_diff_per_val_plot_path = "figs/delta_evolved_spins/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/fss_S_diff_per_aval_N$(N_val)_ar$(replace("$(minimum(a_vals_to_collapse))_$(maximum(a_vals_to_collapse))", "." => "p"))_IC$(num_initial_conds)_L$(join(num_unit_cells_vals .* N_val))_z$(z_val_name).png"
make_path_exist(fss_s_diff_per_val_plot_path)
savefig(fss_s_diff_per_val_plot_path)
println("Saved Plot: $(fss_s_diff_per_val_plot_path)")
display(plt)

# --- Making Log(S_Diff) Plots ---

# First we calculate the error on the log(s_diff)
collected_log_S_diff_errs = Dict{Int, Dict{Float64, Vector{Float64}}}() # Int: L_val, Float64: a_val, Vec{Float64}: avrg S_diff(t)
for L_val in num_unit_cells_vals * N_val
    println("L_val $(L_val)")
    collected_log_S_diff_errs[L_val] = Dict{Float64, Vector{Float64}}()
    for a_val in a_vals
        collected_log_S_diff_errs[L_val][a_val] = (1 ./ (collected_S_diffs[L_val][a_val])) .* collected_S_diff_SEMs[L_val][a_val]
    end
end

num_unit_cell_to_plot = 64

L_val_to_plot = Int(round(num_unit_cell_to_plot * N_val))

a_vals_to_plot = [0.6, 0.65, 0.68, 0.72, 0.76, 0.763, 0.77]

# Create plot
plt = plot(
    title="Log(SDiff) for N=$N_val | L = $(L_val_to_plot)",
    xlabel="t",
    ylabel="Log(S_Diff)",
)

# Plot data for each a_val
for (i, a_val) in enumerate(a_vals_to_plot)
    c = Plots.palette(:auto)[i]

    plot!(plt, log.(collected_S_diffs[L_val_to_plot][a_val][1:round(Int, L_val_to_plot)]),
        yerr = collected_log_S_diff_errs[L_val_to_plot][a_val][1:round(Int, L_val_to_plot)],
        label="a = $(a_val)",
        linestyle=:solid,
        linewidth=1,
        color = c,          # sets line color
        seriescolor = c,    # ensures error bars match
        markerstrokecolor = c) # (optional) markers match too
end

log_s_diff_plot_path = "figs/log_delta_evolved_spins/N$(N_val)/SeveralAs/IC$num_initial_conds/L$(L_val_to_plot)/Log_S_diff$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L_val_to_plot).png"
make_path_exist(log_s_diff_plot_path)
savefig(log_s_diff_plot_path)
println("Saved Plot: $(log_s_diff_plot_path)")
display(plt)

# Zoomed in plot
num_unit_cell_to_plot = 128
L_val_to_plot = num_unit_cell_to_plot * N_val
y_lims = (-30, 0)

# Create plot
plt = plot(
    title="Log(SDiff) for N=$N_val | L = $(L_val_to_plot)",
    xlabel="t",
    ylabel="Log(S_Diif)",
    ylims=y_lims,
)

# Plot data for each a_val
for (i,a_val) in enumerate(a_vals_to_plot)
    c = Plots.palette(:auto)[i]

    plot!(plt, log.(collected_S_diffs[L_val_to_plot][a_val][1:round(Int, L_val_to_plot)]),
    yerr = collected_log_S_diff_errs[L_val_to_plot][a_val][1:round(Int, L_val_to_plot)],
        label="a = $(a_val)",
        linestyle=:solid,
        linewidth=1,
        color = c,          # sets line color
        seriescolor = c,    # ensures error bars match
        markerstrokecolor = c)
end

log_s_diff_plot_path = "figs/log_delta_evolved_spins/N$(N_val)/SeveralAs/IC$num_initial_conds/L$(L_val_to_plot)/Zoomed_Log_S_diff$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L_val_to_plot).png"
make_path_exist(log_s_diff_plot_path)
savefig(log_s_diff_plot_path)
println("Saved Plot: $(log_s_diff_plot_path)")
display(plt)

# --- Fitting Log of S_diff ---
# From the analysis of the plots, it seems that the first linear domain is as long as Log(S_diff) > some value, so we will 
# Fit accordingly

times_to_fit = Dict{Int, Dict{Float64, Vector{Int}}}()
times_to_fit[512] = Dict{Float64, Vector{Int}}(0.67 => [5, 80],
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
times_to_fit[256] = Dict{Float64, Vector{Int}}(0.67 => [5, 70],
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
times_to_fit[128] = Dict{Float64, Vector{Int}}(0.67 => [5, 60],
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
times_to_fit[64] = Dict{Float64, Vector{Int}}(0.67 => [1, 65],
                                               0.68 => [1, 90], 
                                               0.69 => [1, 100], 
                                               0.7 => [1, 140], 
                                               0.71 => [1, 220], 
                                               0.72 => [1, 280], 
                                               0.73 => [1, 340], 
                                               0.74 => [1, Int(round(64^1.6))], 
                                               0.75 => [1, Int(round(64^1.6))], 
                                               0.7525 => [1, Int(round(64^1.6))], 
                                               0.755 => [1, Int(round(64^1.6))], 
                                               0.7575 => [1, Int(round(64^1.6))], 
                                               0.76 => [1, Int(round(64^1.6))], 
                                               0.7625 => [1, Int(round(64^1.6))], 
                                               0.763=> [1, Int(round(64^1.6))],)

times_to_fit[32] = Dict{Float64, Vector{Int}}(0.67 => [1, 60],
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


# Getting data to fit
# Plot data for each a_val
L_val_to_plot = 256
a_vals_to_plot = sort([a_val for a_val in keys(times_to_fit[L_val_to_plot])])
log_s_diff_to_fit = Dict{Float64, Vector{Float64}}()
for a_val in a_vals_to_plot
    println("a_val: $(a_val)")
    if  !in(a_val, a_vals)
        continue
    end
    log_s_diff_to_fit[a_val] = [val for val in log.(collected_S_diffs[L_val_to_plot][a_val][times_to_fit[L_val_to_plot][a_val][1]:times_to_fit[L_val_to_plot][a_val][2]])]
end

log_s_diff_fits = Dict{Float64, Vector{Float64}}()
log_s_diff_fit_errors = Dict{Float64, Vector{Float64}}()  # store [intercept_err, slope_err]

# Get slope and intercept for each a_val
for a_val in a_vals_to_plot
    if !in(a_val, a_vals)
        continue
    end
    println("Fitting for a_val: $(a_val)")
    data = log_s_diff_to_fit[a_val]
    xs = 1:length(data)

    df = DataFrame(x=xs, y=data)
    model = lm(@formula(y ~ x), df)

    log_s_diff_fits[a_val] = coef(model)  # gives [intercept, slope]
    log_s_diff_fit_errors[a_val] = stderror(model)    
end

# Plotting the old data along with the fits
# Create plot
plt = plot(
    title="Log(SDiff) for N=$N_val | L = $(L_val_to_plot)",
    xlabel="t",
    ylabel="Log(S_Diif)",
)

# Plot data for each a_val
for (i, a_val) in enumerate(a_vals_to_plot)
    if !in(a_val, a_vals)
        continue
    end
    c = Plots.palette(:auto)[i]

    xs = 1:length(log_s_diff_to_fit[a_val])

    plot!(plt, xs, log_s_diff_to_fit[a_val],
        label="a = $(a_val)",
        linestyle=:solid,
        linewidth=1,)

    plot!(plt, xs, log_s_diff_fits[a_val][2] .* xs .+ log_s_diff_fits[a_val][1],
        yerr = log_s_diff_fit_errors[a_val][2] .* xs .+ log_s_diff_fit_errors[a_val][1],
        label="a = $(a_val)",
        linestyle=:dash,
        linewidth=1,
        color = c,          # sets line color
        seriescolor = c,    # ensures error bars match
        markerstrokecolor = c)
end

fitted_log_s_diff_plot_path = "figs/log_delta_evolved_spins/N$(N_val)/SeveralAs/IC$num_initial_conds/L$(L_val_to_plot)/Fitted_Log_S_diff$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L_val_to_plot).png"
make_path_exist(fitted_log_s_diff_plot_path)
savefig(fitted_log_s_diff_plot_path)
display(plt)
println("Saved Plot: $(fitted_log_s_diff_plot_path)")

# Making fit params csv file
# Build a DataFrame with columns: a_val, intercept, slope
fit_df = DataFrame(
    a_val = Float64[],
    intercept = Float64[],
    slope = Float64[]
)

for (a_val, coeffs) in log_s_diff_fits
    push!(fit_df, (a_val, coeffs[1], coeffs[2]))
end

# Save to CSV
fitted_log_s_diff_csv_path = "data/log_delta_evolved_spins/N$(N_val)/SeveralAs/IC$num_initial_conds/L$(L_val_to_plot)/Fit_Params_Log_S_diff$(N_val)_ar$(replace("$(minimum(a_vals_to_plot))_$(maximum(a_vals_to_plot))", "." => "p"))_IC$(num_initial_conds)_L$(L_val_to_plot).csv"
make_path_exist(fitted_log_s_diff_csv_path)
CSV.write(fitted_log_s_diff_csv_path, fit_df)


# --- Decay Timescale as Func of a for all L --- 

decay_a_vals = [val for val in a_vals if 0.67 <= val <= 0.76]

# Calculating the fits
all_log_s_diff_slopes = Dict{Int, Dict{Float64, Float64}}()
all_log_s_diff_slope_errs = Dict{Int, Dict{Float64, Float64}}()

for L_val in N_val .* num_unit_cells_vals
    all_log_s_diff_slopes[L_val] = Dict{Float64, Float64}()
    all_log_s_diff_slope_errs[L_val] = Dict{Float64, Float64}()

    for a_val in decay_a_vals
        println("L: $L_val | a_val: $a_val")
        log_s_diff_to_fit = [val for val in log.(collected_S_diffs[L_val][a_val][times_to_fit[L_val][a_val][1]:times_to_fit[L_val][a_val][2]])]

        xs = 1:length(log_s_diff_to_fit)

        df = DataFrame(x=xs, y=log_s_diff_to_fit)
        model = lm(@formula(y ~ x), df)

        all_log_s_diff_slopes[L_val][a_val] = coef(model)[2]
        all_log_s_diff_slope_errs[L_val][a_val] = stderror(model)[2]
    end
end

# Create plot
plt = plot(
    title="ξ_τ As Func of a",
    xlabel="a",
    ylabel="ξ_τ"
)

# Plot data for each L
for (i, L) in enumerate(num_unit_cells_vals * N_val)
    c = Plots.palette(:auto)[i]
    L = Int(L)
    plot!(plt, sort([a_val for a_val in decay_a_vals]), [-1/val for val in values(sort(all_log_s_diff_slopes[L]))],
        yerr = [(1/(val^2)) * val_err for (val, val_err) in zip(values(sort(all_log_s_diff_slopes[L])), values(sort(all_log_s_diff_slope_errs[L])))],
        label="L=$L",
        linestyle=:solid,
        linewidth=1,
        marker = :circle,
        color = c,          # sets line color
        seriescolor = c,    # ensures error bars match
        markerstrokecolor = c)
end

decay_per_a_plot_path = "figs/decay_per_a/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/decay_per_a_N$(N_val)_ar$(replace("$(minimum(decay_a_vals))_$(maximum(decay_a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(N_val .* num_unit_cells_vals)).png"
make_path_exist(decay_per_a_plot_path)
savefig(decay_per_a_plot_path)
display(plt)
println("Saved Plot: $(decay_per_a_plot_path)")

# Log of decay Timescale

a_crit = 0.76

# Create plot
plt = plot(
    title="log(ξ_τ) as Function of a",
    xlabel="log(|a - a_c|)",
    ylabel="log(ξ_τ)"
)

# Plot data for each L
for (i, L) in enumerate(num_unit_cells_vals * N_val)
    c = Plots.palette(:auto)[i]
    L = Int(L)
    plot!(plt, log.([abs.(a_val-a_crit) for a_val in sort(decay_a_vals)]), log.([-1/all_log_s_diff_slopes[L][a_val] for a_val in sort(decay_a_vals)]),
        yerr = [(1/(val)) * val_err for (val, val_err) in zip(values(sort(all_log_s_diff_slopes[L])), values(sort(all_log_s_diff_slope_errs[L])))],
        label="L=$L",
        linestyle=:solid,
        markersize=2,
        linewidth=1,
        marker = :circle,
        color = c,          # sets line color
        seriescolor = c,    # ensures error bars match
        markerstrokecolor = c)
end

log_scaled_decay_per_a_plot_path = "figs/decay_per_a/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/log_decay_per_a_N$(N_val)_ar$(replace("$(minimum(decay_a_vals))_$(maximum(decay_a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(N_val .* num_unit_cells_vals)).png"
make_path_exist(log_scaled_decay_per_a_plot_path)
savefig(log_scaled_decay_per_a_plot_path)
println("Saved Plot: $(log_scaled_decay_per_a_plot_path)")

# --- Scaled Decay Timescale as Func of a for all L --- 

decay_a_vals = [val for val in a_vals if 0.67 <= val < 0.7575]

# save_data_to_collapse
for L in num_unit_cells_vals * N_val
    L = Int(L)
    data = Dict{Float64, Float64}()
    for a_val in decay_a_vals
        data[a_val] = -1/all_log_s_diff_slopes[L][a_val]
    end
    filepath = "data_to_collapse/decay_per_a/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/decay_per_a_N$(N_val)_ar$(replace("$(minimum(decay_a_vals))_$(maximum(decay_a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L).csv"
    save_simple_dict_to_csv(data, filepath)
end

# Plotting the collapse
a_crit = 0.76150000  # pm 4.6599e-04
nu = 1.59192864 # pm 0.05413757
beta = -2.63062017	# pm 0.09477604
z = -beta/nu

a_crit = 0.76083967  # pm 4.6599e-04
nu = 1.61187585 # pm 0.05413757
beta = -2.62542143	# pm 0.09477604
z = -beta/nu
println(z)
println(1/nu * sqrt((0.08997456)^2 + (z * 0.03292336)^2))

# Create plot
plt = plot(
    title="Scaled ξ_τ (z=$(round(z, digits=3)),nu = $(round(nu, digits = 3)),a_c = $(round(a_crit, digits = 4)))",
    xlabel="(a - a_c)L^(1/ν)",
    ylabel="ξ_τ / L^{z}"
)

# Plot data for each L
for (i, L) in enumerate(num_unit_cells_vals * N_val)
    c = Plots.palette(:auto)[i]
    L = Int(L)
    plot!(plt, sort([a_val-a_crit for a_val in decay_a_vals]) .* L^(1/nu), [-1/all_log_s_diff_slopes[L][a_val] for a_val in sort(decay_a_vals)] ./ (L^z),
        yerr = [(1/(val^2 * L^z)) * val_err for (val, val_err) in zip(values(sort(all_log_s_diff_slopes[L])), values(sort(all_log_s_diff_slope_errs[L])))],
        label="L=$L",
        linestyle=:solid,
        markersize=2,
        linewidth=1,
        marker = :circle,
        color = c,          # sets line color
        seriescolor = c,    # ensures error bars match
        markerstrokecolor = c)
end

scaled_decay_per_a_plot_path = "figs/decay_per_a/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/scaled_decay_per_a_N$(N_val)_ar$(replace("$(minimum(decay_a_vals))_$(maximum(decay_a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(N_val .* num_unit_cells_vals)).png"
make_path_exist(scaled_decay_per_a_plot_path)
savefig(scaled_decay_per_a_plot_path)
println("Saved Plot: $(scaled_decay_per_a_plot_path)")
display(plt)

# Plotting log-log of the collapse

# a_crit = 0.7623
# nu = 1.6
# z = 1.6

# Create plot
plt = plot(
    title="Scaled log(ξ_τ) (z=$(round(z, digits=3)),nu = $(round(nu, digits = 3)),a_c = $(round(a_crit, digits = 4)))",
    xlabel="log(|a - a_c|L^(1/ν))",
    ylabel="log(ξ_τ / L^{z})"
)

# Plot data for each L
for (i, L) in enumerate(num_unit_cells_vals * N_val)
    c = Plots.palette(:auto)[i]
    L = Int(L)
    plot!(plt, log.([abs.(a_val-a_crit) for a_val in sort(decay_a_vals)] .* L^(1/nu)), log.([-1/all_log_s_diff_slopes[L][a_val] for a_val in sort(decay_a_vals)] ./ (L^z)),
        yerr = [(1/(val)) * val_err for (val, val_err) in zip(values(sort(all_log_s_diff_slopes[L])), values(sort(all_log_s_diff_slope_errs[L])))],
        label="L=$L",
        linestyle=:solid,
        markersize=2,
        linewidth=1,
        marker = :circle,
        color = c,          # sets line color
        seriescolor = c,    # ensures error bars match
        markerstrokecolor = c)
end

log_scaled_decay_per_a_plot_path = "figs/decay_per_a/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/log_scaled_decay_per_a_N$(N_val)_ar$(replace("$(minimum(decay_a_vals))_$(maximum(decay_a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(N_val .* num_unit_cells_vals)).png"
make_path_exist(log_scaled_decay_per_a_plot_path)
savefig(log_scaled_decay_per_a_plot_path)
println("Saved Plot: $(log_scaled_decay_per_a_plot_path)")
display(plt)

# So far results:
# Collapse of Var(lambda)
# 0.76224524, nu = 1.8, z = 1.6

# Collapse of ξ_τ
# a_c = 0.76238998, nu = 1.82590603	

# No errors given