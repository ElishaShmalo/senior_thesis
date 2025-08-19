
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
# num_unit_cells_vals = [8, 16, 32, 64, 128]
num_unit_cells_vals = [8, 16, 32, 64]
# num_unit_cells_vals = [8]

# --- Trying to Replecate Results ---
num_initial_conds = 1000 # We are avraging over x initial conditions
a_vals = [round(0.6 + i*0.01, digits=2) for i in 0:25] # general a_vals
# a_vals = [0.6, 0.7, 0.8] # 0.6, 0.62, 0.64, 0.66, 0.68, 0.7,
# a_vals = [0.75, 0.7525, 0.755, 0.7575, 0.76, 0.7625, 0.765, 0.7675, 0.77] # trans a_vals

epsilon = 0.1

N_val = 4

# --- Lyop Analysis ---

skip_fracts = [1/2, 3/5, 7/10, 7/8, 4/5, 9/10]

for skip_fract in skip_fracts

    # Our dict for recording results
    collected_lambdas = Dict{Int, Dict{Float64, Float64}}() # Int: L_val, Float64: a_val, Float64: avrg lambda
    collected_lambda_SEMs = Dict{Int, Dict{Float64, Float64}}() # Int: L_val, Float64: a_val, Float64: standard error on the mean for lambda

    avrage_window_name = replace("$(round(1-skip_fract, digits=3))", "." => "p")

    # --- Load in and take avrages from all samples ---
    for num_unit_cells in num_unit_cells_vals
        L = num_unit_cells * N_val
        println("L_val: $L")

        # number of pushes we are going to do
        n = Int(round(L^1.6))

        num_skip = Int(round(skip_fract * n)) # we only keep the last L/8 time samples so that the initial condition is properly lost

        # Define s_naught to be used during control step
        S_NAUGHT = make_spiral_state(L, (2) / N_val)

        # Initializes results for this N_val
        collected_lambdas[L] = Dict(a => 0 for a in a_vals)
        collected_lambda_SEMs[L] = Dict(a => 0 for a in a_vals)

        for a_val in a_vals
            println("L_val: $L | a_val: $a_val")
            a_val_name = replace("$a_val", "." => "p")
            # We will avrage over this later
            current_lambdas = zeros(Float64, num_initial_conds)

            for init_cond in 1:num_initial_conds
                current_spin_dists = zeros(n)

                sample_filepath = "data/spin_dists_per_time/N$N_val/a$a_val_name/IC1/L$L/N$(N_val)_a$(a_val_name)_IC1_L$(L)_sample$(init_cond)"
                df = CSV.read(sample_filepath, DataFrame)

                sample_lambdas = df[!, "lambda"]
                current_lambdas[init_cond] = calculate_lambda_from_lambda_per_time(sample_lambdas[num_skip+1:end], tau, n - num_skip)
            end

            collected_lambdas[L][a_val] = mean(current_lambdas)
            collected_lambda_SEMs[L][a_val] = std(current_lambdas)/sqrt(length(current_lambdas))
        end

        filepath = "N$N_val/SeveralAs/IC$num_initial_conds/L$L/" * "N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L)_FracSkip$(skip_fract)"

        # Make large .csv file
        # Extract and sort keys and values
        lambda_dict = collected_lambdas[L]
        sems_dict = collected_lambda_SEMs[L]
        dict_keys = sort(collect(keys(lambda_dict)))

        # Prepare rows: each row is [aval, lambda, lambda_sem]
        rows = [[aval, lambda_dict[aval], sems_dict[aval]] for aval in dict_keys]

        # Make output CSV path
        csv_path = "data/spin_chain_lambdas/" * filepath * ".csv"

        # Write to CSV with header
        open(csv_path, "w") do io
            writedlm(io, [["aval", "lambda", "lambda_sem"]], ',')  # Header
            writedlm(io, rows, ',')                                # Data rows
        end
    end

    # --- Save the plot ---
    println("Making Plot")
    plt = plot()
    plot_path = "N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/lambda_per_a_N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(N_val .* num_unit_cells_vals))_AW$avrage_window_name"

    for L in num_unit_cells_vals * N_val
        L = Int(L)
        plot!(
            sort(a_vals), 
            [val for val in values(sort(collected_lambdas[L]))], 
            yerror=[val for val in values(sort(collected_lambda_SEMs[L]))], 
            marker = :circle, label="L=$L")
    end

    x_vals = range(minimum(a_vals) - 0.005, stop = 1, length = 1000)

    plot!(plt, x_vals, log.(x_vals), linestyle = :dash, label = "ln(a)", title="λ(a) for N=$N_val")

    xlabel!("a")
    ylabel!("λ")
    # display(plt)

    mkpath(dirname("figs/lambda_per_a/" * plot_path))
    savefig("figs/lambda_per_a/" * plot_path * ".png")
    println("Saved Plot: $("figs/lambda_per_a/" * plot_path * ".png")")

    # --- Varience plot ---
    # Create plot
    plt = plot(
        title="Var(λ(a)) for N=$N_val",
        xlabel="a",
        ylabel="Var(λ)"
    )

    # Plot data for each L
    for L in num_unit_cells_vals * N_val
        L = Int(L)
        plot!(plt, a_vals, [val for val in values(sort(collected_lambda_SEMs[L]))] * sqrt(num_initial_conds-1),
            label="L=$L",
            linestyle=:solid,
            markersize=5,
            linewidth=1,
            marker = :circle)
    end

    var_plot_path = "figs/lambda_per_a/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/lambda_per_a_N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(N_val .* num_unit_cells_vals))_AW$(avrage_window_name)_Vars.png"
    make_path_exist(var_plot_path)
    savefig(var_plot_path)
    println("Saved Plot: $(var_plot_path)")

    # Load in avraged results
    for L in num_unit_cells_vals * N_val
        L = Int(L)
        filepath = "N$(N_val)/SeveralAs/IC$num_initial_conds/L$L/" * "N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L)"
        collected_lambdas[L] = open("data/spin_chain_lambdas/" * filepath * ".dat", "r") do io
            deserialize(io)
        end
        collected_lambda_SEMs[L] = open("data/spin_chain_lambdas/" * filepath * "sems.dat", "r") do io
            deserialize(io)
        end
    end

    # --- Colapsing varience plot ---
    # Create plot
    plt = plot(
        title="Colapssing λ(a) for N=$N_val",
        xlabel="a",
        ylabel="Scaled Var(λ)"
    )
    a_crit = 0.76
    nu = 1.1
    # Plot data for each L
    for L in num_unit_cells_vals * N_val
        L = Int(L)
        plot!(plt, (a_vals .- a_crit) .* L^(1/nu), [val for val in values(sort(collected_lambda_SEMs[L]))] * sqrt(num_initial_conds-1),
            label="L=$L",
            linestyle=:solid,
            markersize=5,
            linewidth=1,
            marker = :circle)
    end

    collapsed_var_plot_path = "figs/lambda_per_a/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/lambda_per_a_N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(N_val .* num_unit_cells_vals))_AW$(avrage_window_name)_Vars_Collapsed.png"
    make_path_exist(collapsed_var_plot_path)
    savefig(collapsed_var_plot_path)
    println("Saved Plot: $(collapsed_var_plot_path)")
end

# --- S_diff analysis

collected_S_diffs = Dict{Int, Dict{Float64, Vector{Float64}}}() # Int: L_val, Float64: a_val, Vec{Float64}: avrg S_diff(t)

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

    for a_val in a_vals
        println("L_val: $L | a_val: $a_val")
        a_val_name = replace("$a_val", "." => "p")
        # We will avrage over this later
        current_S_diffs = zeros(Float64, n)

        for init_cond in 1:num_initial_conds

            sample_filepath = "data/spin_dists_per_time/N$N_val/a$a_val_name/IC1/L$L/N$(N_val)_a$(a_val_name)_IC1_L$(L)_sample$(init_cond)"
            df = CSV.read(sample_filepath, DataFrame)

            current_S_diffs = current_S_diffs .+ df[!, "delta_s"]
        end

        collected_S_diffs[L][a_val] = current_S_diffs ./ num_initial_conds
    end
end

# --- Making S_diff Plots ---

num_unit_cell_to_plot = num_unit_cells_vals[end]
L_val_to_plot = Int(round(num_unit_cell_to_plot * N_val))

# Create plot
plt = plot(
    title="SDiff for N=$N_val | L = $(L_val_to_plot)",
    xlabel="t",
    ylabel="S_Diif"
)
a_vals_to_plot = [0.6, 0.65, 0.68, 0.7, 0.71, 0.73, 0.76, 0.77]
# Plot data for each a_val
for a_val in a_vals_to_plot

    plot!(plt, collected_S_diffs[L_val_to_plot][a_val],
        label="a = $(a_val)",
        linestyle=:solid,
        linewidth=1,)
end

s_diff_plot_path = "figs/delta_evolved_spins/N$(N_val)/SeveralAs/IC$num_initial_conds/L$(L_val_to_plot)/S_diff$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L_val_to_plot).png"
make_path_exist(s_diff_plot_path)
savefig(s_diff_plot_path)
println("Saved Plot: $(s_diff_plot_path)")

# --- Making Log(S_Diff) Plots ---
L_val_to_plot = Int(round(num_unit_cell_to_plot * N_val))

a_vals_to_plot = [0.6, 0.65, 0.68, 0.7, 0.71, 0.73, 0.76, 0.77]

t_limit = Int(round(min(2000, L_val_to_plot^1.6)))

# Create plot
plt = plot(
    title="Log(SDiff )for N=$N_val | L = $(L_val_to_plot)",
    xlabel="t",
    ylabel="Log(S_Diif)",
)

# Plot data for each a_val
for a_val in a_vals_to_plot

    plot!(plt, log.(collected_S_diffs[L_val_to_plot][a_val][1:t_limit]),
        label="a = $(a_val)",
        linestyle=:solid,
        linewidth=1,)
end

log_s_diff_plot_path = "figs/log_delta_evolved_spins/N$(N_val)/SeveralAs/IC$num_initial_conds/L$(L_val_to_plot)/Log_S_diff$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L_val_to_plot).png"
make_path_exist(log_s_diff_plot_path)
savefig(log_s_diff_plot_path)
println("Saved Plot: $(log_s_diff_plot_path)")

# Zoomed in plot
y_lims = (-15, 0)

# Create plot
plt = plot(
    title="Log(SDiff )for N=$N_val | L = $(L_val_to_plot)",
    xlabel="t",
    ylabel="Log(S_Diif)",
    ylims=y_lims
)

# Plot data for each a_val
for a_val in a_vals_to_plot

    plot!(plt, log.(collected_S_diffs[L_val_to_plot][a_val][1:t_limit]),
        label="a = $(a_val)",
        linestyle=:solid,
        linewidth=1,)
end

log_s_diff_plot_path = "figs/log_delta_evolved_spins/N$(N_val)/SeveralAs/IC$num_initial_conds/L$(L_val_to_plot)/Zoomed_Log_S_diff$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L_val_to_plot).png"
make_path_exist(log_s_diff_plot_path)
savefig(log_s_diff_plot_path)
println("Saved Plot: $(log_s_diff_plot_path)")

# --- Fitting Log of S_diff ---
# From the analysis of the plots, it seems that the first linear domain is as long as Log(S_diff) > some value, so we will 
# Fit accordingly

min_val = -7

# Getting data to fit
# Plot data for each a_val
log_s_diff_to_fit = Dict{Float64, Vector{Float64}}()
for a_val in a_vals_to_plot
    println("a_val: $(a_val)")
    log_s_diff_to_fit[a_val] = [val for val in log.(collected_S_diffs[L_val_to_plot][a_val][1:t_limit]) if val >= min_val]
end

log_s_diff_fits = Dict{Float64, Vector{Float64}}()
# Get slope and intercept for each a_val
for a_val in a_vals_to_plot
    println("Fitting for a_val: $(a_val)")
    data = log_s_diff_to_fit[a_val]
    xs = 1:length(data)

    df = DataFrame(x=xs, y=data)
    model = lm(@formula(y ~ x), df)

    log_s_diff_fits[a_val] = coef(model)  # gives [intercept, slope]
end

# Plotting the old data along with the fits
# Create plot
plt = plot(
    title="Log(SDiff) for N=$N_val | L = $(L_val_to_plot)",
    xlabel="t",
    ylabel="Log(S_Diif)",
)

# Plot data for each a_val
for a_val in a_vals_to_plot
    xs = 1:length(log_s_diff_to_fit[a_val])

    plot!(plt, log_s_diff_to_fit[a_val],
        label="a = $(a_val)",
        linestyle=:solid,
        linewidth=1,)

    plot!(plt, xs, log_s_diff_fits[a_val][2] .* xs .+ log_s_diff_fits[a_val][1],
        label="a = $(a_val)",
        linestyle=:dash,
        linewidth=1,)
end

fitted_log_s_diff_plot_path = "figs/log_delta_evolved_spins/N$(N_val)/SeveralAs/IC$num_initial_conds/L$(L_val_to_plot)/Fitted_Log_S_diff$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L_val_to_plot).png"
make_path_exist(fitted_log_s_diff_plot_path)
savefig(fitted_log_s_diff_plot_path)
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
min_val = -10

# a_vals = [val for val in a_vals if 0.74 <= val <= 0.78]

# Calculating the fits
all_log_s_diff_slopes = Dict{Int, Dict{Float64, Float64}}()

for L_val in N_val .* num_unit_cells_vals
    all_log_s_diff_slopes[L_val] = Dict{Float64, Float64}()
    for a_val in a_vals
        log_s_diff_to_fit = [val for val in log.(collected_S_diffs[L_val][a_val]) if val >= min_val]

        xs = 1:length(log_s_diff_to_fit)

        df = DataFrame(x=xs, y=log_s_diff_to_fit)
        model = lm(@formula(y ~ x), df)

        all_log_s_diff_slopes[L_val][a_val] = coef(model)[2]  # gives [intercept, slope]
    end
end

# Create plot
plt = plot(
    title="Decay As Func of a",
    xlabel="a",
    ylabel="Decay"
)

# Plot data for each L
for L in num_unit_cells_vals * N_val
    L = Int(L)
    plot!(plt, a_vals, [1/val for val in values(sort(all_log_s_diff_slopes[L]))],
        label="L=$L",
        linestyle=:solid,
        markersize=5,
        linewidth=1,
        marker = :circle)
end

decay_per_a_plot_path = "figs/decay_per_a/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/decay_per_a_N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(N_val .* num_unit_cells_vals)).png"
make_path_exist(decay_per_a_plot_path)
savefig(decay_per_a_plot_path)
println("Saved Plot: $(decay_per_a_plot_path)")
