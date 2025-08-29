
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
num_unit_cells_vals = [8, 16, 32, 64, 128]
# num_unit_cells_vals = [8]

# --- Trying to Replecate Results ---
num_initial_conds = 1000 # We are avraging over x initial conditions
trans_a_vals = [0.72, 0.73, 0.74, 0.75, 0.7525, 0.755, 0.7575, 0.76, 0.7625, 0.765, 0.7675, 0.77, 0.78, 0.79, 0.8]
# a_vals = sort(union([round(0.6 + i*0.01, digits=2) for i in 0:25], [0.7525, 0.755, 0.7575, 0.7625, 0.765, 0.7675])) # general a_vals
a_vals = sort(union([round(0.7 + i*0.01, digits=2) for i in 0:12], trans_a_vals)) # 0.6, 0.62, 0.64, 0.66, 0.68, 0.7,
# a_vals = [0.75, 0.7525, 0.755, 0.7575, 0.76, 0.7625, 0.765, 0.7675, 0.77] # trans a_vals

epsilon = 0.1

N_val = 4

# --- Lyop Analysis ---
avraging_windows = [1/32]

collected_lambdas = Dict{Float64, Dict{Int, Dict{Float64, Float64}}}()
collected_lambdas_SEMs = Dict{Float64, Dict{Int, Dict{Float64, Float64}}}()
for avraging_window in avraging_windows
    skip_fract = 1 - avraging_window
    avraging_window_name = replace("$(round(avraging_window, digits=3))", "." => "p")

    # Our dict for recording results
    current_collected_lambdas = Dict{Int, Dict{Float64, Float64}}() # Int: L_val, Float64: a_val, Float64: avrg lambda
    current_collected_lambda_SEMs = Dict{Int, Dict{Float64, Float64}}() # Int: L_val, Float64: a_val, Float64: standard error on the mean for lambda

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
        current_collected_lambdas[L] = Dict(a => 0 for a in a_vals)
        current_collected_lambda_SEMs[L] = Dict(a => 0 for a in a_vals)

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

            current_collected_lambdas[L][a_val] = mean(current_lambdas)
            current_collected_lambda_SEMs[L][a_val] = std(current_lambdas)/sqrt(length(current_lambdas))
        end

        filepath = "N$N_val/SeveralAs/IC$num_initial_conds/L$L/" * "N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L)_FracSkip$(skip_fract)"

        # Make large .csv file
        # Extract and sort keys and values
        lambda_dict = current_collected_lambdas[L]
        sems_dict = current_collected_lambda_SEMs[L]
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
    collected_lambdas[avraging_window] = current_collected_lambdas
    collected_lambdas_SEMs[avraging_window] = current_collected_lambda_SEMs
end

# save_data_to_collapse
for avraging_window in avraging_windows
    avraging_window_name = replace("$(round(avraging_window, digits=3))", "." => "p")

    for L in num_unit_cells_vals * N_val
        L = Int(L)
        data = Dict{Float64, Float64}()
        for a_val in trans_a_vals
            data[a_val] = collected_lambdas_SEMs[avraging_window][L][a_val] .* sqrt(num_initial_conds-1)
        end
        filepath = "data_to_collapse/lambda_var_per_a/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/lambda_var_per_a_N$(N_val)_ar$(replace("$(minimum(trans_a_vals))_$(maximum(trans_a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L)_AW$(avraging_window_name).csv"
        save_simple_dict_to_csv(data, filepath)
    end
end

# --- Using Loaded Data ---
avraging_window = 1/32
avraging_window_name = replace("$(round(avraging_window, digits=3))", "." => "p")

# --- Save the plot ---
println("Making Plot")
plt = plot()
plot_path = "N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/lambda_per_a_N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(N_val .* num_unit_cells_vals))_AW$avraging_window_name"

for L in num_unit_cells_vals * N_val
    L = Int(L)
    plot!(
        sort(a_vals), 
        [val for val in values(sort(collected_lambdas[avraging_window][L]))], 
        yerror=[val for val in values(sort(collected_lambdas_SEMs[avraging_window][L]))], 
        marker = :circle, markersize=2, label="L=$L")
end

x_vals = range(minimum(a_vals) - 0.005, stop = 1, length = 1000)

plot!(plt, x_vals, log.(x_vals), linestyle = :dash, label = "ln(a)", title="λ(a) for N=$N_val")
vline!(plt, [0.763], color=:red, lw=1) 

xlabel!("a")
ylabel!("λ")
# display(plt)

mkpath(dirname("figs/lambda_per_a/" * plot_path))
savefig("figs/lambda_per_a/" * plot_path * ".png")
println("Saved Plot: $("figs/lambda_per_a/" * plot_path * ".png")")

# --- Std plot ---
# Create plot
plt = plot(
    title="Std(λ(a)) for N=$N_val | AW=$avraging_window_name",
    xlabel="a",
    ylabel="Std(λ)",
    xticks = minimum(a_vals):0.02:maximum(a_vals)
)

# Plot data for each L
for L in num_unit_cells_vals * N_val
    L = Int(L)
    if L != 32
        continue
    end
    plot!(plt, a_vals, [val for val in values(sort(collected_lambdas_SEMs[avraging_window][L]))] * sqrt(num_initial_conds-1),
        label="L=$L",
        linestyle=:solid,
        markersize=2,
        linewidth=1,
        marker = :circle)
end

var_plot_path = "figs/lambda_per_a/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/lambda_per_a_N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(N_val .* num_unit_cells_vals))_AW$(avraging_window_name)_Stds.png"
make_path_exist(var_plot_path)
savefig(var_plot_path)
println("Saved Plot: $(var_plot_path)")

# --- Colapsing std plot ---
avraging_window = 1/32
avraging_window_name = replace("$(round(avraging_window, digits=3))", "." => "p")


a_crit = 0.76285796	
nu = 1.72541695
# Create plot
plt = plot(
    title="Colapss Std: N=$N_val,a_c = $(round(a_crit, digits=3)),nu = $(round(nu, digits=3))",
    xlabel="a",
    ylabel="Scaled Std(λ) | AW=$avraging_window_name"
)

# Plot data for each L
for L in num_unit_cells_vals * N_val
    L = Int(L)
    plot!(plt, (a_vals .- a_crit) .* L^(1/nu), [val for val in values(sort(collected_lambdas_SEMs[avraging_window][L]))] * sqrt(num_initial_conds-1),
        label="L=$L",
        linestyle=:dash,
        markersize=2,
        linewidth=1,
        marker = :circle)
end

collapsed_var_plot_path = "figs/lambda_per_a/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/lambda_per_a_N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(N_val .* num_unit_cells_vals))_AW$(avraging_window_name)_Stds_Collapsed.png"
make_path_exist(collapsed_var_plot_path)
savefig(collapsed_var_plot_path)
println("Saved Plot: $(collapsed_var_plot_path)")

# --- Zoomed Colapsing std plot ---
# Create plot
plt = plot(
    title="Colapss Std: N=$N_val,a_c = $(round(a_crit, digits=3)),nu = $(round(nu, digits=3))",
    xlabel="a",
    ylabel="Scaled Std(λ) | AW=$avraging_window_name"
)

# Plot data for each L
for L in num_unit_cells_vals * N_val
    L = Int(L)
    plot!(plt, (trans_a_vals .- a_crit) .* L^(1/nu), [collected_lambdas_SEMs[avraging_window][L][a_val] for a_val in sort(trans_a_vals)] * sqrt(num_initial_conds-1),
        label="L=$L",
        linestyle=:dash,
        markersize=2,
        linewidth=1,
        marker = :circle)
end

zoomed_collapsed_var_plot_path = "figs/lambda_per_a/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/Zoomed_lambda_per_a_N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(N_val .* num_unit_cells_vals))_AW$(avraging_window_name)_Stds_Collapsed.png"
make_path_exist(zoomed_collapsed_var_plot_path)
savefig(zoomed_collapsed_var_plot_path)
println("Saved Plot: $(zoomed_collapsed_var_plot_path)")

# --- Plotting Length and Height of Jump as Function of L
jump_start_a_vals = Dict{Int, Float64}(32 => 0.72, 64 => 0.74, 128 => 0.75, 256 => 0.7525)
jump_end_a_vals = Dict{Int, Float64}(32 => 0.8, 64 => 0.79, 128 => 0.785, 256 => 0.78)
jump_start_lambda_vals = Dict{Int, Float64}(32 => -0.296, 64 => -0.264, 128 =>  -0.244, 256 =>  -0.243)
jump_end_lambda_vals = Dict{Int, Float64}(32 => 0.390, 64 => 0.370, 128 =>  0.380, 256 =>  0.380)

base_plot_path = "figs/lambda_analysis/N$N_val/IC$num_initial_conds/SeveralLs/lambda_gap_analysis_N$(N_val)_IC$(num_initial_conds)_L$(join(sort(num_unit_cells_vals)*N_val))"
make_path_exist(base_plot_path)

# Ploting width
plt = plot(
    title="Jump Width",
    xlabel="L",
    ylabel="a diffrence"
)
plot!(plt, log.([L for L in keys(sort(jump_start_a_vals))]), log.([jump_end_a_vals[L] - jump_start_a_vals[L] for L in keys(sort(jump_start_a_vals))]))
savefig("$(base_plot_path)_jump_width.png")
println("Saved ", "$(base_plot_path)_jump_width.png")

# Plotting width gap
plt = plot(
    title="Jump a_val limits",
    xlabel="L",
    ylabel="a"
)
plot!(plt, [L for L in keys(sort(jump_start_a_vals))], [jump_start_a_vals[L] for L in keys(sort(jump_start_a_vals))], label="a_min")
plot!(plt, [L for L in keys(sort(jump_start_a_vals))], [jump_end_a_vals[L] for L in keys(sort(jump_start_a_vals))], label="a_max")
savefig("$(base_plot_path)_jump_width_gap.png")
println("Saved ", "$(base_plot_path)_jump_width_gap.png")

# Plotting height
plt = plot(
    title="Jump Height",
    xlabel="L",
    ylabel="lambda diffrence"
)
plot!(plt, [L for L in keys(sort(jump_end_lambda_vals))], [jump_end_lambda_vals[L] - jump_start_lambda_vals[L] for L in keys(sort(jump_start_a_vals))])
savefig("$(base_plot_path)_jump_height.png")
println("Saved ", "$(base_plot_path)_jump_height.png")

# Plotting height gap
plt = plot(
    title="Jump lambda limits",
    xlabel="L",
    ylabel="lambda"
)
plot!(plt, [L for L in keys(sort(jump_start_lambda_vals))], [jump_start_lambda_vals[L] for L in keys(sort(jump_start_a_vals))], label="lambda_min")
plot!(plt, [L for L in keys(sort(jump_start_lambda_vals))], [jump_end_lambda_vals[L] for L in keys(sort(jump_start_a_vals))], label="lambda_max")
savefig("$(base_plot_path)_jump_height_gap.png")
println("Saved ", "$(base_plot_path)_jump_height_gap.png")


# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# --- S_diff analysis ---

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
a_vals_to_plot = [0.6, 0.65, 0.68, 0.7, 0.71, 0.73, 0.76, 0.77]
num_unit_cell_to_plot = num_unit_cells_vals[end]
L_val_to_plot = Int(round(num_unit_cell_to_plot * N_val))

# Create plot
plt = plot(
    title="SDiff for N=$N_val | L = $(L_val_to_plot)",
    xlabel="t",
    ylabel="S_Diif"
)

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
display(plt)

# --- Making Log(S_Diff) Plots ---
L_val_to_plot = Int(round(num_unit_cell_to_plot * N_val))

a_vals_to_plot = [0.67, 0.68]

t_limit = Int(round(min(L_val_to_plot^1.6, L_val_to_plot^1.6)))

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
display(plt)

# Zoomed in plot
num_unit_cell_to_plot = 64
L_val_to_plot = Int(round(num_unit_cell_to_plot * N_val))

# y_lims = (-15, 0)
a_vals_to_plot = [0.75, 0.7525]
t_limits = [1, Int(round(L_val_to_plot^1.6))]
x_tick = 500
# Create plot
plt = plot(
    title="Log(SDiff )for N=$N_val | L = $(L_val_to_plot)",
    xlabel="t",
    ylabel="Log(S_Diif)",
    # ylims=y_lims,
    xticks=0:x_tick:Int(round(L_val_to_plot^1.6))
)

# Plot data for each a_val
for a_val in a_vals_to_plot

    plot!(plt, log.(collected_S_diffs[L_val_to_plot][a_val][t_limits[1]:t_limits[2]]),
        label="a = $(a_val)",
        linestyle=:solid,
        linewidth=1,)
end

log_s_diff_plot_path = "figs/log_delta_evolved_spins/N$(N_val)/SeveralAs/IC$num_initial_conds/L$(L_val_to_plot)/Zoomed_Log_S_diff$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L_val_to_plot).png"
# make_path_exist(log_s_diff_plot_path)
# savefig(log_s_diff_plot_path)
# println("Saved Plot: $(log_s_diff_plot_path)")
display(plt)

# --- Fitting Log of S_diff ---
# From the analysis of the plots, it seems that the first linear domain is as long as Log(S_diff) > some value, so we will 
# Fit accordingly

times_to_fit = Dict{Int, Dict{Float64, Vector{Int}}}()
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
                                               0.765 => [200, Int(round(256^1.6))], 
                                               0.7675 => [200, Int(round(256^1.6))], 
                                               0.77 => [200, Int(round(256^1.6))])
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
                                               0.765 => [200, Int(round(128^1.6))], 
                                               0.7675 => [200, Int(round(128^1.6))], 
                                               0.77 => [200, Int(round(128^1.6))])
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
                                               0.765 => [1, Int(round(64^1.6))], 
                                               0.7675 => [1, Int(round(64^1.6))], 
                                               0.77 => [1, Int(round(64^1.6))])

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
                                               0.765 => [1, Int(round(32^1.6))], 
                                               0.7675 => [1, Int(round(32^1.6))], 
                                               0.77 => [1, Int(round(32^1.6))])


# Getting data to fit
# Plot data for each a_val
L_val_to_plot = 256
a_vals_to_plot = sort([a_val for a_val in keys(times_to_fit[L_val_to_plot])])
log_s_diff_to_fit = Dict{Float64, Vector{Float64}}()
for a_val in a_vals_to_plot
    println("a_val: $(a_val)")
    log_s_diff_to_fit[a_val] = [val for val in log.(collected_S_diffs[L_val_to_plot][a_val][times_to_fit[L_val_to_plot][a_val][1]:times_to_fit[L_val_to_plot][a_val][2]])]
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

    plot!(plt, xs, log_s_diff_to_fit[a_val],
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

decay_a_vals = [val for val in a_vals if 0.67 <= val <= 0.76]

# Calculating the fits
all_log_s_diff_slopes = Dict{Int, Dict{Float64, Float64}}()

for L_val in N_val .* num_unit_cells_vals
    all_log_s_diff_slopes[L_val] = Dict{Float64, Float64}()
    for a_val in decay_a_vals
        println("L: $L_val | a_val: $a_val")
        log_s_diff_to_fit = [val for val in log.(collected_S_diffs[L_val][a_val][times_to_fit[L_val][a_val][1]:times_to_fit[L_val][a_val][2]])]

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
    plot!(plt, sort([a_val for a_val in decay_a_vals]), [-1/val for val in values(sort(all_log_s_diff_slopes[L]))],
        label="L=$L",
        linestyle=:solid,
        markersize=5,
        linewidth=1,
        marker = :circle)
end

decay_per_a_plot_path = "figs/decay_per_a/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/decay_per_a_N$(N_val)_ar$(replace("$(minimum(decay_a_vals))_$(maximum(decay_a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(N_val .* num_unit_cells_vals)).png"
make_path_exist(decay_per_a_plot_path)
savefig(decay_per_a_plot_path)
println("Saved Plot: $(decay_per_a_plot_path)")

# --- Scaled Decay Timescale as Func of a for all L --- 

decay_a_vals = [val for val in a_vals if 0.67 <= val <= 0.7625]

# Calculating the fits
all_log_s_diff_slopes = Dict{Int, Dict{Float64, Float64}}()

for L_val in N_val .* num_unit_cells_vals
    all_log_s_diff_slopes[L_val] = Dict{Float64, Float64}()
    for a_val in decay_a_vals
        println("L: $L_val | a_val: $a_val")
        log_s_diff_to_fit = [val for val in log.(collected_S_diffs[L_val][a_val][times_to_fit[L_val][a_val][1]:times_to_fit[L_val][a_val][2]])]

        xs = 1:length(log_s_diff_to_fit)

        df = DataFrame(x=xs, y=log_s_diff_to_fit)
        model = lm(@formula(y ~ x), df)

        all_log_s_diff_slopes[L_val][a_val] = coef(model)[2]  # gives [intercept, slope]
    end
end

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

a_crit = 0.76285796
nu = 1.72541695
z = 1.6

# Create plot
plt = plot(
    title="Scaled ξ_τ (z=$(round(z, digits=3)),nu = $(round(nu, digits = 3)),a_c = $(round(a_crit, digits = 3)))",
    xlabel="(a - a_c)L^(1/ν)",
    ylabel="ξ_τ / L^{z}"
)

# Plot data for each L
for L in num_unit_cells_vals * N_val
    L = Int(L)
    plot!(plt, sort([a_val-a_crit for a_val in decay_a_vals]) .* L^(1/nu), [-1/val for val in values(sort(all_log_s_diff_slopes[L]))] ./ (L^z),
        label="L=$L",
        linestyle=:solid,
        markersize=2,
        linewidth=1,
        marker = :circle)
end

scaled_decay_per_a_plot_path = "figs/decay_per_a/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/scaled_decay_per_a_N$(N_val)_ar$(replace("$(minimum(decay_a_vals))_$(maximum(decay_a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(N_val .* num_unit_cells_vals)).png"
make_path_exist(scaled_decay_per_a_plot_path)
savefig(scaled_decay_per_a_plot_path)
println("Saved Plot: $(scaled_decay_per_a_plot_path)")

# Plotting log of the collapse

a_crit = 0.76285796
nu = 1.72541695
z = 1.7

# Create plot
plt = plot(
    title="Scaled ξ_τ (z=$(round(z, digits=3)),nu = $(round(nu, digits = 3)),a_c = $(round(a_crit, digits = 3)))",
    xlabel="(a - a_c)L^(1/ν)",
    ylabel="ξ_τ / L^{z}",
    xlims=(-1, 0)
)

# Plot data for each L
for L in num_unit_cells_vals * N_val
    L = Int(L)
    plot!(plt, sort([a_val-a_crit for a_val in decay_a_vals]) .* L^(1/nu), log.([-1/val for val in values(sort(all_log_s_diff_slopes[L]))] ./ (L^z)),
        label="L=$L",
        linestyle=:solid,
        markersize=2,
        linewidth=1,
        marker = :circle)
end

log_scaled_decay_per_a_plot_path = "figs/decay_per_a/N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/log_scaled_decay_per_a_N$(N_val)_ar$(replace("$(minimum(decay_a_vals))_$(maximum(decay_a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(N_val .* num_unit_cells_vals)).png"
make_path_exist(log_scaled_decay_per_a_plot_path)
savefig(log_scaled_decay_per_a_plot_path)
println("Saved Plot: $(log_scaled_decay_per_a_plot_path)")
