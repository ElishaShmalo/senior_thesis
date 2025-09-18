
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
# a_vals = [0.75, 0.7525, 0.755, 0.7575, 0.76, 0.7625, 0.765, 0.7675, 0.77] # trans a_vals

epsilon = 0.1

N_val = 4

z_val = 1.7
z_val_name = replace("$z_val", "." => "p")

# --- Lyop Analysis ---
avraging_windows = [1/32]
# avraging_windows = [1/32]

collected_lambdas = Dict{Float64, Dict{Int, Dict{Float64, Float64}}}()
collected_lambdas_SEMs = Dict{Float64, Dict{Int, Dict{Float64, Float64}}}()
for avraging_window in avraging_windows
    println("Aw: $(avraging_window)")
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
        n = Int(round(L^z_val))

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

                sample_filepath = "data/spin_dists_per_time/N$N_val/a$a_val_name/IC1/L$L/N$(N_val)_a$(a_val_name)_IC1_L$(L)_z$(z_val_name)_sample$(init_cond).csv"
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

x_vals = range(minimum(a_vals) - 0.005, stop = maximum(a_vals) + 0.00005, length = 1000)

plot!(plt, x_vals, log.(x_vals), linestyle = :dash, label = "ln(a)", title="λ(a) for N=$N_val")

xlabel!("a")
ylabel!("λ")
display(plt)

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
display(plt)

# --- Make seprate plot for each L with window curves ---
avraging_windows_names = replace("$(join(avraging_windows))", "." => "p")
for L in num_unit_cells_vals * N_val
    # Plot Std(lambda)
    plt = plot(
    title="Std(λ(a)) for N=$N_val | L = $L",
    xlabel="a",
    ylabel="Std(λ)",
    xticks = minimum(a_vals):0.02:maximum(a_vals)
    )

    L = Int(L)
    for avraging_window in avraging_windows
        plot!(plt, a_vals, [val for val in values(sort(collected_lambdas_SEMs[avraging_window][L]))] * sqrt(num_initial_conds-1),
            label="Aw = $avraging_window",
            linestyle=:solid,
            markersize=2,
            linewidth=1,
            marker = :circle)
    end

    plot_path = "figs/lambda_per_a/N$(N_val)/SeveralAs/IC$num_initial_conds/L$L/std_lambda_per_a_N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L)_AW$(avraging_windows_names).png"
    make_path_exist(plot_path)
    savefig(plot_path)
    println("Saved Plot: $(plot_path)")
    display(plt)
    
    plt = plot(
    title="λ(a) for N=$N_val | L = $L",
    xlabel="a",
    ylabel="λ",
    xticks = minimum(a_vals):0.02:maximum(a_vals)
    )

    L = Int(L)
    for avraging_window in avraging_windows
        plot!(plt, a_vals, [val for val in values(sort(collected_lambdas[avraging_window][L]))],
            label="Aw = $avraging_window",
            linestyle=:solid,
            markersize=2,
            linewidth=1,
            marker = :circle)
    end

    plot_path = "figs/lambda_per_a/N$(N_val)/SeveralAs/IC$num_initial_conds/L$L/lambda_per_a_N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(L)_AW$(avraging_windows_names).png"
    make_path_exist(plot_path)
    savefig(plot_path)
    println("Saved Plot: $(plot_path)")
    display(plt)
end

# --- Colapsing std plot ---
avraging_window = 1/32
avraging_window_name = replace("$(round(avraging_window, digits=3))", "." => "p")


a_crit = 0.76154106	# pm 2.3231e-04
nu = 1.72930133 # 0.06886529
# a_crit = 0.76185071
# nu = 1.72930133

# Create plot
plt = plot(
    title="Colapss Std: N=$N_val,a_c = $(round(a_crit, digits = 4)),nu = $(round(nu, digits=3))",
    xlabel="(a - a_c) * L^{1/ν}",
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
display(plt)

# --- Zoomed Colapsing std plot ---

a_vals_to_plot = [a_val for a_val in a_vals if a_crit - 0.05 < a_val < a_crit + 0.05]

# Create plot
plt = plot(
    title="Colapss Std: N=$N_val,a_c = $(round(a_crit, digits = 5)),nu = $(round(nu, digits=3))",
    xlabel="(a - a_c) * L^{1/ν}",
    ylabel="Scaled Std(λ) | AW=$avraging_window_name"
)

# Plot data for each L
for L in num_unit_cells_vals * N_val
    L = Int(L)
    plot!(plt, (a_vals_to_plot .- a_crit) .* L^(1/nu), [collected_lambdas_SEMs[avraging_window][L][a_val] for a_val in sort(a_vals_to_plot)] * sqrt(num_initial_conds-1),
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
display(plt)

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
