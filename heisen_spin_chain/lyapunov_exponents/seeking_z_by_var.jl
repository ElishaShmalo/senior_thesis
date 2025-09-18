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


num_initial_conds = 1000 # We are avraging over x initial conditions
# trans_a_vals = [0.72, 0.73, 0.74, 0.75, 0.7525, 0.755, 0.7575, 0.76, 0.7625, 0.765, 0.7675, 0.77, 0.78, 0.79, 0.8]
# a_vals = sort(union([round(0.7 + i*0.02, digits=2) for i in 0:4], [0.7525, 0.755, 0.7575, 0.7625, 0.765, 0.7675], [0.763])) # general a_vals
a_vals = [0.7625]
# a_vals = sort([0.7625, 0.76, 0.763, 0.765]) # trans a_vals
println("a vals: $(a_vals)")
epsilon = 0.1

N_val = 4

avraging_window = 1
skip_fract = 1 - avraging_window
avraging_window_name = replace("$(round(avraging_window, digits=3))", "." => "p")

z_val = 2.0
z_val_name = replace("$(z_val)", "." => "p")

# --- Load in and take avrages from all samples ---
# Our dict for recording results
collected_lambda_series = Dict{Int, Dict{Float64, Vector{Float64}}}() # Int: L_val, Float64: a_val, Float64: avrg lambda
collected_lambda_STD_series = Dict{Int, Dict{Float64, Vector{Float64}}}() # Int: L_val, Float64: a_val, Float64: standard error on the mean for lambda

for num_unit_cells in num_unit_cells_vals
    L = num_unit_cells * N_val
    println("L_val: $L")

    # number of pushes we are going to do
    n = Int(round(L^z_val))

    num_skip = Int(round(skip_fract * n)) + 1 

    # Define s_naught to be used during control step
    S_NAUGHT = make_spiral_state(L, (2) / N_val)

    # Initializes results for this N_val
    if ! haskey(collected_lambda_series, L)
        collected_lambda_series[L] = Dict(a => zeros(Float64, Int(round(L^z_val) - num_skip)) for a in a_vals)
        collected_lambda_STD_series[L] = Dict(a => zeros(Float64, Int(round(L^z_val) - num_skip)) for a in a_vals)
    end

    for a_val in a_vals
        println("L_val: $L | a_val: $a_val")
        a_val_name = replace("$a_val", "." => "p")
        # We will avrage over this later
        current_lambdas = [zeros(Float64, Int(round(L^z_val))) for _ in 1:num_initial_conds]

        for init_cond in 1:num_initial_conds
            current_spin_dists = zeros(n)

            sample_filepath = "data/spin_dists_per_time/N$N_val/a$a_val_name/IC1/L$L/N$(N_val)_a$(a_val_name)_IC1_L$(L)_z$(z_val_name)_sample$(init_cond).csv"
            df = CSV.read(sample_filepath, DataFrame)

            sample_lambdas = df[!, "lambda"]
            current_lambdas[init_cond] = sample_lambdas[num_skip:end]
        end

        collected_lambda_series[L][a_val] = mean(current_lambdas)
        collected_lambda_STD_series[L][a_val] = std(current_lambdas)
    end
end

# --- Save the plot ---
println("Making Plot")

a_vals_to_plot = [0.7625]

for L in num_unit_cells_vals * N_val
    plt = plot(
        title="λ(t) for N=$N_val | L = $L | z=$(z_val_name) | AW=$avraging_window_name",
        xlabel="t",
        ylabel="λ"
    )

    L = Int(L)
    for a_val in a_vals_to_plot
        plot!(
            collected_lambda_series[L][a_val], 
            # yerror=collected_lambda_STD_series[L][a_val],
            label="a = $a_val")
    end
    
    plot_path = "N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/lambda_per_a_N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(N_val .* num_unit_cells_vals))_z$(z_val_name)_AW$avraging_window_name"
    display(plt)
end

# Varience as func of time
println("Making Plot")
a_vals_to_plot = [0.7625]

for L in num_unit_cells_vals * N_val
    plt = plot(
        title="Var(λ(t)) for N=$N_val | z=$(z_val_name) |  L = $L | AW=$avraging_window_name",
        xlabel="t",
        ylabel="Var(λ)"
    )

    L = Int(L)
    for a_val in a_vals_to_plot
        plot!(
            collected_lambda_STD_series[L][a_val].^2, 
            # yerror=collected_lambda_STD_series[L][a_val],
            label="a = $a_val")
    end
    
    plot_path = "figs/lambda_per_t/N$(N_val)/SeveralAs/IC$num_initial_conds/L$L/std_lambda_per_t_N$(N_val)_ar$(replace("$(minimum(a_vals_to_plot))_$(maximum(a_vals_to_plot))", "." => "p"))_IC$(num_initial_conds)_L$(L)_z$(z_val_name)"
    make_path_exist(plot_path)
    savefig(plot_path)
    println("Saved Plot: $(plot_path).png")
end


# --- Log scale t ---
a_vals_to_plot = [0.7625]

plt = plot(
    title="Var(λ(t)) for N=$N_val | t_f=L^$(z_val)",
    xlabel="log(t)",
    ylabel="Var(λ)"
)
for L in num_unit_cells_vals * N_val
    L = Int(L)
    for a_val in a_vals_to_plot
        plot!(log.(1:length(collected_lambda_STD_series[L][a_val])),
            collected_lambda_STD_series[L][a_val].^2, 
            # yerror=collected_lambda_STD_series[L][a_val],
            label="L = $L")
    end
end

log_t_plot_path = "figs/lambda_per_t/N$(N_val)/SeveralAs/IC$num_initial_conds/LSeveral/std_lambda_per_log_t_N$(N_val)_ar$(replace("$(minimum(a_vals_to_plot))_$(maximum(a_vals_to_plot))", "." => "p"))_IC$(num_initial_conds)_z$(z_val_name)"
make_path_exist(log_t_plot_path)
savefig(log_t_plot_path)
println("Saved Plot: $(log_t_plot_path).png")
display(plt)

# --- Log scale t, find peak limits ---
a_vals_to_plot = [0.7625]
peak_frac_lims = [1/30, 9/10]

plt = plot(
    title="Var(λ(t)) for N=$N_val | t_f=L^$(z_val)",
    xlabel="log(t)",
    ylabel="Var(λ)"
)
for L in num_unit_cells_vals * N_val
    L = Int(L)
    for a_val in a_vals_to_plot
        num_time_steps = length(collected_lambda_STD_series[L][a_val])
        plot!(log.(trunc(Int, num_time_steps*peak_frac_lims[1]):trunc(Int, num_time_steps*peak_frac_lims[2])),
            (collected_lambda_STD_series[L][a_val].^2)[trunc(Int, num_time_steps*peak_frac_lims[1]):trunc(Int, num_time_steps*peak_frac_lims[2])], 
            # yerror=collected_lambda_STD_series[L][a_val],
            label="L = $L")
    end
end

display(plt)

# --- Getting z properly ---

# first lets get the log(time) in which there is a peak
log_peak_times = Dict{Int, Float64}(32 => 5.854, 64 => 6.919, 128 => 8.177, 256 => 9.231)

a_vals_to_plot = [0.7625]
peak_frac_lims = [1/30, 9/10]

for L in num_unit_cells_vals * N_val
    plt = plot(
        title="Var(λ(t)) for N=$N_val | t_f=L^$(z_val)",
        xlabel="log(t)",
        ylabel="Var(λ)"
    )
    L = Int(L)
    for a_val in a_vals_to_plot
        num_time_steps = length(collected_lambda_STD_series[L][a_val])
        plot!(log.(trunc(Int, num_time_steps*peak_frac_lims[1]):trunc(Int, num_time_steps*peak_frac_lims[2])),
            (collected_lambda_STD_series[L][a_val].^2)[trunc(Int, num_time_steps*peak_frac_lims[1]):trunc(Int, num_time_steps*peak_frac_lims[2])], 
            # yerror=collected_lambda_STD_series[L][a_val],
            label="L = $L")
    end
    vline!(plt, [log_peak_times[L]], label="x = $(log_peak_times[L])")
    display(plt)
end

# Now we plot the values along with a linear fit
plt = plot(
    title="log(t_peak) as a function of L",
    xlabel="L",
    ylabel="log(t)"
)

plot!([k for k in keys(sort(log_peak_times))],
    [log_peak_times[k] for k in keys(sort(log_peak_times))], 
    # yerror=collected_lambda_STD_series[L][a_val],
    marker = :circle)

display(plt)

# --- Collapsed Log scale t ---
a_vals_to_plot = [0.7625]

z_collapse_val = 1.65

plt = plot(
    title="Collapsed Var(λ(t)) for N=$N_val | t_f=L^$(z_val) | z=$(z_collapse_val)",
    xlabel="log(t / L^z)",
    ylabel="Var(λ)"
)
for L in num_unit_cells_vals * N_val
    L = Int(L)
    for a_val in a_vals_to_plot
        num_time_steps = length(collected_lambda_STD_series[L][a_val])
        plot!(log.((trunc(Int, num_time_steps*peak_frac_lims[1]):trunc(Int, num_time_steps*peak_frac_lims[2])) ./ (L^z_collapse_val)),
            (collected_lambda_STD_series[L][a_val].^2)[trunc(Int, num_time_steps*peak_frac_lims[1]):trunc(Int, num_time_steps*peak_frac_lims[2])], 
            # yerror=collected_lambda_STD_series[L][a_val],
            label="L = $L")
    end
end

zoomed_col_log_t_plot_path = "figs/lambda_per_t/N$(N_val)/SeveralAs/IC$num_initial_conds/LSeveral/zoomed_std_lambda_per_log_t_N$(N_val)_ar$(replace("$(minimum(a_vals_to_plot))_$(maximum(a_vals_to_plot))", "." => "p"))_IC$(num_initial_conds)_z$(z_val_name)_col"
make_path_exist(zoomed_col_log_t_plot_path)
savefig(zoomed_col_log_t_plot_path)
println("Saved Plot: $(zoomed_col_log_t_plot_path).png")
display(plt)

plt = plot(
    title="Collapsed Var(λ(t)) for N=$N_val | t_f=L^$(z_val) | z=$(z_collapse_val)",
    xlabel="log(t / L^z)",
    ylabel="Var(λ)"
)
for L in num_unit_cells_vals * N_val
    L = Int(L)
    for a_val in a_vals_to_plot
        num_time_steps = length(collected_lambda_STD_series[L][a_val])
        plot!((1:num_time_steps) ./ (L^z_collapse_val),
            (collected_lambda_STD_series[L][a_val].^2), 
            # yerror=collected_lambda_STD_series[L][a_val],
            label="L = $L")
    end
end

col_log_t_plot_path = "figs/lambda_per_t/N$(N_val)/SeveralAs/IC$num_initial_conds/LSeveral/std_lambda_per_log_t_N$(N_val)_ar$(replace("$(minimum(a_vals_to_plot))_$(maximum(a_vals_to_plot))", "." => "p"))_IC$(num_initial_conds)_z$(z_val_name)_col"
make_path_exist(col_log_t_plot_path)
savefig(col_log_t_plot_path)
println("Saved Plot: $(col_log_t_plot_path).png")
display(plt)
