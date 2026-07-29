# Imports
using Random, LinearAlgebra, Plots, DifferentialEquations, Serialization, Statistics, DelimitedFiles, SharedArrays, CSV, DataFrames, GLM, LaTeXStrings

# Other files   
include("../utils/make_spins.jl")
include("../utils/general.jl")
include("../utils/dynamics.jl")
include("../utils/lyapunov.jl")
include("../analytics/spin_diffrences.jl")


default(
    guidefont = 22,     # alternative, some backends use 'guidefont'
    tickfont = 15      # font size for axis tick marks
)

J = 1    # energy factor

# Time to evolve until push back to S_A
tau = 1 * J

# General Variables
# num_unit_cells_vals = [8, 16, 32, 64, 128]
num_unit_cells_vals = [8, 16, 32, 64, 128]
num_uc = length(num_unit_cells_vals)
blue_palette = cgrad([RGB(0.55, 0.75, 0.85), RGB(0.2, 0.35, 0.9)], num_uc, categorical=true) # from light to dark blue

# num_unit_cells_vals = [8]


num_initial_conds = 1000 # We are avraging over x initial conditions
# trans_a_vals = [0.72, 0.73, 0.74, 0.75, 0.7525, 0.755, 0.7575, 0.76, 0.7616, 0.765, 0.7675, 0.77, 0.78, 0.79, 0.8]
# a_vals = sort(union([round(0.7 + i*0.02, digits=2) for i in 0:4], [0.7525, 0.755, 0.7575, 0.7616, 0.765, 0.7675], [0.7616])) # general a_vals
a_vals = [0.7616]
# a_vals = sort([0.7616, 0.76, 0.7616, 0.765]) # trans a_vals
println("a vals: $(a_vals)")
epsilon = 0.1

N_val = 4

avraging_window = 1
skip_fract = 1 - avraging_window
avraging_window_name = replace("$(round(avraging_window, digits=3))", "." => "p")

z_val = 1.7
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
        current_lambdas = [zeros(Float64, Int(round(L^z_val)))[num_skip:end] for _ in 1:num_initial_conds]

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

a_vals_to_plot = [0.7616]

for (i, L) in enumerate(num_unit_cells_vals * N_val)
    c = blue_palette[i]
    plt = plot(
        title=L"$λ(t)$ for $N=%$N_val$ | $L = %$L$ | $z=%$(z_val)$",
        xlabel=L"t",
        ylabel=L"λ"
    )

    L = Int(L)
    for a_val in a_vals_to_plot
        plot!(
            collected_lambda_series[L][a_val], 
            # yerror=collected_lambda_STD_series[L][a_val],
            label="a = $a_val",
            color = c)
    end
    
    plot_path = "N$(N_val)/SeveralAs/IC$num_initial_conds/SeveralLs/lambda_per_a_N$(N_val)_ar$(replace("$(minimum(a_vals))_$(maximum(a_vals))", "." => "p"))_IC$(num_initial_conds)_L$(join(N_val .* num_unit_cells_vals))_z$(z_val_name)_AW$avraging_window_name"
    display(plt)
end

# Std as func of time
println("Making Plot")
a_vals_to_plot = [0.7616]

for (i, L) in enumerate(num_unit_cells_vals * N_val)
    c = blue_palette[i]
    plt = plot(
        title=L"$\mathrm{Std}(λ(t))$ for $N=%$N_val$ | $z=%$(z_val_name)$ |  $L = %$(L)$",
        xlabel=L"t",
        ylabel=L"\mathrm{Std}(λ)"
    )

    L = Int(L)
    for a_val in a_vals_to_plot
        plot!(
            collected_lambda_STD_series[L][a_val], 
            # yerror=collected_lambda_STD_series[L][a_val],
            label="a = $a_val",
            color=c)
    end
    
    plot_path = "figs/lambda_per_t/N$(N_val)/SeveralAs/IC$num_initial_conds/L$L/std_lambda_per_t_N$(N_val)_ar$(replace("$(minimum(a_vals_to_plot))_$(maximum(a_vals_to_plot))", "." => "p"))_IC$(num_initial_conds)_L$(L)_z$(z_val_name)"
    make_path_exist(plot_path)
    savefig(plot_path)
    println("Saved Plot: $(plot_path).png")
    display(plt)
end


# --- Log scale t ---
a_vals_to_plot = [0.7616]

log_plt = plot(
    title=L"$\mathrm{Std}(λ(t))$ for $N=%$N_val$ | $t_f=L^{%$(z_val)}$",
    xlabel=L"\mathrm{log}(t)",
    ylabel=L"\mathrm{Std}(λ)"
)
plot!(log_plt, [NaN], [NaN], label = "L =", linecolor = RGBA(0,0,0,0))
for (i, L) in enumerate(num_unit_cells_vals * N_val)
    c = blue_palette[i]
    L = Int(L)
    for a_val in a_vals_to_plot
        plot!(log_plt, log.(1:length(collected_lambda_STD_series[L][a_val])),
            collected_lambda_STD_series[L][a_val], 
            # yerror=collected_lambda_STD_series[L][a_val],
            label="$L",
            color=c)
    end
end

log_t_plot_path = "figs/lambda_per_t/N$(N_val)/SeveralAs/IC$num_initial_conds/LSeveral/std_lambda_per_log_t_N$(N_val)_ar$(replace("$(minimum(a_vals_to_plot))_$(maximum(a_vals_to_plot))", "." => "p"))_IC$(num_initial_conds)_z$(z_val_name)"
make_path_exist(log_t_plot_path)
savefig(log_plt, log_t_plot_path)
println("Saved Plot: $(log_t_plot_path).png")
display(log_plt)

# --- t, find peak limits ---
a_vals_to_plot = [0.7616]
peak_frac_lims = [1/30, 9/10]

for a_val in a_vals_to_plot
    plt = plot(
        title=L"$\mathrm{Std}(λ(t))$ for $N=%$N_val$ | $t_f=L^{%$(z_val)}$",
        xlabel=L"\mathrm{log}(t)",
        ylabel=L"\mathrm{Std}(λ)"
    )
    for (i, L) in enumerate(num_unit_cells_vals * N_val)
        L = Int(L)
        c = blue_palette[i]
            num_time_steps = length(collected_lambda_STD_series[L][a_val])
            plot!(log.(trunc(Int, num_time_steps*peak_frac_lims[1]):trunc(Int, num_time_steps*peak_frac_lims[2])),
                (collected_lambda_STD_series[L][a_val])[trunc(Int, num_time_steps*peak_frac_lims[1]):trunc(Int, num_time_steps*peak_frac_lims[2])], 
                # yerror=collected_lambda_STD_series[L][a_val],
                label="L = $L",
                color=c)
        end
    display(plt)
end


# -- Save data to collapse
a_val_to_save = [0.7616]
for (i, L) in enumerate(num_unit_cells_vals * N_val)
    L = Int(L)
    for a_val in a_val_to_save
        data = Dict{Float64, Float64}()
        num_time_steps = length(collected_lambda_STD_series[L][a_val])
        for t_step in trunc(Int, num_time_steps*peak_frac_lims[1]):trunc(Int, num_time_steps*peak_frac_lims[2])
            data[t_step] = (collected_lambda_STD_series[L][a_val])[t_step]
        end
        save_simple_dict_to_csv(data, "data_to_collapse/lambda_std_per_t/N$(N_val)/SeveralAs/IC$num_initial_conds/L$(L)/lambda_std_per_t_N$(N_val)_a$(replace("$(a_val)", "." => "p"))_IC$(num_initial_conds)_L$(L)_z$(z_val_name).csv")
    end
end

# --- Collapsed scale t ---
a_vals_to_plot = [0.7616]

z_collapse_val = round(1.6, digits=4) # 6.459684070797505e-08

plt = plot(
    title=L"Collapsed $\mathrm{Std}(λ(t))$ for $N=%$N_val$ " * " \n " * L"$t_f=L^{%$(z_val)}$ | $z=%$(z_collapse_val)$",
    xlabel=L"\frac{t}{L^z}",
    ylabel=L"\mathrm{Std}(λ)"
)
for (i, L) in enumerate(num_unit_cells_vals * N_val)
    L = Int(L)
    c = blue_palette[i]
    for a_val in a_vals_to_plot
        num_time_steps = length(collected_lambda_STD_series[L][a_val])
        plot!((trunc(Int, num_time_steps*peak_frac_lims[1]):trunc(Int, num_time_steps*peak_frac_lims[2])) ./ (L^z_collapse_val),
            (collected_lambda_STD_series[L][a_val])[trunc(Int, num_time_steps*peak_frac_lims[1]):trunc(Int, num_time_steps*peak_frac_lims[2])], 
            # yerror=collected_lambda_STD_series[L][a_val],
            label="L = $L",
            color=c)
    end
end

zoomed_col_t_plot_path = "figs/lambda_per_t/N$(N_val)/SeveralAs/IC$num_initial_conds/LSeveral/zoomed_std_lambda_per_t_N$(N_val)_ar$(replace("$(minimum(a_vals_to_plot))_$(maximum(a_vals_to_plot))", "." => "p"))_IC$(num_initial_conds)_z$(z_val_name)_col"
make_path_exist(zoomed_col_t_plot_path)
savefig(zoomed_col_t_plot_path)
println("Saved Plot: $(zoomed_col_t_plot_path).png")
display(plt)

plt = plot(
    title=L"Collapsed $\mathrm{Std}(λ(t))$ for $N=%$N_val$" * "\n" * L"$t_f=L^{%$(z_val)}$ | $z=%$(z_collapse_val)$",
    xlabel=L"\frac{t}{L^z}",
    ylabel=L"\mathrm{Std}(λ)"
)
for (i, L) in enumerate(num_unit_cells_vals * N_val)
    c = blue_palette[i]
    L = Int(L)
    for a_val in a_vals_to_plot
        num_time_steps = length(collected_lambda_STD_series[L][a_val])
        plot!((1:num_time_steps) ./ (L^z_collapse_val),
            (collected_lambda_STD_series[L][a_val]), 
            # yerror=collected_lambda_STD_series[L][a_val],
            label="L = $L",
            color=c)
    end
end

col_t_plot_path = "figs/lambda_per_t/N$(N_val)/SeveralAs/IC$num_initial_conds/LSeveral/std_lambda_per_t_N$(N_val)_ar$(replace("$(minimum(a_vals_to_plot))_$(maximum(a_vals_to_plot))", "." => "p"))_IC$(num_initial_conds)_z$(z_val_name)_col"
make_path_exist(col_t_plot_path)
savefig(col_t_plot_path)
println("Saved Plot: $(col_t_plot_path).png")
display(plt)


# -- Log(t) collapse

a_vals_to_plot = [0.7616]

plt = plot(
    title=L"Collapsed $\mathrm{Std}(λ(t))$ for $N=%$N_val$" * "\n" * L"$t_f=L^{%$(z_val)}$ | $z=%$(z_collapse_val)$",
    xlabel=L"\mathrm{log}(\frac{t}{L^z})",
    ylabel=L"\mathrm{Std}(λ)"
)
for (i, L) in enumerate(num_unit_cells_vals * N_val)
    L = Int(L)
    c = blue_palette[i]
    for a_val in a_vals_to_plot
        num_time_steps = length(collected_lambda_STD_series[L][a_val])
        plot!(log.(trunc(Int, num_time_steps*peak_frac_lims[1]):trunc(Int, num_time_steps*peak_frac_lims[2])) .- log(L^z_collapse_val),
            (collected_lambda_STD_series[L][a_val])[trunc(Int, num_time_steps*peak_frac_lims[1]):trunc(Int, num_time_steps*peak_frac_lims[2])], 
            # yerror=collected_lambda_STD_series[L][a_val],
            label="L = $L",
            color=c)
    end
end

display(plt)
log_col_t_plot_path = "figs/lambda_per_t/N$(N_val)/SeveralAs/IC$num_initial_conds/LSeveral/log_collapsed_std_lambda_per_t_N$(N_val)_ar$(replace("$(minimum(a_vals_to_plot))_$(maximum(a_vals_to_plot))", "." => "p"))_IC$(num_initial_conds)_z$(z_val_name)"
make_path_exist(log_col_t_plot_path)
savefig(log_col_t_plot_path)
println("Saved Plot: $(log_col_t_plot_path).png")

# Making inset plot

inset_log_lin = plot(
    title="",
    xlabel=L"\mathrm{log}(t)",
    ylabel=L"\mathrm{Std}(λ)"
)

plt_combined_log_lin = plot(log_plt, inset_subplots = [(inset_log_lin, bbox(0.26, 0.1, 0.5, 0.45))])

# Plot data for each L
for (i, L) in enumerate(num_unit_cells_vals * N_val)
    L = Int(L)
    c = blue_palette[i]
    for a_val in a_vals_to_plot
        num_time_steps = length(collected_lambda_STD_series[L][a_val])
        plot!(plt_combined_log_lin[2], log.(trunc(Int, num_time_steps*peak_frac_lims[1]):trunc(Int, num_time_steps*peak_frac_lims[2])) .- log(L^z_collapse_val),
            (collected_lambda_STD_series[L][a_val])[trunc(Int, num_time_steps*peak_frac_lims[1]):trunc(Int, num_time_steps*peak_frac_lims[2])], 
            # yerror=collected_lambda_STD_series[L][a_val],
            # label="L = $L",
            color=c)
    end
end

plot!(plt_combined_log_lin[2], xlabel=L"$\mathrm{log}(t/L^z)$",
    ylabel=L"\mathrm{Std}(λ)", xticks=[], yticks=[], guidefont = font(12), legend=nothing, framestyle = :box,)

fss_loglin_inset_scaled_std_lambda_per_a_plot_path = "figs/lambda_per_t/N$(N_val)/SeveralAs/IC$num_initial_conds/LSeveral/inset_log_collapsed_std_lambda_per_t_N$(N_val)_ar$(replace("$(minimum(a_vals_to_plot))_$(maximum(a_vals_to_plot))", "." => "p"))_IC$(num_initial_conds)_z$(z_val_name)"
make_path_exist(fss_loglin_inset_scaled_std_lambda_per_a_plot_path)
savefig(plt_combined_log_lin, fss_loglin_inset_scaled_std_lambda_per_a_plot_path)
println("Saved Plot: $(fss_loglin_inset_scaled_std_lambda_per_a_plot_path)")
display(plt_combined_log_lin)
