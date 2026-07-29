# Imports
using Random
using LinearAlgebra
using Plots
using DifferentialEquations
using StaticArrays
using Serialization
using Statistics
using LaTeXStrings
using Plots.PlotMeasures


# Other files   
include("utils/make_spins.jl")
include("utils/general.jl")
include("utils/dynamics.jl")
include("analytics/spin_diffrences.jl")

default(
    guidefont = 30,     # alternative, some backends use 'guidefont'
    tickfont = 28,      # font size for axis tick marks
    margin = 5mm
)
# To switch bw OTOC and SDiff change "data_type_to_heat", "num_init_cond", and "epsilon=0.01/0.0"
a_vals = [0.72, 0.75, 0.755, 0.758, 0.8]


# General Variables
L = 4*16  # number of spins
N_val = 4
num_init_cond = 1
num_init_cond = 5000
epsilon = 0.01
epsilon_name = ""
if epsilon != 0
    epsilon_name = replace("ep$(epsilon)_", "." => "p")
end

mid_pert = true

Js_rand = 0

for (i, a_val) in enumerate(a_vals)
    aval_path = "$(replace("$a_val", "." => "p"))"
    for trial_num in 1:trial_nums
        data_type_to_heat = "OTOC"
        c_map = :jet

        # results_file_name = "N$(N_val)/a$(aval_path)/IC$(num_init_cond)/L$L/N$(N_val)_a" * replace("$a_val", "." => "p") * "_IC$(num_init_cond)_L$(L)_rand$Js_rand"
        results_file_path = "data/delta_evolved_spins/N4/a$(aval_path)/IC$(num_init_cond)/L$(L)/N4_a$(aval_path)_IC$(num_init_cond)_L$(L)_$(epsilon_name)trial$(trial_num)_time_rand_$(data_type_to_heat)_MidPert$(mid_pert).data" # 

        delta_spins = open(results_file_path, "r") do io
            deserialize(io)
        end
        if mid_pert
            delta_spins = 1/sum(delta_spins[1]) .* delta_spins
        else
            delta_spins = [spin_chain ./ delta_spins[1] for spin_chain in delta_spins]
        end


        if typeof(delta_spins) == Vector{Vector{Float64}}
            delta_spins = hcat(delta_spins...)'
        end
        plt = plot()
        
        y_lims = [0, 100]
        heatmap!(log.(abs.(delta_spins)),
            # colorbar_title="δS",
            c=c_map,
            yflip=false,
            # margin=10 # adds space around everything, including the colorbar title
            ylims = y_lims,
            size = (700, 600),
            xlabel = L"\mathbf{j}",
            # ylabel = (i == 1 ? L"\mathbf{t}" : ""),
            right_margin = 20mm,   
            # colorbar = (i == length(a_vals)),
            # xticks = round.(range(50, 450, length=3)),
            # yticks = i==1 ? round.(range(10, y_lims[2]-12, length=4)) : [],
            # clims = (0, 0.9)
        )

        title = "$(data_type_to_heat)"

        title!("$(title) | a = $a_val")
        # figpath = "figs/delta_spin_heatmaps/N$(N_val)/a$(aval_path)/IC$(num_init_cond)/L$(L)/N$(N_val)/a$(aval_path)_IC$(num_init_cond)_L$(L)_trial$(trial_num)_delta$(data_type_to_heat).png"
        figpath = "figs/delta_spin_heatmaps/a$(aval_path)_IC$(num_init_cond)_L$(L)_trial$(trial_num)_delta$(data_type_to_heat)_MidPert$(mid_pert).png"
        make_path_exist(figpath)
        savefig(plt, figpath)
        display(plt)
        println(figpath)
    end
end


# prep_save_plot("figs/delta_spin_heatmaps/$(results_file_path).png")
# savefig("figs/delta_spin_heatmaps/$(results_file_path).png")

# println(readdir("data/delta_evolved_spins/"*"N$(N_val)/a$(aval_path)/IC$(num_init_cond)/L$(get_nearest(N_val, L))/"))
