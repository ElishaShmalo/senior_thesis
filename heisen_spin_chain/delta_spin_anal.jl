# Imports
using Random
using LinearAlgebra
using Plots
using DifferentialEquations
using StaticArrays
using Serialization
using Statistics

# Other files   
include("utils/make_spins.jl")
include("utils/general.jl")
include("utils/dynamics.jl")
include("analytics/spin_diffrences.jl")

# General Variables
L = 4*128  # number of spins
N_val = 4
num_init_cond = 1000

Js_rand = 0

# Heat map of delta_spins
a_vals = [0.75]
trial_nums = 1

for a_val in a_vals
    aval_path = "$(replace("$a_val", "." => "p"))"
    for trial_num in 1:trial_nums
        data_type_to_heat = "deltaS"
        data_type_to_heat = "OTOC"
        c_map = :hsv
        if data_type_to_heat == "OTOC"
            c_map = :jet
        end

        # results_file_name = "N$(N_val)/a$(aval_path)/IC$(num_init_cond)/L$L/N$(N_val)_a" * replace("$a_val", "." => "p") * "_IC$(num_init_cond)_L$(L)_rand$Js_rand"
        results_file_path = "data/delta_evolved_spins/N4/a$(aval_path)/IC$(num_init_cond)/L$(L)/N4_a$(aval_path)_IC$(num_init_cond)_L$(L)_trial$(trial_num)_time_rand_$(data_type_to_heat).data" # 

        delta_spins = open(results_file_path, "r") do io
            deserialize(io)
        end

        if typeof(delta_spins) == Vector{Vector{Float64}}
            delta_spins = hcat(delta_spins...)'
        end
        plt = plot()
        heatmap!(delta_spins,
            # colorbar_title="δS",
            c=c_map,
            yflip=false,
            # margin=10 # adds space around everything, including the colorbar title
        )

        xlabel!("x")
        ylabel!("t")
        title!("$(data_type_to_heat) | N = $N_val | a = $a_val | IC = $num_init_cond | L = $(get_nearest(N_val, L))")
        figpath = "figs/delta_spin_heatmaps/N$(N_val)/a$(aval_path)/IC$(num_init_cond)/L$(L)/N$(N_val)/a$(aval_path)_IC$(num_init_cond)_L$(L)_trial$(trial_num)_delta$(data_type_to_heat).png"
        make_path_exist(figpath)
        savefig(plt, figpath)
        display(plt)
        println(figpath)
    end
end


# prep_save_plot("figs/delta_spin_heatmaps/$(results_file_path).png")
# savefig("figs/delta_spin_heatmaps/$(results_file_path).png")

# println(readdir("data/delta_evolved_spins/"*"N$(N_val)/a$(aval_path)/IC$(num_init_cond)/L$(get_nearest(N_val, L))/"))
