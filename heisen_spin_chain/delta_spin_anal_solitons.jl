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
    guidefont = 22,     # alternative, some backends use 'guidefont'
    tickfont = 15,      # font size for axis tick marks
    margin = 5mm
)

# General Variables
L = 4*256  # number of spins
N_val = 4
num_init_cond = 1
epsilon = 0.0
epsilon_name = ""
if epsilon != 0
    epsilon_name = replace("ep$(epsilon)_", "." => "p")
end

Js_rand = 0

# Heat map of delta_spins
a_vals = [0.72] 
trial_nums = 1

for a_val in a_vals
    aval_path = "$(replace("$a_val", "." => "p"))"
    for trial_num in 1:trial_nums
        data_type_to_heat = "deltaS"
        # data_type_to_heat = "OTOC"
        c_map = :jet
        if data_type_to_heat == "deltaS"
            c_map = :gist_rainbow
        end

        # results_file_name = "N$(N_val)/a$(aval_path)/IC$(num_init_cond)/L$L/N$(N_val)_a" * replace("$a_val", "." => "p") * "_IC$(num_init_cond)_L$(L)_rand$Js_rand"
        # results_file_path = "data/delta_evolved_spins/N4/a$(aval_path)/IC$(num_init_cond)/L$(L)/N4_a$(aval_path)_IC$(num_init_cond)_L$(L)_$(epsilon_name)trial$(trial_num)_time_rand_$(data_type_to_heat).data" # 
        results_file_path = "data/delta_evolved_spins/N4/a0p6/IC1/L1024/N4_a0p62_IC1_L1024_rand0_seksolNumOff40_EppOff0p01_true_rand_avg.dat" # 

        delta_spins = open(results_file_path, "r") do io
            deserialize(io)
        end

        if typeof(delta_spins) == Vector{Vector{Float64}}
            delta_spins = hcat(delta_spins...)'
        end
        plt = plot()
        y_lims = nothing
        if data_type_to_heat == "OTOC"
            y_lims = [0, 100]
        end
        heatmap!(delta_spins,
            # colorbar_title="δS",
            c=c_map,
            yflip=false,
            # margin=10 # adds space around everything, including the colorbar title
            ylims = y_lims,
            size = (600, 500) # square figure
        )

        title = "$(data_type_to_heat)"
        if data_type_to_heat == "deltaS"
            title = "δS"
        end

        xlabel!(L"\mathbf{j}")
        ylabel!(L"\mathbf{t}")
        # title!("$(title) | N = $N_val | a = $a_val | IC = $num_init_cond | L = $(get_nearest(N_val, L))")
        # figpath = "figs/delta_spin_heatmaps/N$(N_val)/a$(aval_path)/IC$(num_init_cond)/L$(L)/N$(N_val)/a$(aval_path)_IC$(num_init_cond)_L$(L)_trial$(trial_num)_delta$(data_type_to_heat).png"
        figpath = "figs/delta_spin_heatmaps/a$(aval_path)_IC$(num_init_cond)_L$(L)_trial$(trial_num)_solitons_delta$(data_type_to_heat).png"
        make_path_exist(figpath)
        savefig(plt, figpath)
        display(plt)
        println(figpath)
    end
end


# prep_save_plot("figs/delta_spin_heatmaps/$(results_file_path).png")
# savefig("figs/delta_spin_heatmaps/$(results_file_path).png")

# println(readdir("data/delta_evolved_spins/"*"N$(N_val)/a$(aval_path)/IC$(num_init_cond)/L$(get_nearest(N_val, L))/"))

# ----- Let us do a fouier transfrom -----

# a_vals = [0.5, 0.59, 0.6, 0.61, 0.62, 0.7]
a_vals = [0.62]

include("utils/fft_komega.jl")

dt = 1.0
dx = 1.0

for a_val in a_vals
    aval_path = "$(replace("$a_val", "." => "p"))"
    for trial_num in 1:trial_nums
        results_file_path = "data/delta_evolved_spins/N4/a$(aval_path[1:3])/IC1/L1024/N4_a$(aval_path)_IC1_L1024_rand0_seksolNumOff40_EppOff0p01_true_rand_avg.dat"

        delta_spins = open(results_file_path, "r") do io
            deserialize(io)
        end
        if typeof(delta_spins) == Vector{Vector{Float64}}
            delta_spins = hcat(delta_spins...)'
        end

        # verify_roundtrip(delta_spins; dt=1., dx=1., verbose=true)
        
        F, ks, ωs = fft_komega(delta_spins; dt=dt, dx=dx)
        D, js, ts = ifft_komega(F; dj=1., dt=1.)

        orig = sqrt.(abs2.(D))

        plt = heatmap(js, ts, orig,
            c=:gist_rainbow,          
            yflip=false,
            size=(800, 700),
            colorbar_title="\n |δS|",
            colorbar_titlefontsize=18,   
            right_margin=20mm,
        )

        xlabel!(L"\mathbf{j}")
        ylabel!(L"\mathbf{t}")
        # Optional: zoom to the physical first Brillouin zone / low frequencies
        # xlims!(-π, π)
        # ylims!(0, maximum(ωs))
        title!("N = $N_val | a = $a_val | IC = $num_init_cond | L = $(get_nearest(N_val, L))")
        figpath = "figs/komega_heatmaps/a$(aval_path)_IC$(num_init_cond)_L$(L)_trial$(trial_num)_komega.png"
        make_path_exist(figpath)
        savefig(plt, figpath)
        display(plt)
        println(figpath)

        # --- Fourier transform into (k, ω) space ---
        S, ks, ωs = to_komega(delta_spins; dt=dt, dx=dx,
                              logscale=true, subtract_mean=false)
        orig_data = 

        plt = heatmap(ks, ωs, S,
            c=:viridis,          
            yflip=false,
            size=(600, 500),
            colorbar_title=L"\log_{10}|S(k,\omega)|^2",
            colorbar_titlefontsize=18,   
            right_margin=12mm,
        )
        
        xlabel!(L"\mathbf{k}")
        ylabel!(L"\mathbf{\omega}")
        # Optional: zoom to the physical first Brillouin zone / low frequencies
        # xlims!(-π, π)
        # ylims!(0, maximum(ωs))
        title!("N = $N_val | a = $a_val | IC = $num_init_cond | L = $(get_nearest(N_val, L))")
        figpath = "figs/komega_heatmaps/a$(aval_path)_IC$(num_init_cond)_L$(L)_trial$(trial_num)_komega.png"
        make_path_exist(figpath)
        savefig(plt, figpath)
        display(plt)
        println(figpath)
    end
end

# --- Test inverse ---

dt = 1.0
dx = 1.0

for a_val in a_vals
    aval_path = "$(replace("$a_val", "." => "p"))"
    for trial_num in 1:trial_nums
        results_file_path = "data/delta_evolved_spins/N4/a$(aval_path)/IC1/L1024/N4_a$(aval_path)_IC1_L1024_rand0_seksolNumOff40_EppOff0p01_true_rand_avg.dat"

        delta_spins = open(results_file_path, "r") do io
            deserialize(io)
        end
        if typeof(delta_spins) == Vector{Vector{Float64}}
            delta_spins = hcat(delta_spins...)'
        end

        # verify_roundtrip(delta_spins; dt=1., dx=1., verbose=true)

        F, ks, ωs = fft_komega(delta_spins; dt=dt, dx=dx)
        D, js, ts = ifft_komega(F; dj=1., dt=1.)

        orig = sqrt.(abs2.(D))

        plt = heatmap(js, ts, orig,
            c=:gist_rainbow,          
            yflip=false,
            size=(800, 700),
            colorbar_title="\n |δS|",
            colorbar_titlefontsize=18,   
            right_margin=20mm,
        )

        xlabel!(L"\mathbf{j}")
        ylabel!(L"\mathbf{t}")
        # Optional: zoom to the physical first Brillouin zone / low frequencies
        # xlims!(-π, π)
        # ylims!(0, maximum(ωs))
        title!("N = $N_val | a = $a_val | IC = $num_init_cond | L = $(get_nearest(N_val, L))")
        figpath = "figs/komega_heatmaps/a$(aval_path)_IC$(num_init_cond)_L$(L)_trial$(trial_num)_komega.png"
        make_path_exist(figpath)
        savefig(plt, figpath)
        display(plt)
        println(figpath)
    end
end

