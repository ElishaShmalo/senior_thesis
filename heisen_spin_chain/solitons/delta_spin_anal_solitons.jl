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
include("../utils/make_spins.jl")
include("../utils/general.jl")
include("../utils/dynamics.jl")
include("../analytics/spin_diffrences.jl")

default(
    guidefont = 22,     # alternative, some backends use 'guidefont'
    tickfont = 15,      # font size for axis tick marks
    margin = 5mm
)

# General Variables
L = 500  # number of spins
N_val = 5
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

# ----- Let us do a fouier transfrom -----

# a_vals = [0.67, 0.7, 0.72, 0.74, 0.8, 1.0] 
a_vals = [0.6, 0.65, 0.67, 0.69, 0.8, 1.0] 
# a_vals = [0.99] 


include("../utils/fft_komega.jl")

dt = 1.0
dx = 1.0

for a_val in a_vals
    aval_path = "$(replace("$a_val", "." => "p"))"
    for trial_num in 1:trial_nums
        results_file_path = "data/delta_evolved_spins/N$(N_val)/a$(aval_path[1:3])/IC1/L$(L)/N$(N_val)_a$(aval_path)_IC1_L$(L)_rand0_seksolNumOff50_EppOff0p01_true_rand_avg.dat"

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
        figpath = "heisen_spin_chain/solitons/figs/N$(N_val)/a$(aval_path)_IC$(num_init_cond)_L$(L)_trial$(trial_num).png"
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
        figpath = "heisen_spin_chain/solitons/figs/N$(N_val)/a$(aval_path)_IC$(num_init_cond)_L$(L)_trial$(trial_num)_komega.png"
        make_path_exist(figpath)
        savefig(plt, figpath)
        display(plt)
        println(figpath)
    end
end
