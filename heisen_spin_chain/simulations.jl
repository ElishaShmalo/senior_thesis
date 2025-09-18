# Hyzenberg Spin Chain Simulation in Julia
# Doing ferromagnet because it's cooler :)

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

# Set plotting theme
Plots.theme(:dark)
# General Variables
global_L = 256  # number of spins
J = 1       # energy factor

# J vector
J_vec = J .* [1, 1, 1]

# Time step for evolution
Tau_F = div(1,J)

# Evolve until
T = global_L

# --- Trying to Replecate Results ---
num_init_cond = 1 # We are avraging over x initial conditions
# a_vals = [round(0.5 + i*0.1, digits = 2) for i in 0:5] 
a_vals = [0.74, 0.755] 

println(a_vals)

# We will use an array to store the results of the simulation for each a_val. 
# First dimention represents the particular initial condition, Second is the time, third is the state at that time, 
# fourth is the particular spin value
# i.e. evolved_states[1, 2, 3, :] is the third spin at time t=2 for the first initial condition 
# i.e. evolved_states[1, 2, :, :] is an array represnting the state of t=2, first initial condition

# N_vals = [2, 4, 3, 6, 9, 10]
N_vals = [4]

N4_rand = 1 # N4_rand ∈ {0, 1, 2}. 0: No random Js, 1: random J_x and J_y, 2: random J_x

for N_val in N_vals
    println("N: $N_val")

    L = get_nearest(N_val, global_L)
    
    # Define s_naught as a constant
    S_NAUGHT = make_spiral_state(L, (2) / N_val)
    
    num_timesteps = L

    state_evolve_func = global_control_evolve
    if N_val == 4 && N4_rand == 1 # Need to evolve with randomized J_vec for N=4
        state_evolve_func = random_global_control_evolve
    elseif N_val == 4 && N4_rand == 2
        state_evolve_func = semirand_global_control_evolve
    end

    J_vec = J .* [1, 1, 1]

    for a_val in a_vals
        println("N: $N_val | a_val $a_val")
        current_spin_delta = [Vector{Vector{Float64}}([]) for _ in 1:num_init_cond] # []: Initial condition, Vec: Time, Vec: spin location, Float: δs

        for i in 1:num_init_cond
            println("N: $N_val | a_val $a_val | IC $(i)")
            original_random = make_random_state(L)
            returned_states = state_evolve_func(J_vec, original_random, a_val, num_timesteps, Tau_F, S_NAUGHT)

            current_spin_delta[i] = [get_delta_spin(state, S_NAUGHT) for state in returned_states]
        end

        # saving avrage of δs for future ref
        aval_path = "$(replace("$a_val", "." => "p"))"
        if length(aval_path) > 3
            aval_path = "$(aval_path[1:3])" * "/a$(aval_path)"
        end
        results_file_name = "N$(N_val)/a$(aval_path)/IC$(num_init_cond)/L$L/N$(N_val)_a" * replace("$a_val", "." => "p") * "_IC$(num_init_cond)_L$(L)_rand$N4_rand"
        
        make_path_exist("data/delta_evolved_spins/" * results_file_name * "_avg.dat")
        open("data/delta_evolved_spins/" * results_file_name * "_avg.dat", "w") do io
            serialize(io, sum(current_spin_delta)/num_init_cond)
        end
        println("Made file $("data/delta_evolved_spins/" * results_file_name * "_avg.dat")")
    end
end

# --- Plotting the Dynamics of S_diif ---

for N_val in N_vals
    L = get_nearest(N_val, global_L)
    plt = plot()
    other = true
    for a_val in a_vals
        other = !other
        if other
            results_file_name = "N$(N_val)/N_$(N_val)_a_val_" * replace("$a_val", "." => "p") * "_IC$(num_init_cond)_L$(L)"
            if N_val == 4 && N4_rand == 0
                results_file_name = results_file_name * "_nonrand"
            elseif N_val == 4 && N4_rand == 2
                results_file_name = results_file_name * "_semirand"
            end

            delta_spins = open("data/delta_evolved_spins/" * results_file_name * "_avg.dat", "r") do io
                deserialize(io)
            end

            ts = [i * Tau_F for i in 0:size(delta_spins)[1]-1]

            S_diff = get_spin_diffrence_from_delta(delta_spins)

            plot!(ts, S_diff, label="a = $(a_val)")
        end
    end

    xlabel!("time")
    ylabel!("S_diff")
    title!("Spin Dynamics for N = $N_val")
    # display(plt)

    pic_file_name = "N$(N_val)/s_diff_plot_diffrent_a_vals_N_$(N_val)_IC$(num_init_cond)_L$(L)"
    if N_val == 4 && N4_rand == 0
        pic_file_name = pic_file_name * "_nonrand"
    elseif N_val == 4 && N4_rand == 2
        pic_file_name = pic_file_name * "_semirand"
    end

    savefig("figs/delta_evolved_spins/$(pic_file_name).png")
end
