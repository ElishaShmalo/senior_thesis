# In this file we numerically approximate the Lyapunov for diffrent a-vals and N (for the spiral) vals using the tech from Benettin
using Distributed

# SPIN_LOCAL_TEST=1 (2026-09): 2 local workers, tiny sizes, output under
# heisen_spin_chain/tests/_local_test_output/. Used by heisen_spin_chain/tests/run_spin_tests.sh.
# Without it the script runs on Slurm exactly as before.
const LOCAL_TEST = get(ENV, "SPIN_LOCAL_TEST", "0") == "1"
if LOCAL_TEST
    nprocs() == 1 && addprocs(2)
    println("LOCAL TEST MODE: $(nworkers()) local workers")
else
    @eval using SlurmClusterManager
    Base.invokelatest() do
        println("We are adding $(SlurmManager()) workers")
        addprocs(SlurmManager())
    end
end
@everywhere workers() const LOCAL_TEST = $LOCAL_TEST
data_root = LOCAL_TEST ? normpath(joinpath(@__DIR__, "..", "tests", "_local_test_output", "data")) : "data"

@everywhere begin
    println("Hello from worker $(myid()) on host $(gethostname())")
end

# Imports
@everywhere using Random, LinearAlgebra, DifferentialEquations, Serialization, Statistics, DelimitedFiles, SharedArrays, CSV, DataFrames

# Other files   
@everywhere include($(joinpath(@__DIR__, "..", "utils", "make_spins.jl")))
@everywhere include($(joinpath(@__DIR__, "..", "utils", "general.jl")))
@everywhere include($(joinpath(@__DIR__, "..", "utils", "dynamics.jl")))
@everywhere include($(joinpath(@__DIR__, "..", "utils", "lyapunov.jl")))
@everywhere include($(joinpath(@__DIR__, "..", "analytics", "spin_diffrences.jl")))

@time begin

# General Variables
@everywhere num_unit_cells_vals = [16]
# @everywhere num_unit_cells_vals = [128]
# @everywhere num_unit_cells_vals = [64]
@everywhere J = 1    # energy factor

# J vector with some randomness
@everywhere J_vec = J .* [1, 1, 1]

# Time to evolve until push back to S_A
@everywhere tau = 1 * J

# --- Trying to Replecate Results ---
@everywhere num_initial_conds = 1000 # We are avraging over x initial conditions
@everywhere init_cond_name_offset = 0
a_vals = [0.55, 0.58, 0.6, 0.62, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.7525, 0.755, 0.7563, 0.7575, 0.7588,
          0.7594, 0.76, 0.7605, 0.761, 0.7615, 0.762, 0.7625, 0.763, 0.765,
          0.7675, 0.77, 0.78, 0.79, 0.8] # general a_vals
# a_vals = [0.7] # general a_vals

@everywhere epsilon = 0.01

# NEW (2026-09): write only every record_every-th time step (linear time). "lambda" is then the
# mean over the record_every steps ending at t; appears as _timestep<k> in the file names.
@everywhere record_every = 1

@everywhere N_val = 4

z_val = 1.7
z_val_name = replace("$z_val", "." => "p")

if LOCAL_TEST   # tiny run for heisen_spin_chain/tests/run_spin_tests.sh
    @everywhere num_unit_cells_vals = [2, 4]
    @everywhere num_initial_conds = 2
    @everywhere init_cond_name_offset = 0
    a_vals = [0.7, 0.76]
end

# --- geting spin dists ---

for num_unit_cells in num_unit_cells_vals
    L = num_unit_cells * N_val
    println("L_val: $L")

    # number of pushes we are going to do
    n = Int(round(L^z_val))

    states_evolve_func = random_evolve_spins_to_time   # used inside benettin_lambda_sdiff

    # Define s_naught to be used during control step
    S_NAUGHT = make_spiral_state(L, (2) / N_val)

    for a_val in a_vals
        println("L_val: $L | a_val: $a_val")
        a_val_name = replace("$a_val", "." => "p")

        # define the variables for the workers to use
        let a_val=a_val, L=L, n=n, S_NAUGHT=S_NAUGHT, num_initial_conds=num_initial_conds, states_evolve_func=states_evolve_func, data_root=data_root
            @sync @distributed for init_cond in 1:num_initial_conds
                println("L_val: $L | a_val: $a_val | IC: $init_cond / $num_initial_conds")

                spin_chain_A = make_random_state(L) # our S_A

                # Benettin run, moved to utils/lyapunov.jl (2026-09) so it can be tested. B = A with the
                # middle spin kicked; each step: evolve A and B, ln(|d|/epsilon), S_diff(A), push B back.
                # Rows are written at t = record_every, 2*record_every, ...; "lambda" is the mean of
                # ln(|d|/epsilon) over the record_every steps ending at t (record_every = 1: every step, as before).
                times, lambdas, sdiffs = benettin_lambda_sdiff(J_vec, spin_chain_A, a_val, n, tau, J, S_NAUGHT, epsilon;
                                                               record_every=record_every)

                sample_filepath = "$(data_root)/spin_dists_per_time_v2/N$N_val/a$a_val_name/IC1/L$L/N$(N_val)_a$(a_val_name)_IC1_L$(L)_z$(z_val_name)_timestep$(record_every)_sample$(init_cond+init_cond_name_offset).csv"
                make_path_exist(sample_filepath)
                df = DataFrame("t" => times, "lambda" => lambdas, "delta_s" => sdiffs)
                CSV.write(sample_filepath, df)
            end
        end
    end
end

end