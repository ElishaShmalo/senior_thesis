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
@everywhere using Random, LinearAlgebra, DifferentialEquations, Serialization, Statistics, DelimitedFiles, CSV, DataFrames

# Other files   
@everywhere include($(joinpath(@__DIR__, "..", "utils", "make_spins.jl")))
@everywhere include($(joinpath(@__DIR__, "..", "utils", "general.jl")))
@everywhere include($(joinpath(@__DIR__, "..", "utils", "dynamics.jl")))
@everywhere include($(joinpath(@__DIR__, "..", "utils", "lyapunov.jl")))
@everywhere include($(joinpath(@__DIR__, "..", "analytics", "spin_diffrences.jl")))

@time begin
    
# General Variables
@everywhere num_unit_cells_vals = [500]
# @everywhere num_unit_cells_vals = [128]
# @everywhere num_unit_cells_vals = [64]
@everywhere J = 1    # energy factor

# J vector with some randomness
@everywhere J_vec = J .* [1, 1, 1]

# Time to evolve until push back to S_A
@everywhere tau = 1 * J

# --- Trying to Replecate Results ---
@everywhere num_initial_conds = 500 # We are avraging over x initial conditions
@everywhere init_cond_name_offset = 1000

a_c = 0.758
rate = 0.0003
a_vals = [round(a_c + i *rate, digits=6) for i in -3:3] # general a_vals
# a_vals = [0.7] # general a_vals

@everywhere N_val = 4

time_prefact = 20
step_size = 200

if LOCAL_TEST   # tiny run for heisen_spin_chain/tests/run_spin_tests.sh
    @everywhere num_unit_cells_vals = [2, 4]
    @everywhere num_initial_conds = 2
    @everywhere init_cond_name_offset = 0
    a_vals = [0.7, 0.76]
    time_prefact = 2
    step_size = 3
end

# --- geting spin dists ---

for num_unit_cells in num_unit_cells_vals
    L = num_unit_cells * N_val
    println("L_val: $L")

    # Time we evolve to
    n = Int(round(L*time_prefact))

    states_evolve_func = random_global_control_evolve   # not used any more; see random_global_control_sdiff below

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

                # 2026-09 (review 4.1.2-4.1.4): S_diff measured on the fly (O(L) memory instead of storing
                # the whole trajectory); t column holds the true times 0, step_size, 2*step_size, ..., n
                times, current_sdiffs = random_global_control_sdiff(J_vec, spin_chain_A, a_val, n, J, S_NAUGHT;
                                                                     record_every=step_size)

                sample_filepath = "$(data_root)/s_diff_per_time_v2/N$N_val/a$a_val_name/IC1/L$L/N$(N_val)_a$(a_val_name)_IC1_L$(L)_timepref$(time_prefact)_timestep$(step_size)_sample$(init_cond+init_cond_name_offset).csv"
                make_path_exist(sample_filepath)
                df = DataFrame(t = times, s_diff = current_sdiffs)
                CSV.write(sample_filepath, df)
            end
        end
    end
end

end
