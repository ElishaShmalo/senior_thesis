# Tests for the 2026-09 spin-chain changes (review 4.1.2-4.1.4 and the get_good_data record_every knob).
# Run from the senior_thesis root:
#     julia --project=. heisen_spin_chain/tests/test_spin_refactor.jl
# (run_spin_tests.sh does this and more.)
#
# Checks:
#   1. random_global_control_evolve: same trajectory as the pre-2026-09 version (fixed seed, fresh
#      J vector), and the caller's J vector is no longer modified;
#   2. random_global_control_sdiff == S_diff of the stored trajectory, subsampled, with true times
#      (0, k, 2k, ...), for several record_every;
#   3. benettin_lambda_sdiff(record_every = 1) is bit-identical to the old inline loop of
#      get_good_data_severalL*.jl;
#   4. benettin_lambda_sdiff(record_every = k) gives block means of the k = 1 lambda and the k = 1
#      S_diff at the recorded times (same seed);
#   5. random_evolve_spins_to_time keeps |J_x|, |J_y| (only the sign is random).

using Test, Random, LinearAlgebra, Statistics, DifferentialEquations, SymPy
const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "utils", "make_spins.jl"))
include(joinpath(ROOT, "utils", "general.jl"))
include(joinpath(ROOT, "utils", "dynamics.jl"))
include(joinpath(ROOT, "utils", "lyapunov.jl"))
include(joinpath(ROOT, "analytics", "spin_diffrences.jl"))
include(joinpath(@__DIR__, "reference_pre_2026_09.jl"))

const L = 16
const S0 = make_spiral_state(L, 0.5)

@testset "spin-chain 2026-09 changes" begin

    @testset "random_global_control_evolve unchanged, caller J untouched" begin
        for seed in 1:3, a in (0.7, 0.76)
            Random.seed!(seed); s = make_random_state(L)
            Random.seed!(100 + seed); old = random_global_control_evolve_old([1, 1, 1], s, a, 25, 1, S0)
            Random.seed!(100 + seed); J = [1, 1, 1]
            new = random_global_control_evolve(J, s, a, 25, 1, S0)
            @test new == old
            @test J == [1, 1, 1]
            @test length(new) == 26 && new[1] == s          # element k is time k-1
        end
    end

    @testset "random_global_control_sdiff == stored trajectory, true times" begin
        for seed in 1:3, k in (1, 3, 7), T in (21, 25)
            Random.seed!(seed); s = make_random_state(L)
            Random.seed!(200 + seed)
            full = weighted_spin_difference_vs_time(random_global_control_evolve_old([1, 1, 1], s, 0.75, T, 1, S0), S0)
            Random.seed!(200 + seed); J = [1, 1, 1]
            times, sd = random_global_control_sdiff(J, s, 0.75, T, 1, S0; record_every=k)
            @test times == collect(0:k:T)                   # t = 0, k, 2k, ... (true times)
            @test sd == full[1:k:end]                       # full[i] is the state at t = i - 1
            @test J == [1, 1, 1]
        end
        @test_throws ArgumentError random_global_control_sdiff([1, 1, 1], make_random_state(L), 0.7, 5, 1, S0; record_every=0)
    end

    @testset "benettin_lambda_sdiff(record_every = 1) == old inline loop" begin
        for seed in 1:3, a in (0.7, 0.76), n in (1, 12, 40)
            Random.seed!(seed); s = make_random_state(L)
            Random.seed!(300 + seed); t_old, lam_old, sd_old = old_inline_benettin([1, 1, 1], s, a, n, 1, 1, S0, 0.01)
            Random.seed!(300 + seed); J = [1, 1, 1]
            t_new, lam_new, sd_new = benettin_lambda_sdiff(J, s, a, n, 1, 1, S0, 0.01)
            @test t_new == t_old
            @test lam_new == lam_old
            @test sd_new == sd_old
            @test J == [1, 1, 1]
        end
    end

    @testset "benettin_lambda_sdiff(record_every = k): block means and subsampled S_diff" begin
        for seed in 1:2, k in (2, 5), n in (20, 23)
            Random.seed!(seed); s = make_random_state(L)
            Random.seed!(400 + seed); t1, lam1, sd1 = benettin_lambda_sdiff([1, 1, 1], s, 0.75, n, 1, 1, S0, 0.01)
            Random.seed!(400 + seed); tk, lamk, sdk = benettin_lambda_sdiff([1, 1, 1], s, 0.75, n, 1, 1, S0, 0.01; record_every=k)
            nrec = div(n, k)
            @test tk == collect(k:k:nrec*k)
            @test sdk == sd1[k:k:nrec*k]
            @test lamk ≈ [mean(lam1[(j-1)*k+1:j*k]) for j in 1:nrec] rtol=1e-12
            # the window average used by analysis_lyapunov_fixed is unchanged
            @test mean(lamk) ≈ mean(lam1[1:nrec*k]) rtol=1e-12
        end
    end

    @testset "random_evolve_spins_to_time keeps |J|" begin
        Jv = [2.5, 0.5, 1.0]
        s = make_random_state(L)
        for _ in 1:10
            random_evolve_spins_to_time(Jv, s, s, 0.7, 1, 1, S0)
            @test abs.(Jv) == [2.5, 0.5, 1.0]
        end
    end
end
