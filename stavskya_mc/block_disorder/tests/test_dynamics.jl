# Tests for the 2026-09 refactor of stavskya_mc/utils/dynamics.jl
#
# Run from anywhere with the senior_thesis project environment:
#     julia --project=<path to senior_thesis> stavskya_mc/block_disorder/tests/test_dynamics.jl
#
# The script checks that:
#   1. the new type-stable kernel is bit-identical to the old code for every
#      evolve function, with block_len = 1 and a fixed seed (identical output
#      AND identical RNG state afterwards);
#   2. the output element type matches the input's (the old code alternated
#      between Int and Float64 depending on whether time_steps was odd or even);
#   3. block_len > 1 redraws epsilon exactly once per block, and the recorder
#      keeps the block phase aligned to absolute time;
#   4. time_random_p_record_rho gives exactly the same rho(t) as chunked
#      evolution (block_len = 1) and as a hand-written reference loop
#      (block_len > 1);
#   5. make_log_times behaves as documented;
#   6. absorbing-state and basic phase sanity checks pass;
# and it prints a speed comparison of old vs new.

using Test, Random, Statistics

const HERE = @__DIR__
include(joinpath(HERE, "..", "..", "utils", "dynamics.jl"))     # new code (+ general.jl)
include(joinpath(HERE, "reference_dynamics_pre_refactor.jl"))   # frozen old code, *_old names

rho(s) = sum(s) / length(s)

@testset "Stavskaya dynamics refactor" begin

    @testset "bit-identical to old code (block_len = 1)" begin
        cases = [
            ("evolve_state",        (s, T) -> evolve_state(s, T, 0.28),                         (s, T) -> evolve_state_old(s, T, 0.28)),
            ("delta",               (s, T) -> time_random_delta_evolve_state(s, T, 0.29, 0.05), (s, T) -> time_random_delta_evolve_state_old(s, T, 0.29, 0.05)),
            ("n",                   (s, T) -> time_random_n_evolve_state(s, T, 0.25, 0.5),      (s, T) -> time_random_n_evolve_state_old(s, T, 0.25, 0.5)),
            ("gauss",               (s, T) -> time_random_gauss_evolve_state(s, T, 0.29, 0.1),  (s, T) -> time_random_gauss_evolve_state_old(s, T, 0.29, 0.1)),
            ("binary",              (s, T) -> time_random_binary_evolve_state(s, T, 0.15),      (s, T) -> time_random_binary_evolve_state_old(s, T, 0.15)),
            ("upper/lower p",       (s, T) -> time_random_p_evolve_state(s, T, 0.3337, 0.016687, 0.8),
                                    (s, T) -> time_random_p_evolve_state_old(s, T, 0.3337, 0.016687, 0.8)),
        ]
        for (name, fnew, fold) in cases, L in (1, 2, 97, 1000), T in (0, 1, 2, 7, 50, 301), seed in (1, 42)
            Random.seed!(seed); s0 = make_rand_state(L, 0.5)
            Random.seed!(seed + 1000); snew = fnew(s0, T); after_new = rand()
            Random.seed!(seed + 1000); sold = fold(s0, T); after_old = rand()
            @test Int.(snew) == Int.(sold)
            @test after_new == after_old          # same number of RNG draws, same order
        end
    end

    @testset "type stability of the returned state" begin
        s0 = make_rand_state(100, 0.5)
        for T in (0, 1, 2, 3)
            @test eltype(time_random_p_evolve_state(s0, T, 0.33, 0.017, 0.8)) == Int
            @test eltype(evolve_state(s0, T, 0.3)) == Int
        end
        @test (@inferred evolve_state(s0, 5, 0.3)) isa Vector{Int}
    end

    @testset "block_len semantics" begin
        for b in (1, 2, 5, 6), T in (0, 1, 5, 6, 7, 30, 31)
            ncalls = Ref(0)
            _evolve_with_epsilon_draw(make_rand_state(20, 0.5), T, () -> (ncalls[] += 1; 0.3); block_len=b)
            @test ncalls[] == cld(T, b)
        end
        @test_throws ArgumentError time_random_p_evolve_state(make_rand_state(10, 0.5), 5, 0.3, 0.01, 0.8; block_len=0)
        # default keyword == block_len = 1
        Random.seed!(7); s0 = make_rand_state(200, 0.5)
        Random.seed!(8); a = time_random_p_evolve_state(s0, 100, 0.33, 0.017, 0.8)
        Random.seed!(8); b = time_random_p_evolve_state(s0, 100, 0.33, 0.017, 0.8; block_len=1)
        @test a == b
    end

    @testset "recorder == chunked evolution (block_len = 1)" begin
        for seed in 1:5
            Random.seed!(seed); s0 = make_rand_state(300, 0.5)
            ts = [0, 1, 2, 5, 10, 11, 40, 100, 250]
            Random.seed!(seed + 99)
            r_rec = time_random_p_record_rho(s0, ts, 0.3337, 0.016687, 0.8)
            Random.seed!(seed + 99)
            s = copy(s0); r_chunk = Float64[]; tprev = 0
            for t in ts
                s = time_random_p_evolve_state(s, t - tprev, 0.3337, 0.016687, 0.8)
                push!(r_chunk, rho(s)); tprev = t
            end
            @test r_rec == r_chunk
        end
    end

    @testset "recorder == reference loop (block_len > 1, phase aligned to absolute time)" begin
        for b in (2, 6), seed in 1:3
            Random.seed!(seed); s0 = make_rand_state(250, 0.5)
            ts = [0, 3, 4, 13, 60, 200]
            Random.seed!(seed + 7)
            r_rec = time_random_p_record_rho(s0, ts, 0.43, 0.0215, 0.48; block_len=b)
            # reference: plain loop, epsilon redrawn when (t-1) % b == 0
            Random.seed!(seed + 7)
            cur = copy(s0); L = length(cur); eps = 0.0; r_ref = Float64[]
            ts[1] == 0 && push!(r_ref, rho(cur))
            for t in 1:last(ts)
                (t - 1) % b == 0 && (eps = rand() > 0.48 ? 0.0215 : 0.43)
                u = rand(L)
                nxt = similar(cur)
                nxt[1] = u[1] < eps ? 1 : cur[L] * cur[1]
                for i in 2:L
                    nxt[i] = u[i] < eps ? 1 : cur[i-1] * cur[i]
                end
                cur = nxt
                t in ts && push!(r_ref, rho(cur))
            end
            @test r_rec == r_ref
        end
        @test_throws ArgumentError time_random_p_record_rho(make_rand_state(10, 0.5), [5, 3], 0.3, 0.01, 0.8)
    end

    @testset "make_log_times" begin
        for tmax in (1, 7, 10, 1000, 123_456), ppd in (1, 10, 20)
            ts = make_log_times(tmax; points_per_decade=ppd)
            @test first(ts) == 0 && last(ts) == tmax
            @test issorted(ts) && allunique(ts)
            @test length(ts) <= ceil(Int, ppd * log10(max(tmax, 1))) + 3
        end
        ts = make_log_times(10^6; points_per_decade=20)
        @test 100 < length(ts) < 125          # about 20 per decade over 6 decades
    end

    @testset "absorbing state and phase sanity" begin
        s = ones(Int, 500)
        @test all(==(1.0), time_random_p_record_rho(s, [0, 10, 1000], 0.3, 0.01, 0.8))
        @test all(==(1), evolve_state(s, 100, 0.2))
        # clean model: deep inactive (eps = 0.6 > eps_c ~ 0.2945) is absorbed quickly
        Random.seed!(3)
        @test rho(evolve_state(make_rand_state(2000, 0.5), 300, 0.6)) == 1.0
        # clean model: deep active (eps = 0.1) keeps a finite activity
        @test 1 - rho(evolve_state(make_rand_state(2000, 0.5), 300, 0.1)) > 0.3
        # recorder fills with ones once absorbed
        Random.seed!(4)
        r = time_random_p_record_rho(make_rand_state(200, 0.5), make_log_times(10^5), 0.9, 0.9, 0.8)
        @test last(r) == 1.0 && !any(isnan, r)
    end
end

# ---------------------------------------------------------------------------
# Speed comparison (informational, not a test)
# ---------------------------------------------------------------------------
let L = 20_000, T = 2_000
    s0 = make_rand_state(L, 0.5)
    time_random_p_evolve_state(s0, 10, 0.33, 0.017, 0.8); time_random_p_evolve_state_old(s0, 10, 0.33, 0.017, 0.8)
    t_new = @elapsed time_random_p_evolve_state(s0, T, 0.33, 0.017, 0.8)
    t_old = @elapsed time_random_p_evolve_state_old(s0, T, 0.33, 0.017, 0.8)
    a_new = @allocated time_random_p_evolve_state(s0, T, 0.33, 0.017, 0.8)
    a_old = @allocated time_random_p_evolve_state_old(s0, T, 0.33, 0.017, 0.8)
    println("\nL=$L, T=$T:  old $(round(t_old, digits=3)) s, $(round(a_old/2^20, digits=1)) MiB allocated")
    println("             new $(round(t_new, digits=3)) s, $(round(a_new/2^20, digits=1)) MiB allocated")
    println("             speed-up x$(round(t_old/t_new, digits=2))")
end
