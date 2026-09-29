# Tests for the spreading code added 2026-09-29 (BVH reproduction):
#   * time_random_spreading / spreading_chunk (utils/dynamics.jl)
#   * spreading_chunk_path (generators/naming.jl)
#
#   julia --project=. stavskya_mc/block_disorder/tests/test_spreading.jl
# (run_all_tests.sh runs it.)

using Test, Random, Statistics

const HERE = @__DIR__
include(joinpath(HERE, "..", "..", "utils", "dynamics.jl"))
include(joinpath(HERE, "..", "generators", "naming.jl"))

# Reference: the same spreading run on a whole periodic lattice with the plain kernel.
# Single active site at x0 = L ÷ 4; L must exceed x0 + t_max so nothing wraps around.
function reference_spreading(record_times, u, l, p, block_len, L)
    x0 = L ÷ 4
    cur = ones(Int, L); nxt = similar(cur); buf = Vector{Float64}(undef, L)
    cur[x0] = 0
    n_out = zeros(Int, length(record_times)); x2_out = zeros(length(record_times)); inside = true
    k = 1
    while k <= length(record_times) && record_times[k] == 0
        n_out[k] = 1; k += 1
    end
    epsilon = 0.0
    for t in 1:last(record_times)
        if (t - 1) % block_len == 0
            epsilon = draw_upper_lower(u, l, p)
        end
        stavskaya_step!(nxt, cur, buf, epsilon)
        cur, nxt = nxt, cur
        if k <= length(record_times) && record_times[k] == t
            act = findall(==(0), cur)
            inside &= all(x0 .<= act .<= x0 + t)
            n_out[k] = length(act)
            x2_out[k] = sum((act .- (x0 + t / 2)) .^ 2; init=0.0)
            k += 1
        end
    end
    return n_out, x2_out, inside
end

@testset "spreading runs (2026-09-29)" begin

    @testset "spreading: exact cases" begin
        rec = [0, 1, 2, 3, 5, 8, 13, 20, 30, 40]
        n, x2 = time_random_spreading(rec, 0.0, 0.0, 0.5)          # nothing ever heals
        @test n == rec .+ 1
        @test x2 ≈ rec .* (rec .+ 1) .* (rec .+ 2) ./ 12            # sum_{k=0}^t (k - t/2)^2
        n, x2 = time_random_spreading(rec, 1.0, 1.0, 0.5)          # everything heals at t = 1
        @test n == [1; zeros(Int, length(rec) - 1)]
        @test all(iszero, x2)
        n, _ = time_random_spreading(rec, 1.0, 0.0, 0.0)           # p = 0: always the lower value (0)
        @test n == rec .+ 1
        n, _ = time_random_spreading([7, 50], 0.0, 0.0, 0.5)       # t = 0 not recorded
        @test n == [8, 51]
        @test time_random_spreading(Int[], 0.3, 0.1, 0.5) == (Int[], Float64[])
        @test_throws ArgumentError time_random_spreading([5, 3], 0.3, 0.1, 0.5)
        @test_throws ArgumentError time_random_spreading([1, 5], 0.3, 0.1, 0.5; block_len=0)
    end

    @testset "spreading: agrees with a full-lattice simulation" begin
        rec = [0, 1, 2, 3, 5, 8, 13, 20, 30, 40]
        R = 3000
        for (u, l, p, b) in ((0.33, 0.0165, 0.5, 3), (0.6, 0.03, 0.2, 6), (0.2945, 0.014725, 1.0, 1))
            Random.seed!(11)
            A = [time_random_spreading(rec, u, l, p; block_len=b) for _ in 1:R]
            B = [reference_spreading(rec, u, l, p, b, 120) for _ in 1:R]
            @test all(r[3] for r in B)                              # light cone x0 <= i <= x0 + t
            for (name, fa, fb) in (("P_s", r -> Float64.(r[1] .> 0), r -> Float64.(r[1] .> 0)),
                                   ("N", r -> Float64.(r[1]), r -> Float64.(r[1])),
                                   ("x2", r -> r[2], r -> r[2]))
                a = reduce(hcat, fa.(A)); c = reduce(hcat, fb.(B))
                z = (mean(a; dims=2) .- mean(c; dims=2)) ./ sqrt.(var(a; dims=2) ./ R .+ var(c; dims=2) ./ R .+ 1e-300)
                zmax = maximum(abs.(z[2:end]))
                zmax < 5 || println("  $name at (u=$u, p=$p, b=$b): max |z| = $zmax")
                @test zmax < 5
            end
        end
    end

    @testset "spreading_chunk sums the individual runs" begin
        rec = make_log_times(300; points_per_decade=10)
        Random.seed!(7)
        S = spreading_chunk(rec, 0.6, 0.03, 0.2, 25; block_len=6)
        Random.seed!(7)
        surv = zeros(Int, length(rec)); sn = zeros(length(rec)); sx = zeros(length(rec)); sr = zeros(length(rec))
        for _ in 1:25
            n, x2 = time_random_spreading(rec, 0.6, 0.03, 0.2; block_len=6)
            alive = n .> 0
            surv .+= alive; sn .+= n; sx .+= x2
            sr .+= ifelse.(alive, x2 ./ max.(n, 1), 0.0)
        end
        @test S.time == rec
        @test all(==(25), S.runs)
        @test S.surv == surv
        @test S.sum_n == sn
        @test S.sum_x2 ≈ sx
        @test S.sum_r2 ≈ sr
        @test all(S.sum_n2 .>= S.sum_n)
        @test S.surv[1] == 25 && S.sum_n[1] == 25                   # t = 0: every run has its seed
    end

    @testset "spreading file name" begin
        @test spreading_chunk_path("R", "time_rand_window_binary", 100000, 0.6, 0.03, 0.2, 1, 20, 500, 7) ==
              joinpath("R", "time_rand_window_binary", "spreading", "tmax100000", "epsilonu0p6", "epsilonl0p03", "pval0p2", "blocklen1",
                       "spreading_tmax100000_epsilonu0p6_epsilonl0p03_pval0p2_blocklen1_ppd20_runs500_chunk7.csv")
    end
end

# rough speed check (not a test): site updates per second in spreading runs
let rec = make_log_times(20000)
    time_random_spreading(rec, 0.6, 0.03, 0.2)                   # compile
    Random.seed!(1)
    t0 = time()
    for _ in 1:20
        time_random_spreading(rec, 0.6, 0.03, 0.2)
    end
    println("spreading, eps_u=0.6, p=0.2, t_max=20000: $(round((time() - t0) / 20, digits=3)) s per run")
end
