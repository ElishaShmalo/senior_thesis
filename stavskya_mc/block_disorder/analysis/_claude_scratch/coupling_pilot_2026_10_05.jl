# Pilot (2026-10-05, PROJECT_HISTORY.md §7.19): common random numbers across the eps grid with the production kernel.
#
# Each sample is run at three eps_bar values 0.00012 apart around the block_len 6 critical estimate, once with the SAME
# seed at every eps (coupled: same initial state, same disorder history, same site uniforms) and once with independent
# seeds (what the generators do now). Prints the response chi(t) = -d ln<A>/d eps_bar with bootstrap errors for both.
# Small sizes (L = 4000, t <= L, 300 samples, ~40 s on one core); nothing is written.
#
#   julia --project=. stavskya_mc/block_disorder/analysis/_claude_scratch/coupling_pilot_2026_10_05.jl
using Random, Statistics
const REPO = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
include(joinpath(REPO, "stavskya_mc", "utils", "general.jl"))
include(joinpath(REPO, "stavskya_mc", "utils", "dynamics.jl"))

L, b, p, S = 4000, 6, 0.2, 300
rec = make_log_times(L; points_per_decade=10)
f = p + (1 - p) / 20
dE = 0.00012
eus = [round(e / f, digits=6) for e in (0.1104 - dE, 0.1104, 0.1104 + dE)]
Ac = zeros(S, 3, length(rec)); Au = zeros(S, 3, length(rec))
for s in 1:S, (j, eu) in enumerate(eus)
    Random.seed!(1_000 + s)                      # coupled
    Ac[s, j, :] = 1 .- time_random_p_record_rho(make_rand_state(L, 0.5), rec, eu, round(eu / 20, digits=6), p; block_len=b)
    Random.seed!(1_000_000 + 10s + j)            # uncoupled
    Au[s, j, :] = 1 .- time_random_p_record_rho(make_rand_state(L, 0.5), rec, eu, round(eu / 20, digits=6), p; block_len=b)
end
println("coupled samples pathwise monotone in eps: ",
        all(Ac[:, 1, :] .>= Ac[:, 2, :]) && all(Ac[:, 2, :] .>= Ac[:, 3, :]))

chi(a1, a3) = (log.(vec(mean(a1, dims=1))) .- log.(vec(mean(a3, dims=1)))) ./ (2dE)
rng = Xoshiro(0)
boot(A, coupled) = std([begin
        i = rand(rng, 1:S, S); j = coupled ? i : rand(rng, 1:S, S)
        chi(A[i, 1, :], A[j, 3, :])
    end for _ in 1:400])
cc, cu = chi(Ac[:, 1, :], Ac[:, 3, :]), chi(Au[:, 1, :], Au[:, 3, :])
sc, su = boot(Ac, true), boot(Au, false)
println("     t |   chi coupled    |   chi uncoupled   | error ratio")
for tq in (10, 30, 100, 300, 1000, 4000)
    k = argmin(abs.(rec .- tq))
    println(lpad(rec[k], 6), " | ", lpad(round(cc[k], digits=1), 6), " ± ", rpad(round(sc[k], digits=2), 6),
            " | ", lpad(round(cu[k], digits=1), 7), " ± ", rpad(round(su[k], digits=1), 6), " | ", round(su[k] / sc[k], digits=1))
end
