include("general.jl")
using Random

# =============================================================================
# Stavskaya automaton dynamics
#
#   eta_i(t+1) = 1                          with probability epsilon(t)
#              = eta_i(t) * eta_{i-1}(t)    otherwise            (periodic BC)
#
# eta = 1 is "healthy"; the all-healthy state is absorbing.
#
# 2026-09 refactor (see REFEREE_CONFLICT_REVIEW.md, section 7):
#   * All evolve functions share one type-stable, allocation-free kernel
#     (`stavskaya_step!`).  The old code allocated `rand(L)`, an Int
#     `choose_rule` array and `[upper_ep, lower_ep]` every step, and swapped an
#     Int array with a Float64 array, so the state changed type every step.
#   * The kernel draws the per-site uniforms with `rand!(u)` into a
#     preallocated buffer.  `rand(L)` is implemented as `rand!` on a fresh
#     Vector{Float64}, and the scalar draws happen in the same order as before,
#     so for a fixed seed the results are bit-identical to the old code
#     (checked by block_disorder/tests/test_dynamics.jl).
#   * Every temporally random evolve function takes a keyword `block_len`
#     (default 1 = old behaviour).  epsilon(t) is redrawn only at the start of
#     each block of `block_len` consecutive steps, i.e. at t = 1,
#     block_len+1, 2block_len+1, ...  This is the Delta t of Barghathi, Vojta
#     and Hoyos (arXiv:1603.08075).
#   * The returned state has the same element type as the input state (Int for
#     `make_rand_state`).  Before, it alternated between Int and Float64
#     depending on whether time_steps was even or odd.
# =============================================================================

"""
    stavskaya_step!(new_state, cur, u, epsilon)

One Stavskaya update from `cur` into `new_state`. `u` is a preallocated
`Vector{Float64}` of length `L` that gets filled with uniforms. Site `i` is set to
healthy (1) if `u[i] < epsilon`; otherwise it becomes `cur[i-1]*cur[i]`, with
site 1 looking at site `L`.
"""
@inline function stavskaya_step!(new_state::AbstractVector{T}, cur::AbstractVector{T},
                                 u::Vector{Float64}, epsilon::Real) where {T<:Real}
    L = length(cur)
    rand!(u)
    @inbounds new_state[1] = u[1] < epsilon ? one(T) : cur[L] * cur[1]
    @inbounds for i in 2:L
        new_state[i] = u[i] < epsilon ? one(T) : cur[i-1] * cur[i]
    end
    return new_state
end

# Shared driver: `draw_epsilon()` is called once at the start of every block.
function _evolve_with_epsilon_draw(state, time_steps::Integer, draw_epsilon::F;
                                   block_len::Integer=1) where {F}
    block_len >= 1 || throw(ArgumentError("block_len must be >= 1, got $block_len"))
    cur = copy(state)
    nxt = similar(cur)
    u = Vector{Float64}(undef, length(cur))
    epsilon = 0.0
    for t in 1:time_steps
        if (t - 1) % block_len == 0
            epsilon = draw_epsilon()
        end
        stavskaya_step!(nxt, cur, u, epsilon)
        cur, nxt = nxt, cur          # swap references, no copy
    end
    return cur
end

# --- clean (time-independent) model -----------------------------------------
function evolve_state(state, time_steps, epsilon)
    cur = copy(state)
    nxt = similar(cur)
    u = Vector{Float64}(undef, length(cur))
    for t in 1:time_steps
        stavskaya_step!(nxt, cur, u, epsilon)
        cur, nxt = nxt, cur
    end
    return cur
end

# --- temporally random variants ---------------------------------------------
# epsilon uniform in [epsilon' - delta, epsilon' + delta]
function time_random_delta_evolve_state(state, time_steps, epsilon_prime, delta; block_len::Integer=1)
    return _evolve_with_epsilon_draw(state, time_steps,
        () -> (epsilon_prime - delta) + 2*delta*rand(); block_len=block_len)
end

# epsilon = a X^n with X ~ U[0,1] and n chosen so that <epsilon> = epsilon'
function time_random_n_evolve_state(state, time_steps, epsilon_prime, a; block_len::Integer=1)
    n = (a/epsilon_prime) - 1
    return _evolve_with_epsilon_draw(state, time_steps,
        () -> a*rand()^n; block_len=block_len)
end

@inline gaussian(mean::T, std::T) where {T<:AbstractFloat} =
    mean + std * randn(T)

# epsilon ~ N(epsilon', sig)
function time_random_gauss_evolve_state(state, time_steps, epsilon_prime, sig; block_len::Integer=1)
    return _evolve_with_epsilon_draw(state, time_steps,
        () -> gaussian(epsilon_prime, sig); block_len=block_len)
end

# epsilon = 0 or 2 epsilon' with equal probability
function time_random_binary_evolve_state(state, time_steps, epsilon_prime; block_len::Integer=1)
    return _evolve_with_epsilon_draw(state, time_steps,
        () -> 2 * round(rand()) * epsilon_prime; block_len=block_len)
end

# epsilon = upper_ep with probability p_val, lower_ep otherwise
# (same random-number usage as the old `choice = Int(rand() > p_val)` code)
@inline draw_upper_lower(upper_ep, lower_ep, p_val) = rand() > p_val ? lower_ep : upper_ep

function time_random_p_evolve_state(state, time_steps, upper_ep, lower_ep, p_val; block_len::Integer=1)
    return _evolve_with_epsilon_draw(state, time_steps,
        () -> draw_upper_lower(upper_ep, lower_ep, p_val); block_len=block_len)
end

"""
    time_random_p_record_rho(state, record_times, upper_ep, lower_ep, p_val; block_len=1)

Evolve without interruption up to `maximum(record_times)` and return the healthy
fraction rho(t) at every time in `record_times`, which must be sorted, unique and
non-negative integers (t = 0 is the initial state).

Disorder blocks are aligned to absolute time: block k covers steps
(k-1)*block_len+1 : k*block_len. Calling `time_random_p_evolve_state` in chunks
would instead restart the block phase at the beginning of every chunk.

Once the chain is absorbed (rho == 1) it stays absorbed, so the remaining entries
are filled with 1.0 and the loop stops early.
"""
function time_random_p_record_rho(state, record_times::AbstractVector{<:Integer},
                                  upper_ep, lower_ep, p_val; block_len::Integer=1)
    block_len >= 1 || throw(ArgumentError("block_len must be >= 1, got $block_len"))
    issorted(record_times) && allunique(record_times) && (isempty(record_times) || first(record_times) >= 0) ||
        throw(ArgumentError("record_times must be sorted, unique and >= 0"))
    L = length(state)
    cur = copy(state)
    nxt = similar(cur)
    u = Vector{Float64}(undef, L)
    rho = fill(NaN, length(record_times))
    k = 1
    while k <= length(record_times) && record_times[k] == 0
        rho[k] = sum(cur) / L
        k += 1
    end
    epsilon = 0.0
    t_max = isempty(record_times) ? 0 : last(record_times)
    for t in 1:t_max
        if (t - 1) % block_len == 0
            epsilon = draw_upper_lower(upper_ep, lower_ep, p_val)
        end
        stavskaya_step!(nxt, cur, u, epsilon)
        cur, nxt = nxt, cur
        if record_times[k] == t
            r = sum(cur) / L
            rho[k] = r
            k += 1
            if r == 1                     # absorbed: nothing changes any more
                rho[k:end] .= 1.0
                break
            end
            k > length(record_times) && break
        end
    end
    return rho
end

# =============================================================================
# Spreading (single-seed) runs, added 2026-09-29 for the BVH reproduction
# (block_disorder/generators/get_bvh_spreading.jl).
# =============================================================================

"""
    time_random_spreading(record_times, upper_ep, lower_ep, p_val; block_len=1) -> (n, x2)

One spreading run on the infinite chain: at t = 0 a single site x0 is active (0) and
all others are healthy (1). Same dynamics and same disorder as
`time_random_p_record_rho`: epsilon = `draw_upper_lower(upper_ep, lower_ep, p_val)`,
redrawn at the start of every block of `block_len` steps.

Returns, at every time in `record_times` (sorted, unique, >= 0):
  * `n[k]`  = number of active sites N(t);
  * `x2[k]` = sum over active sites of (i - x0 - t/2)^2.

Site i only looks at i-1 and i, so active sites stay inside the light cone
x0 <= i <= x0 + t, whose axis is x0 + t/2 (Stavskaya's one-sided coordinates of the
directed-percolation lattice). A window of t_max + 3 sites therefore represents the
infinite chain exactly, and only the sites from the leftmost active site to one past
the rightmost one are updated. Once no site is active the run is over (n = 0 from
then on).
"""
function time_random_spreading(record_times::AbstractVector{<:Integer}, upper_ep, lower_ep, p_val;
                               block_len::Integer=1)
    block_len >= 1 || throw(ArgumentError("block_len must be >= 1, got $block_len"))
    issorted(record_times) && allunique(record_times) && (isempty(record_times) || first(record_times) >= 0) ||
        throw(ArgumentError("record_times must be sorted, unique and >= 0"))
    nrec = length(record_times)
    n_out = zeros(Int, nrec)
    x2_out = zeros(Float64, nrec)
    nrec == 0 && return n_out, x2_out
    t_max = last(record_times)
    x0 = 2                                  # cur[x0 - 1] = cur[1] stays healthy forever
    cur = ones(Int8, t_max + 3)
    nxt = ones(Int8, t_max + 3)
    cur[x0] = 0
    lo, hi = x0, x0                         # active sites of cur lie in lo:hi
    plo, phi = 1, 0                         # sites of nxt that may still hold 0s (none yet)
    k = 1
    while k <= nrec && record_times[k] == 0
        n_out[k] = 1                        # x2 = 0 at t = 0
        k += 1
    end
    epsilon = 0.0
    for t in 1:t_max
        if (t - 1) % block_len == 0
            epsilon = draw_upper_lower(upper_ep, lower_ep, p_val)
        end
        @inbounds for i in plo:phi          # clear the state from two steps ago
            nxt[i] = one(Int8)
        end
        c = x0 + t / 2
        n = 0
        x2 = 0.0
        nlo, nhi = 0, -1
        @inbounds for i in lo:(hi + 1)
            v = rand() < epsilon ? one(Int8) : cur[i-1] * cur[i]
            nxt[i] = v
            if v == 0
                n += 1
                d = i - c
                x2 += d * d
                nlo == 0 && (nlo = i)
                nhi = i
            end
        end
        cur, nxt = nxt, cur
        plo, phi = lo, hi                   # nxt now holds time t-1, active only in lo:hi
        if n == 0
            break                           # died: n_out, x2_out stay 0 from here on
        end
        lo, hi = nlo, nhi
        if k <= nrec && record_times[k] == t
            n_out[k] = n
            x2_out[k] = x2
            k += 1
            k > nrec && break
        end
    end
    return n_out, x2_out
end

"""
    spreading_chunk(record_times, upper_ep, lower_ep, p_val, n_runs; block_len=1)

`n_runs` independent spreading runs, each with its own disorder sequence (one run per
disorder realization, as in BVH). Returns a named tuple of sums at every record time,
ready for a DataFrame:
  * `runs`     = n_runs;
  * `surv`     = number of runs with N(t) > 0;
  * `sum_n`, `sum_n2` = sums of N and N^2 over all runs (dead runs count 0);
  * `sum_x2`   = sum over all runs of sum_i (i - x0 - t/2)^2;
  * `sum_r2`, `sum_r2sq` = sums over surviving runs of R^2 = x2/N and of (R^2)^2.
Then P_s = surv/runs, <N> = sum_n/runs, R^2 = sum_x2/sum_n (or sum_r2/surv).
"""
function spreading_chunk(record_times, upper_ep, lower_ep, p_val, n_runs::Integer; block_len::Integer=1)
    m = length(record_times)
    surv = zeros(Int, m)
    sum_n = zeros(Float64, m); sum_n2 = zeros(Float64, m); sum_x2 = zeros(Float64, m)
    sum_r2 = zeros(Float64, m); sum_r2sq = zeros(Float64, m)
    for _ in 1:n_runs
        n, x2 = time_random_spreading(record_times, upper_ep, lower_ep, p_val; block_len=block_len)
        @inbounds for j in 1:m
            n[j] > 0 || continue
            nj = Float64(n[j])
            r2 = x2[j] / nj
            surv[j] += 1
            sum_n[j] += nj
            sum_n2[j] += nj * nj
            sum_x2[j] += x2[j]
            sum_r2[j] += r2
            sum_r2sq[j] += r2 * r2
        end
    end
    return (time=collect(record_times), runs=fill(Int(n_runs), m), surv=surv, sum_n=sum_n, sum_n2=sum_n2,
            sum_x2=sum_x2, sum_r2=sum_r2, sum_r2sq=sum_r2sq)
end
