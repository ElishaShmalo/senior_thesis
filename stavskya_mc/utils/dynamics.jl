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
