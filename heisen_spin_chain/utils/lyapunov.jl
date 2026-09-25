# function for calculating S_diff
function calculate_spin_distence(S_A::Vector{Vector{Float64}}, S_B::Vector{Vector{Float64}})
    diff = S_A .- S_B
    dotted = map(dot, diff, diff)
    return sqrt(sum(dotted))
end

# Bring S_B closer to S_A with fixed vector norm ε across the entire chain
function push_back(S_A::Vector{Vector{Float64}}, S_B::Vector{Vector{Float64}}, epsilon_val::Float64)
    flat_A = reduce(vcat, S_A)
    flat_B = reduce(vcat, S_B)

    diff_vec = flat_B - flat_A
    diff_norm = norm(diff_vec)

    pushed_vec = flat_A + (epsilon_val / diff_norm) * diff_vec
    pushed_vec = map(normalize, [pushed_vec[3i-2:3i] for i in 1:length(S_A)]) # Reshape and renormalize

    return pushed_vec
end

# Calculates Lyapunov val given list of spin distences
function calculate_lambda(spin_dists, tau_val, epsilon_val, n_val)
    return sum(map(log, spin_dists ./ epsilon_val)) / (n_val * tau_val) 
end

function calculate_lambda_per_time(spin_dists, epsilon_val)
    return map(log, spin_dists ./ epsilon_val)
end

function calculate_lambda_from_lambda_per_time(lambda_per_time, tau_val, n_val)
    return sum(lambda_per_time) / (n_val * tau_val)
end

## I've chosen to keep this *WRONG* implementation of Benettin for nostalgic pourpouses
## Note that this is the wrong way to do Benettin (though it may seem legit).
## If you want to hear why this is incorrect, feel free to contact me (Elisha Shmalo)
# Bring S_B closer to S_A with factor epsilon_val
# function wrong_push_back(S_A::Vector{Vector{Float64}}, S_B::Vector{Vector{Float64}}, epsilon_val)
#     L = length(S_A)
#     spin_diff = S_B .- S_A
#     thing_to_add = (epsilon_val / sqrt(L)) .* (map(normalize, spin_diff))

#     return S_A .+ thing_to_add
# end


"""
    benettin_lambda_sdiff(L_J_vec, spin_chain_A, a_val, n_steps, tau, t_step, s_0, epsilon; record_every=1)

One sample of the Benettin run that get_good_data_severalL*.jl used to do inline (moved here
2026-09 so it can be tested). B is a copy of A with its middle spin kicked by
`make_random_spin(epsilon)`. Then, n_steps times:
- evolve A and B together (`random_evolve_spins_to_time`, same random J_x, J_y signs, control push);
- record ln(|d|/epsilon) and S_diff(A);
- push B back to distance epsilon from A.

Recording (the new record_every knob):
- `times[j]  = j*record_every*tau`, for j = 1, ..., div(n_steps, record_every);
- `lambda[j]` is the MEAN of ln(|d|/epsilon) over the record_every steps ending at times[j], so
  any time-window average of lambda is the same as with record_every = 1;
- `s_diff[j]` is S_diff(A) at times[j];
- if record_every does not divide n_steps, the last partial block is dropped.

As before, lambda is ln(|d|/epsilon) per step and is NOT divided by tau; the notebooks do that.
With record_every = 1 the output is bit-identical to the old inline loop (same random numbers,
same order). `L_J_vec` is copied, so the caller's vector (the global J_vec) is not modified.
"""
function benettin_lambda_sdiff(L_J_vec, spin_chain_A, a_val, n_steps::Integer, tau, t_step, s_0, epsilon;
                               record_every::Integer=1)
    record_every >= 1 || throw(ArgumentError("record_every must be >= 1, got $record_every"))
    J_work = copy(L_J_vec)
    A = spin_chain_A
    B = copy(A)
    mid = div(length(B), 2)
    B[mid] = normalize(B[mid] + make_random_spin(epsilon))

    n_rec = div(n_steps, record_every)
    times = [j * record_every * tau for j in 1:n_rec]
    lambda = zeros(n_rec)
    s_diff = zeros(n_rec)
    acc = 0.0
    for step in 1:(n_rec * record_every)
        evolved = random_evolve_spins_to_time(J_work, A, B, a_val, tau, t_step, s_0)
        A = evolved[1][end]
        B = evolved[2][end]
        acc += log(calculate_spin_distence(A, B) / epsilon)
        B = push_back(A, B, epsilon)
        if step % record_every == 0
            j = div(step, record_every)
            lambda[j] = acc / record_every
            s_diff[j] = weighted_spin_difference(A, s_0)
            acc = 0.0
        end
    end
    return times, lambda, s_diff
end
