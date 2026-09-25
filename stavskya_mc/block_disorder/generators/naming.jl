# File and directory naming for the block_disorder generators.
#
# The Python loaders in ../analysis/time_log_tools.py build the same paths. If you
# change anything here, change it there too.
#
# Numbers are written with float_str, e.g. 0.333741 -> "0p333741". It always goes
# through Float64, so 100 and 100.0 give the same name "100p0". Julia's string(Float64)
# and Python's str(float) only agree for values between 1e-4 and 1e5: below 1e-4
# the two languages write the exponent differently, and from 1e6 up Julia switches to
# "1.0e6". Keep every parameter inside that range, which already holds for all of ours.

float_str(f) = replace(string(Float64(f)), "." => "p")

"""
Per-sample CSV for a log-time run. Columns: `time` (the log-spaced output times,
starting at 0) and `rho` (the healthy fraction; the activity is 1 - rho).

    <root>/<model_dir>/rho_per_time/IC1/L<L>/epsilonu<u>/epsilonl<l>/pval<p>/blocklen<b>/
        IC1_L<L>_epsilonu<u>_epsilonl<l>_pval<p>_blocklen<b>_timepref<tp>_ppd<ppd>_time_log_sample<k>.csv
"""
function time_log_sample_path(root, model_dir, L, upper_ep, lower_ep, p_val, block_len,
                              time_prefact, ppd, sample)
    u, l, p, tp = float_str(upper_ep), float_str(lower_ep), float_str(p_val), float_str(time_prefact)
    dir = joinpath(root, model_dir, "rho_per_time", "IC1", "L$(L)", "epsilonu$(u)", "epsilonl$(l)",
                   "pval$(p)", "blocklen$(block_len)")
    name = "IC1_L$(L)_epsilonu$(u)_epsilonl$(l)_pval$(p)_blocklen$(block_len)_timepref$(tp)_ppd$(ppd)_time_log_sample$(sample).csv"
    return joinpath(dir, name)
end

"""
One CSV per parameter set for a rho-per-epsilon (FSS at t = L^z) run with block
disorder. Columns: `sample`, `rho`.

    <root>/<model_dir>/rho_per_epsilon/IC1/L<L>/
        IC<n>_L<L>_epsilonu<u>_epsilonl<l>_pval<p>_blocklen<b>_z<z>.csv
"""
function block_rho_per_ep_path(root, model_dir, L, upper_ep, lower_ep, p_val, block_len, z_val, n_ic)
    u, l, p, z = float_str(upper_ep), float_str(lower_ep), float_str(p_val), float_str(z_val)
    return joinpath(root, model_dir, "rho_per_epsilon", "IC1", "L$(L)",
                    "IC$(n_ic)_L$(L)_epsilonu$(u)_epsilonl$(l)_pval$(p)_blocklen$(block_len)_z$(z).csv")
end
