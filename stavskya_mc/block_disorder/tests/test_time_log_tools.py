"""Unit tests for ../analysis/time_log_tools.py.

Run:  python -m pytest stavskya_mc/block_disorder/tests/test_time_log_tools.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import time_log_tools as tl  # noqa: E402


def log_grid(tmax=10**6, ppd=20):
    """Python mirror of make_log_times in stavskya_mc/utils/general.jl."""
    n = int(np.ceil(ppd * np.log10(tmax))) + 1
    ts = np.round(10 ** np.linspace(0, np.log10(tmax), max(n, 2))).astype(int)
    return np.array(list(dict.fromkeys([0, *ts, tmax])))


def test_float_str_matches_julia_convention():
    assert tl.float_str(0.333741) == "0p333741"
    assert tl.float_str(100) == "100p0"          # Julia: string(Float64(100)) = "100.0"
    assert tl.float_str(0.0215) == "0p0215"
    with pytest.warns(UserWarning):
        tl.float_str(1e-5)


def test_paths_match_naming_jl():
    p = tl.time_log_sample_path("/r", "time_rand_slidding_p", 35000, 0.43, 0.0215, 0.47882, 6, 100.0, 20, 17)
    assert str(p) == ("/r/time_rand_slidding_p/rho_per_time/IC1/L35000/epsilonu0p43/epsilonl0p0215/pval0p47882/"
                      "blocklen6/IC1_L35000_epsilonu0p43_epsilonl0p0215_pval0p47882_blocklen6_timepref100p0_ppd20_"
                      "time_log_sample17.csv")
    q = tl.block_rho_per_ep_path("/r", "time_rand_window_binary", 1000, 0.333, 0.01665, 0.8, 1, 1.45, 3000)
    assert str(q) == ("/r/time_rand_window_binary/rho_per_epsilon/IC1/L1000/"
                      "IC3000_L1000_epsilonu0p333_epsilonl0p01665_pval0p8_blocklen1_z1p45.csv")
    assert "time_log" in p.name


def test_running_delta_power_law_exact():
    t = log_grid()
    A = 0.7 * np.where(t > 0, t, 1.0) ** -0.159
    T, d = tl.running_delta(t, A, b=10)
    assert np.allclose(d, 0.159, atol=1e-10)
    assert T.min() >= 10


def test_running_delta_of_inverse_log_matches_formula():
    t = log_grid()
    A = 1.0 / np.log(np.where(t > 1, t, 2.0))
    T, d = tl.running_delta(t, A, b=10)
    m = T > 100
    expect = np.log10(np.log(T[m]) / np.log(T[m] / 10))
    # interpolation in ln t of ln(1/ln t) on a 20-per-decade grid: small error
    assert np.max(np.abs(d[m] - expect)) < 2e-3
    # the value quoted in the review for the Fig. 4 window
    assert abs(np.interp(np.log(2.2e4), np.log(T), d) - 0.114) < 3e-3


def test_running_alpha_recovers_log_exponent():
    t = log_grid()
    A = 3.0 * np.log(np.where(t > 1, t, 2.0)) ** -1.0
    T, a = tl.running_alpha(t, A, b=10)
    assert np.allclose(a[T > 1e3], 1.0, atol=5e-3)


def test_fits_recover_parameters():
    t = log_grid().astype(float)
    tt = t[t > 0]
    A_pow = 0.5 * tt ** -0.1
    A_bvh = 1.0 / (2.0 + 0.3 * np.log(tt))
    sem = 1e-3 * np.ones_like(tt)
    fp = tl.fit_power_law(tt, A_pow, sem * A_pow, 10, 1e6)
    assert abs(fp["delta"] - 0.1) < 1e-10 and fp["chi2r"] < 1e-12
    fb = tl.fit_bvh(tt, A_bvh, sem * A_bvh, 10, 1e6)
    assert abs(fb["a"] - 2.0) < 1e-9 and abs(fb["B"] - 0.3) < 1e-10
    t2 = tt[tt > 1]
    fl = tl.fit_log_power(t2, 2.0 * np.log(t2) ** -0.8, None, 10, 1e6)
    assert abs(fl["alpha"] - 0.8) < 1e-10


def test_inverse_activity_slope_constant_for_bvh():
    t = log_grid()
    tt = t[t > 0]
    for span in (1.5, 2.0, 10.0):
        T, s = tl.inverse_activity_slope(tt, 1.0 / (2.0 + 0.3 * np.log(tt)), span=span)
        assert np.allclose(s, 0.3, atol=1e-9) and T.min() >= span


def test_crossing_time_and_absorption():
    t = np.array([0, 1, 10, 100, 1000, 10000])
    Ac = np.array([1, 1, 0.5, 0.33, 0.25, 0.2])
    Ao = np.array([1, 1, 0.5, 0.30, 0.20, 0.1])     # 1/Ao / (1/Ac) = 1, 1, 1, 1.1, 1.25, 2
    assert tl.crossing_time(t, Ac, Ao, factor=1.2) == 1000
    run = tl.Run(times=t, rho=np.array([[0.5, 0.6, 1, 1, 1, 1], [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]]))
    at = tl.absorption_times(run)
    assert at[0] == 10 and np.isinf(at[1])
    _, std, surv = tl.log_activity_width(run)
    assert np.allclose(surv, [1, 1, 0.5, 0.5, 0.5, 0.5])
    assert np.isnan(std[3]) and np.isfinite(std[0])


def test_loader_roundtrip(tmp_path):
    t = log_grid(1000, 10)
    for k in (1, 2, 3):
        f = tl.time_log_sample_path(tmp_path, "m", 64, 0.43, 0.0215, 0.48, 3, 4.0, 10, k)
        f.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"time": t, "rho": np.linspace(0.5, 1, len(t)) ** k}).to_csv(f, index=False)
    with pytest.warns(UserWarning):             # sample 4 missing
        run = tl.load_time_log_run(tmp_path, "m", 64, 0.43, 0.0215, 0.48, 3, 4.0, 10, n_samples=4)
    assert run.n == 3 and np.array_equal(run.times, t)
    with pytest.raises(FileNotFoundError):
        tl.load_time_log_run(tmp_path, "m", 64, 0.43, 0.0215, 0.49, 3, 4.0, 10, n_samples=4)
