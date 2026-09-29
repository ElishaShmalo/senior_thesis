"""Tests for analysis/spreading_tools.py (spreading runs, added 2026-09-29).

    python -m pytest -q stavskya_mc/block_disorder/tests/test_spreading_tools.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import spreading_tools as st    # noqa: E402

MODEL = "time_rand_window_binary"


def test_spreading_path_matches_julia_literal():
    f = st.spreading_chunk_path("R", MODEL, 100000, 0.6, 0.03, 0.2, 1, 20, 500, 7)
    assert str(f) == ("R/time_rand_window_binary/spreading/tmax100000/epsilonu0p6/epsilonl0p03/pval0p2/blocklen1/"
                      "spreading_tmax100000_epsilonu0p6_epsilonl0p03_pval0p2_blocklen1_ppd20_runs500_chunk7.csv")


def _write_chunks(root, n_chunks, runs, times, Ps, Nmean, R2):
    rng = np.random.default_rng(0)
    for c in range(1, n_chunks + 1):
        surv = rng.binomial(runs, Ps)
        sum_n = Nmean * runs * (1 + 0.01 * rng.standard_normal(len(times)))
        df = pd.DataFrame(dict(time=times, runs=runs, surv=surv, sum_n=sum_n, sum_n2=sum_n ** 2 / runs,
                               sum_x2=R2 * sum_n, sum_r2=R2 * surv, sum_r2sq=R2 ** 2 * surv))
        f = st.spreading_chunk_path(root, MODEL, times[-1], 0.6, 0.03, 0.2, 1, 20, runs, c)
        f.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(f, index=False)


def test_load_spreading_aggregates(tmp_path):
    t = np.array([0, 1, 2, 5, 10, 20, 50, 100])
    Ps = 1 / (1 + 0.5 * np.log1p(t)); N = 1 + t / (1 + np.log1p(t)); R2 = 1 + t ** 2 / 12
    _write_chunks(tmp_path, 30, 400, t, Ps, N, R2)
    run = st.load_spreading(tmp_path, MODEL, 100, 0.6, 0.03, 0.2, 1, 20, 400, 30)
    assert run.n_chunks == 30 and run.n_runs == 12000
    obs = run.observables()
    assert np.allclose(obs["P_s"], Ps, atol=0.02)
    assert np.allclose(obs["N"], N, rtol=0.01)
    assert np.allclose(obs["R2"], R2)
    se = run.bootstrap_se(n_boot=100)
    assert np.all(se["P_s"][1:] > 0) and np.all(se["P_s"] < 0.02)
    with pytest.warns(UserWarning):
        st.load_spreading(tmp_path, MODEL, 100, 0.6, 0.03, 0.2, 1, 20, 400, 31)
    with pytest.raises(FileNotFoundError):
        st.load_spreading(tmp_path, MODEL, 100, 0.61, 0.0305, 0.2, 1, 20, 400, 5)


def test_local_slope_and_log_exponent():
    t = np.unique(np.round(np.logspace(0, 5, 101)).astype(int)).astype(float)
    T, s = st.local_slope(t, t ** -0.3)
    assert np.allclose(s, -0.3)
    g = (2 + 0.5 * np.log(t)) ** -3.6            # BVH form g = (a + b ln t)^(-y)
    y, tab = st.best_log_exponent(t, g, 1e2, 1e5)
    assert abs(y - 3.6) < 0.1
    y_pl, _ = st.best_log_exponent(t, t ** -0.3, 1e2, 1e5)
    assert y_pl > 6                              # a pure power law prefers y -> infinity
