"""Loaders and estimators for the spreading runs of get_upper_lower_binary_spreading*.jl (added 2026-09-29).

Each chunk file holds sums over `runs` independent runs (one disorder realization per run) at
every output time: surv, sum_n, sum_n2, sum_x2, sum_r2, sum_r2sq (see spreading_chunk in
utils/dynamics.jl). Observables, as in BVH (arXiv:1603.08075, Sec. IV):
    P_s(t) = surv / runs                  survival probability
    N(t)   = sum_n / runs                 mean number of active sites (dead runs count 0)
    R2(t)  = sum_x2 / sum_n               mean-square distance from the light-cone axis
Error bars come from a bootstrap over chunks, so they include the disorder fluctuations.

BVH infinite-noise predictions (their Eqs. 23-25, z = 1 with log corrections):
    P_s ~ 1/ln t,   N_s/t ~ (ln t)^(-y_N),   R/t ~ (ln t)^(-y_R)
so 1/P_s, (N/t)^(-1/y_N) and (R/t)^(-1/y_R) are linear in ln t. BVH found y_N = 3.6,
y_R = 1.7 in 1D for the contact process. Clean DP instead gives power laws:
P_s ~ t^-0.1595, N ~ t^0.3137, R2 ~ t^(2/1.5807).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from time_log_tools import float_str

DP = dict(delta=0.159464, theta=0.313686, z=1.580745)   # clean 1+1D DP (Jensen 1999)
COLUMNS = ["time", "runs", "surv", "sum_n", "sum_n2", "sum_x2", "sum_r2", "sum_r2sq"]


def spreading_chunk_path(root, model_dir, t_max, upper_ep, lower_ep, p_val, block_len, ppd,
                         runs_per_chunk, chunk) -> Path:
    """Mirror of spreading_chunk_path in ../generators/naming.jl."""
    u, l, p = map(float_str, (upper_ep, lower_ep, p_val))
    d = (Path(root) / model_dir / "spreading" / f"tmax{t_max}" / f"epsilonu{u}" / f"epsilonl{l}"
         / f"pval{p}" / f"blocklen{block_len}")
    name = (f"spreading_tmax{t_max}_epsilonu{u}_epsilonl{l}_pval{p}_blocklen{block_len}"
            f"_ppd{ppd}_runs{runs_per_chunk}_chunk{chunk}.csv")
    return d / name


@dataclass
class SpreadRun:
    """All chunks of one parameter set; every array has shape (n_chunks, n_times)."""
    times: np.ndarray
    runs: np.ndarray
    surv: np.ndarray
    sum_n: np.ndarray
    sum_x2: np.ndarray
    sum_r2: np.ndarray

    @property
    def n_chunks(self) -> int:
        return self.runs.shape[0]

    @property
    def n_runs(self) -> int:
        return int(self.runs[:, 0].sum())

    @staticmethod
    def _obs(runs, surv, sum_n, sum_x2):
        R, S, N, X = runs.sum(0), surv.sum(0), sum_n.sum(0), sum_x2.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return dict(P_s=S / R, N=N / R, R2=np.where(N > 0, X / N, np.nan), N_surv=np.where(S > 0, N / S, np.nan))

    def observables(self) -> dict:
        return self._obs(self.runs, self.surv, self.sum_n, self.sum_x2)

    def bootstrap_se(self, n_boot=200, seed=0) -> dict:
        """Standard errors of the observables from resampling chunks with replacement."""
        if self.n_chunks < 2:
            return {k: np.full(self.times.shape, np.nan) for k in self.observables()}
        rng = np.random.default_rng(seed)
        draws = {k: [] for k in self.observables()}
        for _ in range(n_boot):
            i = rng.integers(0, self.n_chunks, self.n_chunks)
            for k, v in self._obs(self.runs[i], self.surv[i], self.sum_n[i], self.sum_x2[i]).items():
                draws[k].append(v)
        return {k: np.nanstd(np.array(v), axis=0, ddof=1) for k, v in draws.items()}


def load_spreading(root, model_dir, t_max, upper_ep, lower_ep, p_val, block_len, ppd, runs_per_chunk,
                   n_chunks, chunk_offset=0, min_chunks=1) -> SpreadRun:
    """Load chunks chunk_offset+1 ... chunk_offset+n_chunks, skipping missing ones (with a warning)."""
    times, parts, missing = None, [], 0
    for c in range(chunk_offset + 1, chunk_offset + n_chunks + 1):
        f = spreading_chunk_path(root, model_dir, t_max, upper_ep, lower_ep, p_val, block_len, ppd, runs_per_chunk, c)
        if not f.exists():
            missing += 1
            continue
        df = pd.read_csv(f)
        if list(df.columns) != COLUMNS:
            raise ValueError(f"{f}: columns {list(df.columns)}, expected {COLUMNS}")
        t = df["time"].to_numpy()
        if times is None:
            times = t
        elif not np.array_equal(t, times):
            raise ValueError(f"time grid of {f} differs from the first chunk's")
        parts.append(df)
    if len(parts) < min_chunks:
        raise FileNotFoundError(
            f"only {len(parts)} chunks found for eps_u={upper_ep}, p={p_val}, block_len={block_len} (example path: "
            f"{spreading_chunk_path(root, model_dir, t_max, upper_ep, lower_ep, p_val, block_len, ppd, runs_per_chunk, chunk_offset + 1)})")
    if missing:
        warnings.warn(f"{missing} of {n_chunks} chunk files missing for eps_u={upper_ep}, p={p_val}, block_len={block_len}")
    arr = {k: np.vstack([d[k].to_numpy(dtype=float) for d in parts]) for k in ("runs", "surv", "sum_n", "sum_x2", "sum_r2")}
    return SpreadRun(times=np.asarray(times), **arr)


def local_slope(t, y, span=2.0):
    """d ln y / d ln t measured over a factor `span` in time (ln y interpolated in ln t).
    Returns (T, slope) for the grid points T with T/span inside the data range and y > 0."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    m = (t > 0) & (y > 0) & np.isfinite(y)
    t, y = t[m], y[m]
    T = t[t / span >= t[0]]
    prev = np.interp(np.log(T / span), np.log(t), np.log(y))
    now = np.log(y[np.searchsorted(t, T)])
    return T, (now - prev) / np.log(span)


def straightness(t, f, tmin, tmax):
    """Relative rms deviation of f from a straight line in ln t over [tmin, tmax]
    (0 = perfectly linear). Also returns the slope."""
    t = np.asarray(t, float); f = np.asarray(f, float)
    m = (t >= tmin) & (t <= tmax) & np.isfinite(f)
    if m.sum() < 3:
        return np.nan, np.nan
    x, y = np.log(t[m]), f[m]
    c = np.polyfit(x, y, 1)
    span = np.ptp(np.polyval(c, x))
    return float(np.sqrt(np.mean((y - np.polyval(c, x)) ** 2)) / span) if span > 0 else np.nan, float(c[0])


def best_log_exponent(t, g, tmin, tmax, y_grid=np.linspace(0.3, 8.0, 155)):
    """For g(t) ~ (a + b ln t)^(-y) (e.g. g = N/t or R/t), find the y that makes g^(-1/y)
    straightest against ln t over [tmin, tmax]. Returns (y_best, table of (y, deviation))."""
    g = np.asarray(g, float)
    rows = []
    for y in y_grid:
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where(g > 0, g ** (-1.0 / y), np.nan)
        dev, slope = straightness(t, f, tmin, tmax)
        rows.append((y, dev, slope))
    tab = pd.DataFrame(rows, columns=["y", "deviation", "slope"])
    good = tab.dropna()
    y_best = float(good.loc[good.deviation.idxmin(), "y"]) if len(good) else np.nan
    return y_best, tab
