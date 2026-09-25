"""Loaders and estimators for the block_disorder / log-time Stavskaya data.

File layout: the paths built here must match ../generators/naming.jl exactly.

Estimators: every estimator works on an arbitrary increasing time grid (log spaced,
possibly starting at t = 0), so none of them assumes a uniform ``time_step``. The
time-ratio estimators (``running_delta``, ``running_alpha``) interpolate
ln(activity) linearly in ln(t) to evaluate A(t/b).

Conventions:
    rho      = healthy fraction, as written by the generators
    activity = 1 - rho  (the order parameter; the absorbing state has activity 0)

Written 2026-09; see REFEREE_CONFLICT_REVIEW.md, section 7.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# naming (mirror of naming.jl)
# --------------------------------------------------------------------------

def float_str(x) -> str:
    """0.333741 -> '0p333741'; always goes through float, so 100 -> '100p0'.

    Matches Julia's replace(string(Float64(x)), "." => "p") for values between
    1e-4 and 1e5.
    """
    x = float(x)
    if x != 0 and not (1e-4 <= abs(x) < 1e5):
        warnings.warn(f"float_str({x}): Julia and Python format this number differently")
    return str(x).replace(".", "p")


def time_log_sample_path(root, model_dir, L, upper_ep, lower_ep, p_val, block_len,
                         time_prefact, ppd, sample) -> Path:
    u, l, p, tp = map(float_str, (upper_ep, lower_ep, p_val, time_prefact))
    d = (Path(root) / model_dir / "rho_per_time" / "IC1" / f"L{L}" / f"epsilonu{u}" / f"epsilonl{l}"
         / f"pval{p}" / f"blocklen{block_len}")
    name = (f"IC1_L{L}_epsilonu{u}_epsilonl{l}_pval{p}_blocklen{block_len}"
            f"_timepref{tp}_ppd{ppd}_time_log_sample{sample}.csv")
    return d / name


def block_rho_per_ep_path(root, model_dir, L, upper_ep, lower_ep, p_val, block_len, z_val, n_ic) -> Path:
    u, l, p, z = map(float_str, (upper_ep, lower_ep, p_val, z_val))
    return (Path(root) / model_dir / "rho_per_epsilon" / "IC1" / f"L{L}"
            / f"IC{n_ic}_L{L}_epsilonu{u}_epsilonl{l}_pval{p}_blocklen{block_len}_z{z}.csv")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

@dataclass
class Run:
    """All samples of one parameter set. ``rho`` has shape (n_samples, n_times)."""
    times: np.ndarray
    rho: np.ndarray

    @property
    def activity(self) -> np.ndarray:
        return 1.0 - self.rho

    @property
    def n(self) -> int:
        return self.rho.shape[0]

    @property
    def mean(self) -> np.ndarray:
        return self.activity.mean(axis=0)

    @property
    def sem(self) -> np.ndarray:
        if self.n < 2:
            return np.full(self.times.shape, np.nan)
        return self.activity.std(axis=0, ddof=1) / np.sqrt(self.n)

    @property
    def surviving_fraction(self) -> np.ndarray:
        """Fraction of samples not yet absorbed (activity > 0) at each time."""
        return (self.activity > 0).mean(axis=0)


def load_time_log_run(root, model_dir, L, upper_ep, lower_ep, p_val, block_len, time_prefact, ppd,
                      n_samples, offset=0, min_samples=1) -> Run:
    """Load samples offset+1 ... offset+n_samples, skipping missing files (with a warning)."""
    times, rows, missing = None, [], 0
    for k in range(offset + 1, offset + n_samples + 1):
        f = time_log_sample_path(root, model_dir, L, upper_ep, lower_ep, p_val, block_len, time_prefact, ppd, k)
        if not f.exists():
            missing += 1
            continue
        df = pd.read_csv(f)
        t = df["time"].to_numpy()
        if times is None:
            times = t
        elif not np.array_equal(times, t):
            raise ValueError(f"time grid of {f} differs from the first sample's")
        rows.append(df["rho"].to_numpy(dtype=float))
    if len(rows) < min_samples:
        raise FileNotFoundError(
            f"only {len(rows)} samples found for L={L}, eps_u={upper_ep}, eps_l={lower_ep}, p={p_val}, "
            f"block_len={block_len} under {root}/{model_dir} (example path: "
            f"{time_log_sample_path(root, model_dir, L, upper_ep, lower_ep, p_val, block_len, time_prefact, ppd, offset + 1)})")
    if missing:
        warnings.warn(f"{missing} of {n_samples} sample files missing for L={L}, p={p_val}, eps_u={upper_ep}")
    rho = np.vstack(rows)
    if np.isnan(rho).any():
        raise ValueError("NaN in rho: a sample was not evolved up to its last output time")
    return Run(times=np.asarray(times), rho=rho)


# --------------------------------------------------------------------------
# estimators on arbitrary (log) time grids
# --------------------------------------------------------------------------

def _positive(t, A):
    t = np.asarray(t, float)
    A = np.asarray(A, float)
    m = (t > 0) & (A > 0) & np.isfinite(A)
    return t[m], A[m]


def _lnA_at(t, A, tq):
    """ln A interpolated linearly in ln t at the query times tq (inside the data range)."""
    return np.interp(np.log(tq), np.log(t), np.log(A))


def running_delta(t, A, b=10.0):
    """Power-law running exponent  delta_eff(T) = log_b[A(T/b) / A(T)].

    Constant (= delta) for a power law, and drifts to 0 like log_b[ln T / ln(T/b)]
    for BVH's A ~ 1/ln t. Returns (T, delta_eff) for the grid points with T/b
    inside the data range.
    """
    t, A = _positive(t, A)
    T = t[t / b >= t[0]]
    d = (_lnA_at(t, A, T / b) - np.log(A[np.searchsorted(t, T)])) / np.log(b)
    return T, d


def running_alpha(t, A, b=10.0, min_ln=1.0):
    """Log-scaling running exponent  alpha_eff(T) = ln[A(T/b)/A(T)] / ln[ln T / ln(T/b)].

    Constant (= alpha) for A ~ (ln t)^(-alpha); BVH predict alpha -> 1 (delta_bar = 1).
    Only uses T with ln(T/b) >= min_ln so the denominator stays well defined.
    """
    t, A = _positive(t, A)
    T = t[(t / b >= t[0]) & (np.log(t / b) >= min_ln)]
    num = _lnA_at(t, A, T / b) - np.log(A[np.searchsorted(t, T)])
    den = np.log(np.log(T) / np.log(T / b))
    return T, num / den


def inverse_activity_slope(t, A, span=2.0):
    """Local slope d(1/A)/d ln t, measured over a factor `span` in time:
    [1/A(T) - 1/A(T/span)] / ln(span), with 1/A interpolated linearly in ln t.

    BVH: 1/A = a + B ln t at criticality, so the slope is the constant B. It grows on
    the inactive side and falls toward 0 on the active side. A span > the grid spacing
    averages out point-to-point noise.
    """
    t, A = _positive(t, A)
    T = t[t / span >= t[0]]
    inv_prev = np.interp(np.log(T / span), np.log(t), 1.0 / A)
    inv_now = 1.0 / A[np.searchsorted(t, T)]
    return T, (inv_now - inv_prev) / np.log(span)


def _window(t, A, sem, tmin, tmax):
    t = np.asarray(t, float); A = np.asarray(A, float)
    sem = np.asarray(sem, float) if sem is not None else np.full_like(A, np.nan)
    m = (t >= tmin) & (t <= tmax) & (A > 0)
    t, A, sem = t[m], A[m], sem[m]
    if not np.all(np.isfinite(sem)) or np.any(sem <= 0):
        sem = np.full_like(A, np.nan)
    return t, A, sem


def _wls(x, y, sigma):
    """Weighted least squares for y = c0 + c1 x; unweighted if sigma is all-NaN."""
    X = np.c_[np.ones_like(x), x]
    w = np.ones_like(y) if np.all(np.isnan(sigma)) else 1.0 / sigma
    coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    r = (y - X @ coef) * w
    dof = max(len(y) - 2, 1)
    chi2r = float(np.sum(r ** 2) / dof) if not np.all(np.isnan(sigma)) else np.nan
    return coef, chi2r


def fit_power_law(t, A, sem, tmin, tmax):
    """ln A = c - delta ln t over [tmin, tmax]."""
    t, A, sem = _window(t, A, sem, tmin, tmax)
    coef, chi2r = _wls(np.log(t), np.log(A), sem / A)
    return {"delta": -coef[1], "amp": np.exp(coef[0]), "chi2r": chi2r, "n": len(t)}


def fit_bvh(t, A, sem, tmin, tmax):
    """BVH infinite-noise form 1/A = a + B ln t over [tmin, tmax].

    Linear in ln t, so there is no extrapolation. a/B = ln t0 absorbs the
    non-universal microscopic time scale.
    """
    t, A, sem = _window(t, A, sem, tmin, tmax)
    coef, chi2r = _wls(np.log(t), 1.0 / A, sem / A ** 2)
    return {"a": coef[0], "B": coef[1], "ln_t0": coef[0] / coef[1] if coef[1] else np.nan,
            "chi2r": chi2r, "n": len(t)}


def fit_log_power(t, A, sem, tmin, tmax):
    """ln A = c - alpha ln ln t over [tmin, tmax] (the pure (ln t)^-alpha form)."""
    t, A, sem = _window(t, A, sem, tmin, tmax)
    t, A, sem = t[t > np.e], A[t > np.e], sem[t > np.e]
    coef, chi2r = _wls(np.log(np.log(t)), np.log(A), sem / A)
    return {"alpha": -coef[1], "amp": np.exp(coef[0]), "chi2r": chi2r, "n": len(t)}


def log_activity_width(run: Run):
    """Spread over disorder realizations of x = -ln(activity) at each time.

    Only samples that are still active count. BVH: std(x) grows without bound,
    proportional to ln t, at criticality. At a conventional (finite-disorder)
    critical point it saturates. Returns (t, std_x, surviving_fraction).
    """
    act = run.activity
    std = np.full(run.times.shape, np.nan)
    for j in range(act.shape[1]):
        a = act[:, j]
        a = a[a > 0]
        if len(a) >= 2:
            std[j] = np.std(-np.log(a), ddof=1)
    return run.times, std, run.surviving_fraction


def crossing_time(t, A_crit, A_off, factor=1.1):
    """First time at which 1/A_off exceeds factor * (1/A_crit) (off-critical curve on the
    inactive side leaving the critical line). BVH (their Fig. 7) predict
    ln t_x ~ r^(-1/2); a power-law critical point gives ln t_x ~ nu_t ln(1/r)."""
    t = np.asarray(t, float)
    m = (t > 0) & (np.asarray(A_crit) > 0)
    t, Ac, Ao = t[m], np.asarray(A_crit, float)[m], np.asarray(A_off, float)[m]
    with np.errstate(divide="ignore"):
        exceed = (1.0 / Ao) > factor / Ac
    idx = np.argmax(exceed) if exceed.any() else None
    return float(t[idx]) if idx is not None else np.nan


def absorption_times(run: Run):
    """Per sample, the first output time with activity == 0 (np.inf if never absorbed).
    The resolution is that of the log grid."""
    absorbed = run.activity <= 0
    first = np.where(absorbed.any(axis=1), run.times[np.argmax(absorbed, axis=1)], np.inf)
    return first
