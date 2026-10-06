"""Checks behind the 2026-10-05 assessment of the Stavskaya next steps (PROJECT_HISTORY.md §7.19).

Run from stavskya_mc/block_disorder/analysis:
    python3 _claude_scratch/review_checks_2026_10_05.py            # all sections (~1 min)
    python3 _claude_scratch/review_checks_2026_10_05.py 4 5        # only sections 4 and 5

Inputs (nothing is written anywhere):
  * spreading chunks: ../../data/spreading/... (the three bvh/clean spreading copies);
  * decay means/SEMs: _claude_scratch/summary.json (aggregated from the external-drive CSVs in
    the 2026-10-04 session). Without it, sections 1, 2 (decay rows), 5 and 7 (decay rows) are skipped.

Sections
  1  duality: <A(t)> from a half-filled start vs (1 - eps_bar) <P_s(t-1)>  (exact for block_len 1)
  2  single-observable straightness with eps_c free (Box-Cox kappa: 0 power law, 1 BVH 1/A linear in ln t)
  3  how much the interpolated block_len 1 verdicts depend on interpolation scheme and window
  4  joint test: zero-curvature eps of P_s, N, R (one eps for all at a conventional critical point)
  5  eps-response exponent 1/nu_eff(t) = d ln chi / d ln t, chi = -d ln A / d eps_bar
  6  early/late local spreading exponents for every eps of each grid
  7  per-decade delta from decade end points, with errors
  8  log-correction exponents y_R, y_N where P_s is BVH-straight vs where it is a power law
  9  the joint test of section 4 in short windows a spin-chain run can reach (added 2026-10-05, H §7.20)
 10  the joint test without interpolation: exponent drift between decades at every measured eps (H §7.21)
 11  response exponent of spreading P_s at a point, quadratic in eps through three grid values (H §7.21)
The coupled-vs-uncoupled pilot is the separate coupling_pilot_2026_10_05.jl.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AN = HERE.parent
sys.path.insert(0, str(AN))
import spreading_tools as st  # noqa: E402
import time_log_tools as tl  # noqa: E402

SPREAD_ROOT = AN / ".." / ".." / "data" / "spreading"
SUMMARY = HERE / "summary.json"
MODELS = {"clean": (0.2945, 0.0001, 1.0, 1), "b1": (0.144, 0.0012, 0.2, 1), "b6": (0.1104, 0.0012, 0.2, 6)}
KAPPAS = np.round(np.arange(-0.5, 2.51, 0.05), 3)
LAMS = np.round(np.arange(-0.4, 1.21, 0.02), 3)


# ---------------------------------------------------------------- loading
def param_sets(avg_c, rate, p, steps, lower_div=20):
    f = p + (1 - p) / lower_div
    out = []
    for i in steps:
        u = round(avg_c / f + i * (rate / f), 6)
        l = round(u / lower_div, 6)
        out.append((round(p * u + (1 - p) * l, 6), u, l))
    return out


def load_all():
    spread, decay = {}, {}
    summ = json.load(open(SUMMARY)) if SUMMARY.exists() else None
    for name, (c, r, p, b) in MODELS.items():
        for eb, u, l in param_sets(c, r, p, range(-2, 3)):
            run = st.load_spreading(SPREAD_ROOT, "time_rand_window_binary", 100000, u, l, p, b, 20, 500, 200)
            o, se = run.observables(), run.bootstrap_se(n_boot=300)
            spread[(name, eb)] = dict(t=run.times.astype(float), Ps=o["P_s"], N=o["N"], R2=o["R2"], Ps_se=se["P_s"],
                                      N_se=se["N"], R2_se=se["R2"],
                                      chunks=dict(runs=run.runs, surv=run.surv, sum_n=run.sum_n, sum_x2=run.sum_x2))
        if summ is None:
            continue
        for eb, u, l in param_sets(c, r, p, range(-3, 4)):
            key = f"epsilonu{tl.float_str(u)}_epsilonl{tl.float_str(l)}_pval{tl.float_str(p)}_blocklen{b}"
            s = summ[key]
            decay[(name, eb)] = {k: np.asarray(s[k], float) for k in ("t", "mean", "sem", "surv", "sd_lnA")}
    return spread, decay


# ---------------------------------------------------------------- helpers
def bc_chi2(t, A, sA, tmin, tmax, kappa):
    """chi2r of a straight-line fit of the Box-Cox transform (A^-kappa - 1)/kappa against ln t."""
    m = (t >= tmin) & (t <= tmax) & (A > 0) & (sA > 0) & np.isfinite(A)
    x, a, s = np.log(t[m]), A[m], sA[m]
    z, sz = (-np.log(a), s / a) if abs(kappa) < 1e-9 else ((a ** -kappa - 1) / kappa, a ** (-kappa - 1) * s)
    X = np.c_[np.ones_like(x), x] / sz[:, None]
    coef, *_ = np.linalg.lstsq(X, z / sz, rcond=None)
    r = z / sz - X @ coef
    return float(r @ r / (len(z) - 2))


def scan(t, A, sA, tmin, tmax, grid=KAPPAS):
    c = np.array([bc_chi2(t, A, sA, tmin, tmax, k) for k in grid])
    return grid[np.argmin(c)], c.min(), c[np.argmin(abs(grid))], c[np.argmin(abs(grid - 1))]


def curve(spread, decay, src, key):
    if src == "decay":
        d = decay[key]
        return d["t"], d["mean"], d["sem"]
    s = spread[key]
    return s["t"], s["Ps"], s["Ps_se"]


def obs_from_chunks(ch):
    R, S, N, X = ch["runs"].sum(0), ch["surv"].sum(0), ch["sum_n"].sum(0), ch["sum_x2"].sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"P_s": S / R, "N": N / R, "R": np.sqrt(X / N)}


def curvature(t, y, win):
    m = (t >= win[0]) & (t <= win[1]) & (y > 0)
    u = np.log(t[m])
    return np.polyfit(u - u.mean(), np.log(y[m]), 2)[0]


def zero_cross(eps, q):
    i = np.where(np.sign(q[:-1]) * np.sign(q[1:]) < 0)[0]
    return np.nan if len(i) == 0 else eps[i[0]] + (eps[i[0] + 1] - eps[i[0]]) * q[i[0]] / (q[i[0]] - q[i[0] + 1])


# ---------------------------------------------------------------- sections
def s1_duality(spread, decay):
    print("\n[1] <A(t)> / [(1-eps_bar) <P_s(t-1)>]  at t = 10, 1e2, 1e3, 1e4, 1e5  (block_len 1: exactly 1 once the IC is forgotten)")
    for key in sorted(spread):
        if key not in decay:
            continue
        s, d, eb = spread[key], decay[key], key[1]
        m = s["Ps"] > 0
        row = []
        for tq in (10, 100, 1000, 10000, 100000):
            j = np.argmin(abs(d["t"] - tq))
            P = np.exp(np.interp(np.log(d["t"][j] - 1), np.log(s["t"][m & (s["t"] > 0)]), np.log(s["Ps"][m & (s["t"] > 0)])))
            jj = np.argmin(abs(s["t"] - tq))
            A = d["mean"][j]
            if A > 0 and s["Ps"][jj] > 0:
                ratio = A / ((1 - eb) * P)
                err = ratio * np.hypot(d["sem"][j] / A, s["Ps_se"][jj] / s["Ps"][jj])
                row.append(f"{ratio:.3f}±{err:.3f}")
            else:
                row.append("   --    ")
        print(f"   {key[0]:5s} {eb:<7} " + "  ".join(row))


def s2_profile(spread, decay):
    print("\n[2] eps_c free: best eps and chi2r for a power law (kappa=0) and for BVH (kappa=1), linear interpolation of ln A")
    for win in ((1e2, 1e5), (1e3, 1e5)):
        print(f"   window {win}")
        for src, name, step in (("decay", "clean", 1e-5), ("spread", "clean", 1e-5), ("decay", "b1", 5e-5),
                                ("spread", "b1", 5e-5), ("spread", "b6", 5e-5)):
            pool = spread if src == "spread" else decay
            keys = sorted(k for k in pool if k[0] == name)
            if not keys:
                continue
            best0, best1 = (np.nan, np.inf), (np.nan, np.inf)
            for k1, k2 in zip(keys[:-1], keys[1:]):
                t, A1, s1 = curve(spread, decay, src, k1)
                _, A2, s2 = curve(spread, decay, src, k2)
                for e in np.arange(k1[1], k2[1] + 1e-12, step):
                    w = (e - k1[1]) / (k2[1] - k1[1])
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ln = (1 - w) * np.log(A1) + w * np.log(A2)
                        sl = np.hypot((1 - w) * s1 / A1, w * s2 / A2)
                    ok = np.isfinite(ln)
                    if ((t[ok] >= win[0]) & (t[ok] <= win[1])).sum() < 10:
                        continue
                    A = np.exp(ln[ok])
                    c0, c1 = bc_chi2(t[ok], A, sl[ok] * A, *win, 0.0), bc_chi2(t[ok], A, sl[ok] * A, *win, 1.0)
                    best0 = min(best0, (e, c0), key=lambda x: x[1])
                    best1 = min(best1, (e, c1), key=lambda x: x[1])
            print(f"     {name:5s} {src:6s} power law: eps={best0[0]:.5f} chi2r={best0[1]:7.2f} | BVH: eps={best1[0]:.5f} chi2r={best1[1]:8.2f}")


def s3_interp(spread, decay):
    print("\n[3] block_len 1: Box-Cox verdict at interpolated eps, by interpolation nodes and window")
    for src in ("decay", "spread"):
        pool = spread if src == "spread" else decay
        if ("b1", 0.1428) not in pool:
            continue
        for nodes in ((0.1428, 0.144), (0.1416, 0.1428, 0.144)):
            t = curve(spread, decay, src, ("b1", nodes[0]))[0]
            lnA = [np.log(curve(spread, decay, src, ("b1", e))[1]) for e in nodes]
            sl = [curve(spread, decay, src, ("b1", e))[2] / curve(spread, decay, src, ("b1", e))[1] for e in nodes]
            for e in (0.1430, 0.1432):
                w = [np.prod([(e - nodes[j]) / (nodes[i] - nodes[j]) for j in range(len(nodes)) if j != i]) for i in range(len(nodes))]
                A = np.exp(sum(wi * li for wi, li in zip(w, lnA)))
                sA = A * np.sqrt(sum((wi * si) ** 2 for wi, si in zip(w, sl)))
                out = []
                for win in ((1e2, 1e5), (1e3, 1e5)):
                    k, cmin, c0, c1 = scan(t, A, sA, *win)
                    out.append(f"{win}: best kappa {k:+.2f}, chi2r power {c0:5.2f} / BVH {c1:5.2f}")
                print(f"   {src:6s} nodes {str(nodes):24s} eps={e}: " + " | ".join(out))


def s4_joint(spread, decay, nboot=400):
    print("\n[4] zero-curvature eps of ln O vs ln t, O = P_s, N, R (spreading) and A (decay); chunk bootstrap")
    rng = np.random.default_rng(7)
    for win in ((1e3, 1e5), (3e3, 1e5), (1e3, 3e4)):
        print(f"   window {win}")
        for name in ("clean", "b1", "b6"):
            keys = sorted(k for k in spread if k[0] == name)
            eps = np.array([k[1] for k in keys])
            q0, qb = {o: [] for o in ("P_s", "N", "R")}, {o: [] for o in ("P_s", "N", "R")}
            for k in keys:
                ch, t = spread[k]["chunks"], spread[k]["t"]
                n = ch["runs"].shape[0]
                for o, y in obs_from_chunks(ch).items():
                    q0[o].append(curvature(t, y, win))
                bs = {o: [] for o in q0}
                for _ in range(nboot):
                    i = rng.integers(0, n, n)
                    for o, y in obs_from_chunks({kk: a[i] for kk, a in ch.items()}).items():
                        bs[o].append(curvature(t, y, win))
                for o in q0:
                    qb[o].append(bs[o])
            parts = []
            for o in ("P_s", "N", "R"):
                c = zero_cross(eps, np.array(q0[o]))
                cb = np.array([zero_cross(eps, np.array(qb[o])[:, j]) for j in range(nboot)])
                parts.append(f"{o}: {c:.5f}±{np.nanstd(cb, ddof=1):.5f}")
            dk = sorted(k for k in decay if k[0] == name)
            if dk:
                qa = np.array([curvature(decay[k]["t"], decay[k]["mean"], win) for k in dk])
                parts.append(f"A(decay): {zero_cross(np.array([k[1] for k in dk]), qa):.5f}")
            print(f"     {name:5s} " + "   ".join(parts))


def s5_response(spread, decay, span=10 ** 0.5):
    print("\n[5] 1/nu_eff(t) = d ln chi / d ln t over a factor 10^0.5 ending at t; (a) = forward/backward difference ratio (1 = linear)")
    TQ = 10 ** np.arange(2.0, 5.01, 0.5)
    print("   t =              " + "".join(f"{x:>18.0e}" for x in TQ))
    for src, name in (("decay", "clean"), ("decay", "b1"), ("spread", "b1"), ("decay", "b6"), ("spread", "b6")):
        pool = spread if src == "spread" else decay
        keys = sorted(k for k in pool if k[0] == name)
        for i in range(1, len(keys) - 1):
            cs = [curve(spread, decay, src, keys[i + d]) for d in (-1, 0, 1)]
            e0, e2 = keys[i - 1][1], keys[i + 1][1]
            row = []
            for tq in TQ:
                vals = []
                for tt in (tq / span, tq):
                    l, sl = [], []
                    for (t, A, s) in cs:
                        m = (t > 0) & (A > 0)
                        l.append(np.interp(np.log(tt), np.log(t[m]), np.log(A[m])))
                        sl.append(np.interp(np.log(tt), np.log(t[m]), (s / np.where(A > 0, A, 1))[m]))
                    chi = (l[0] - l[2]) / (e2 - e0)
                    vals.append((chi, np.hypot(sl[0], sl[2]) / (e2 - e0), (l[1] - l[2]) / (l[0] - l[1])))
                (ca, sa, _), (cb, sb, asym) = vals
                row.append(f"{np.log(cb / ca) / np.log(span):5.2f}±{np.hypot(sa / ca, sb / cb) / np.log(span):4.2f}(a{asym:3.1f})"
                           if ca > 0 and cb > 0 else f"{'--':>18s}")
            print(f"   {name:5s} {src:6s} {keys[i][1]:<7}" + "".join(f"{x:>18s}" for x in row))
    print("   DP 0.577 | paper nu_t = 2.25 -> 0.444 | BVH 2/(ln t + c'), c'=0: " + " ".join(f"{2 / np.log(x):.2f}" for x in TQ))


def s6_local(spread, decay):
    print("\n[6] local spreading exponents (span 2): mean over t in [50,400] -> mean over t in [1e4,1e5]")
    for key in sorted(spread):
        s, t = spread[key], spread[key]["t"]
        row = []
        for y, sign in ((s["Ps"], -1), (s["N"], 1), (np.sqrt(s["R2"]), 1)):
            T, sl = st.local_slope(t, y, span=2.0)
            early = sign * sl[(T >= 50) & (T <= 400)].mean()
            late = sign * sl[(T >= 1e4) & (T <= 1e5)].mean() if ((T >= 1e4) & np.isfinite(sl)).any() else np.nan
            row.append(f"{early:6.3f} -> {late:6.3f}")
        print(f"   {key[0]:5s} {key[1]:<7} delta {row[0]}   theta {row[1]}   1/z {row[2]}")


def s7_decades(spread, decay):
    print("\n[7] per-decade delta = log10[A(t/10)/A(t)] from the decade end points (same block phase for block_len 6)")
    for src, pool in (("decay", decay), ("spread", spread)):
        for key in sorted(pool):
            if key[1] not in (0.2945, 0.1416, 0.1428, 0.144, 0.1092, 0.1104, 0.1116):
                continue
            t, A, s = curve(spread, decay, src, key)
            out = []
            for a, b in ((1e2, 1e3), (1e3, 1e4), (1e4, 1e5)):
                ia, ib = np.argmin(abs(t - a)), np.argmin(abs(t - b))
                out.append(f"{np.log10(A[ia] / A[ib]):.4f}±{np.hypot(s[ia] / A[ia], s[ib] / A[ib]) / np.log(10):.4f}")
            print(f"   {src:6s} {key[0]:5s} {key[1]:<7} " + "  ".join(out))


def s8_bvh_point(spread, decay, win=(1e3, 1e5)):
    print("\n[8] at the eps where P_s is BVH-straight (kappa = 1) and where it is a power law (kappa = 0): Box-Cox lam = 1/y of R/t and N/t")
    for name, e_bvh, e_pl, nodes in (("b6", 0.1102, 0.11045, (0.1092, 0.1104, 0.1116)), ("b1", 0.1430, 0.14305, (0.1428, 0.144))):
        for e in (e_bvh, e_pl):
            j = max(i for i in range(len(nodes) - 1) if nodes[i] <= e)
            k1, k2 = (name, nodes[j]), (name, nodes[j + 1])
            w = (e - nodes[j]) / (nodes[j + 1] - nodes[j])
            out = []
            for obs in ("R", "N"):
                g, sg = [], []
                for k in (k1, k2):
                    s, t = spread[k], spread[k]["t"]
                    y, sy = (np.sqrt(s["R2"]), 0.5 * s["R2_se"] / s["R2"]) if obs == "R" else (s["N"], s["N_se"] / s["N"])
                    g.append(np.log(y / np.where(t > 0, t, np.nan))); sg.append(sy)
                ln = (1 - w) * g[0] + w * g[1]
                A, sA = np.exp(ln), np.exp(ln) * np.hypot((1 - w) * sg[0], w * sg[1])
                ok = np.isfinite(A) & (t > 0)
                lam, cmin, c0, _ = scan(t[ok], A[ok], sA[ok], *win, grid=LAMS)
                out.append(f"{obs}/t: y = {1 / lam if lam > 0.01 else np.inf:5.2f} (chi2r {cmin:6.2f}; power law {c0:7.2f})")
            print(f"   {name} eps={e:.5f}  " + "   ".join(out))


def s9_short_windows(spread, decay, nboot=200):
    """Added 2026-10-05 (H §7.20): the joint test of [4] in windows a spin-chain run can reach (t <= 1e4)."""
    print("\n[9] joint test in short windows: zero-curvature eps of P_s, N, R, and their spread (max - min) / eps_c")
    rng = np.random.default_rng(7)
    for win in ((1e2, 1e3), (1e2, 3e3), (3e2, 3e3), (3e2, 1e4), (1e3, 1e5)):
        print(f"   window {win}")
        for name in ("clean", "b1", "b6"):
            keys = sorted(k for k in spread if k[0] == name)
            eps = np.array([k[1] for k in keys])
            q0, qb = {o: [] for o in ("P_s", "N", "R")}, {o: [] for o in ("P_s", "N", "R")}
            for k in keys:
                ch, t = spread[k]["chunks"], spread[k]["t"]
                n = ch["runs"].shape[0]
                for o, y in obs_from_chunks(ch).items():
                    q0[o].append(curvature(t, y, win))
                bs = {o: [] for o in q0}
                for _ in range(nboot):
                    i = rng.integers(0, n, n)
                    for o, y in obs_from_chunks({kk: a[i] for kk, a in ch.items()}).items():
                        bs[o].append(curvature(t, y, win))
                for o in q0:
                    qb[o].append(bs[o])
            cs, parts = [], []
            for o in ("P_s", "N", "R"):
                c = zero_cross(eps, np.array(q0[o]))
                cb = np.array([zero_cross(eps, np.array(qb[o])[:, j]) for j in range(nboot)])
                cs.append(c)
                parts.append(f"{o}: {c:.5f}±{np.nanstd(cb, ddof=1):.5f}")
            spread_rel = (np.nanmax(cs) - np.nanmin(cs)) / np.nanmean(cs)
            print(f"     {name:5s} " + "   ".join(parts) + f"   spread {100 * spread_rel:.2f}%")


def _resample(ch, rng):
    i = rng.integers(0, ch["runs"].shape[0], ch["runs"].shape[0])
    return {k: a[i] for k, a in ch.items()}


def s10_measured_drift(spread, decay, nboot=300):
    """Added 2026-10-05 (H §7.21): the joint test without interpolation in eps. At every MEASURED grid value, the
    local exponent of P_s, N, R over the decades 1e3-1e4 and 1e4-1e5, and its drift (late - early); 0 = power law."""
    print("\n[10] measured local exponents on 1e3-1e4 | 1e4-1e5 and their drift (late - early) ± chunk bootstrap")
    rng = np.random.default_rng(1)

    def slopes(t, y):
        out = []
        for a, b in ((1e3, 1e4), (1e4, 1e5)):
            ia, ib = np.argmin(abs(t - a)), np.argmin(abs(t - b))
            out.append(np.log(y[ib] / y[ia]) / np.log(t[ib] / t[ia]))
        return np.array(out)

    for name in ("clean", "b1", "b6"):
        print(f"   {name}")
        for k in sorted(k for k in spread if k[0] == name):
            ch, t = spread[k]["chunks"], spread[k]["t"]
            o = obs_from_chunks(ch)
            s0 = {q: slopes(t, o[q]) for q in o}
            bs = {q: [] for q in o}
            for _ in range(nboot):
                ob = obs_from_chunks(_resample(ch, rng))
                for q in o:
                    bs[q].append(slopes(t, ob[q]))
            row = []
            for q, lab in (("P_s", "-delta"), ("N", "theta"), ("R", "1/z")):
                d = s0[q][1] - s0[q][0]
                sd = np.nanstd([b[1] - b[0] for b in bs[q]], ddof=1)
                row.append(f"{lab} {s0[q][0]:+.3f}|{s0[q][1]:+.3f} drift {d:+.3f}±{sd:.3f}")
            print(f"     {k[1]:<7} " + "   ".join(row))


def s11_response_at_point(spread, decay, nboot=300):
    """Added 2026-10-05 (H §7.21): 1/nu_eff(t) = d ln chi / d ln t of spreading P_s, with chi = -d ln P_s / d eps at a
    point from a quadratic in eps through three measured grid values (per time). Errors are statistical only; the
    quadratic over a 0.0024 span is a model (late times are nonlinear), which is why N1-S / N1-C are needed."""
    print("\n[11] 1/nu_eff per half decade from spreading P_s, quadratic in eps through three grid values (stat. errors only)")
    rng = np.random.default_rng(2)
    TQ = 10 ** np.arange(2.0, 5.01, 0.5)
    for name, nodes, e0s in (("clean", (0.2944, 0.2945, 0.2946), (0.2945,)),
                             ("b1", (0.1416, 0.1428, 0.144), (0.14306, 0.1434)),
                             ("b6", (0.1092, 0.1104, 0.1116), (0.11043, 0.1108))):
        t = spread[(name, nodes[0])]["t"]
        idx = [np.argmin(abs(t - x)) for x in TQ]
        chs = [spread[(name, e)]["chunks"] for e in nodes]

        def inv_nu(chs_, e0):
            Y = np.array([np.log(c["surv"].sum(0) / c["runs"].sum(0)) for c in chs_])[:, idx]
            chi = -np.polyfit(np.array(nodes) - e0, Y, 2)[1]
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.log(chi[1:] / chi[:-1]) / np.log(10 ** 0.5)

        for e0 in e0s:
            v0 = inv_nu(chs, e0)
            se = np.nanstd([inv_nu([_resample(c, rng) for c in chs], e0) for _ in range(nboot)], axis=0, ddof=1)
            print(f"   {name:5s} at {e0:<8}" + "  ".join(f"{x:.0e}: {v:.2f}±{s:.2f}" for x, v, s in zip(TQ[1:], v0, se)))
    print("   DP 0.577 | paper nu_t = 2.25 -> 0.444 | BVH 2/ln t (no offset) at 1e3, 1e4, 1e5: 0.29, 0.22, 0.17")


if __name__ == "__main__":
    spread, decay = load_all()
    if not decay:
        print("summary.json not found: decay rows are skipped")
    wanted = set(sys.argv[1:]) or {str(i) for i in range(1, 12)}
    for n, f in (("1", s1_duality), ("2", s2_profile), ("3", s3_interp), ("4", s4_joint), ("5", s5_response),
                 ("6", s6_local), ("7", s7_decades), ("8", s8_bvh_point), ("9", s9_short_windows),
                 ("10", s10_measured_drift), ("11", s11_response_at_point)):
        if n in wanted:
            with np.errstate(divide="ignore", invalid="ignore"):
                f(spread, decay)
