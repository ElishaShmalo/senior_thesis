"""Spin-chain checks behind PROJECT_HISTORY.md §7.20 (2026-10-05). Nothing is written anywhere.

Run from anywhere:
    python3 heisen_spin_chain/_claude_scratch/spin_checks_2026_10_05.py          # all sections (~10 s)
    python3 heisen_spin_chain/_claude_scratch/spin_checks_2026_10_05.py 1 3      # only sections 1 and 3

Sections
  1  a-response exponent 1/nu_eff(t) = d ln chi / d ln t, chi = d ln<S_diff>/da, from the OLD local means
     (data/s_diff_per_time/N4/a*/IC1700/L2000: L = 2000, 1700 samples, every step to t = 1e4, first-octant ICs)
  2  relative sample spread std(S_diff)/<S_diff> against the finite-chain estimate sqrt(xi/L), xi = t^0.75
  3  J-sign steps are one Heisenberg step up to a symmetry that fixes the target spiral (exact identity, checked
     numerically on L = 12 with the EOM of heisen_spin_chain/utils/dynamics.jl)
"""
import glob
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OLD = REPO / "data" / "s_diff_per_time" / "N4"


def load_old():
    A = {}
    for d in sorted(OLD.glob("a*")):
        m, s = glob.glob(f"{d}/*/L2000/*mean.pickle"), glob.glob(f"{d}/*/L2000/*stds.pickle")
        if not m:
            continue
        a = float(d.name[1:].replace("p", "."))
        A[a] = (np.asarray(pickle.load(open(m[0], "rb"))), np.asarray(pickle.load(open(s[0], "rb"))))
    return A   # row i <-> t = i (timestep 1)


def s1_response(A, n=1700):
    print("\n[1] a-response from the old means: chi = d ln<S>/da (weighted linear fit across a), 1/nu_eff over (t/sqrt10, t]")

    def chi(aset, i):
        a = np.array(aset)
        y = np.array([np.log(A[x][0][i]) for x in aset])
        s = np.array([A[x][1][i] / np.sqrt(n) / A[x][0][i] for x in aset])
        w = 1 / s ** 2
        X = np.c_[np.ones_like(a), a - a.mean()]
        C = np.linalg.inv(X.T @ (X * w[:, None]))
        b = C @ (X.T @ (w * y))
        r = (y - X @ b) / s
        return b[1], np.sqrt(C[1, 1]), r @ r / max(len(a) - 2, 1)

    sets = {"narrow 0.7570-0.7595 (9 values)": [a for a in A if 0.7569 < a < 0.7596],
            "wide 0.7535-0.7615 (13 values)": sorted(A),
            "outer pair 0.7555/0.7595": [0.7555, 0.7595]}
    span = 10 ** 0.5
    for name, aset in sets.items():
        print(f"   {name}\n      t      chi ± se       chi2r(linear)   1/nu_eff")
        for tq in (30, 100, 300, 1000, 3000, 9000):
            c1, s1, _ = chi(aset, int(round(tq / span)))
            c2, s2, q2 = chi(aset, tq)
            inv = np.log(c2 / c1) / np.log(span) if c1 > 0 and c2 > 0 else np.nan
            print(f"   {tq:6d}  {c2:7.1f} ± {s2:5.1f}   {q2:8.2f}        {inv:5.2f} ± {np.hypot(s1 / c1, s2 / c2) / np.log(span):4.2f}")
    print("   DP 0.577 | paper nu_t = 2.24 -> 0.446 | BVH 2/ln t (no offset): "
          + " ".join(f"{2 / np.log(x):.2f}" for x in (30, 100, 300, 1000, 3000, 9000)))


def s2_spread(A):
    print("\n[2] std(S_diff)/<S_diff> (old data) vs finite-chain estimate sqrt(xi/L), xi = t^0.75, L = 2000")
    T = (10, 30, 100, 300, 1000, 3000, 9000)
    for a in (0.7555, 0.758, 0.7595):
        m, s = A[a]
        print(f"   a = {a:<7}" + "".join(f"{s[t] / m[t]:7.2f}" for t in T))
    print("   estimate " + "".join(f"{np.sqrt(t ** 0.75 / 2000):7.2f}" for t in T) + "      (t = " + ", ".join(map(str, T)) + ")")


def s3_symmetry():
    from scipy.integrate import solve_ivp
    print("\n[3] phi_s(S) = G_s^-1 phi_(++)(G_s S) with G_s fixing the spiral (so G_s commutes with the push and keeps S_diff)")
    L = 12
    rng = np.random.default_rng(3)

    def phi(S, J, T=1.0):
        J = np.asarray(J, float)

        def f(t, u):
            s = u.reshape(L, 3)
            return np.cross(-J * (np.roll(s, 1, 0) + np.roll(s, -1, 0)), s).ravel()
        return solve_ivp(f, (0, T), S.ravel(), method="DOP853", rtol=1e-12, atol=1e-12).y[:, -1].reshape(L, 3)

    odd = np.arange(L) % 2 == 1

    def Z(S):   # pi rotation about z on odd sites
        S = S.copy(); S[odd, :2] *= -1; return S

    def Y(S):   # pi rotation about y on odd sites
        S = S.copy(); S[odd, 0] *= -1; S[odd, 2] *= -1; return S

    def R(S):   # reflection j -> -j
        return S[(-np.arange(L)) % L]

    def K(S):   # S_j -> -S_{j+2}: anti-canonical, so it reverses the direction of time
        return -np.roll(S, -2, 0)

    def Kinv(S):
        return -np.roll(S, 2, 0)

    S0 = np.array([[0, np.cos(np.pi * j / 2), np.sin(np.pi * j / 2)] for j in range(L)])
    G = {(-1, -1): (Z, Z),
         (1, -1): (lambda S: K(R(Y(S))), lambda S: Y(R(Kinv(S)))),
         (-1, 1): (lambda S: K(R(Y(Z(S)))), lambda S: Z(Y(R(Kinv(S)))))}
    S = rng.normal(size=(L, 3))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    a = 0.758

    def push(S):
        v = (1 - a) * S0 + a * S
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    for s, (g, ginv) in G.items():
        err = abs(phi(S, (s[0], s[1], 1)) - ginv(phi(g(S), (1, 1, 1)))).max()
        diff = abs(phi(S, (s[0], s[1], 1)) - phi(S, (1, 1, 1))).max()
        print(f"   J signs {s}: identity error {err:.1e} (the step itself differs from (++) by {diff:.2f}); "
              f"G fixes S0: {np.allclose(g(S0), S0)}; G commutes with the push: {np.allclose(push(g(S)), g(push(S)))}")


if __name__ == "__main__":
    wanted = set(sys.argv[1:]) or set("123")
    A = load_old() if wanted & {"1", "2"} else None
    if "1" in wanted:
        s1_response(A)
    if "2" in wanted:
        s2_spread(A)
    if "3" in wanted:
        s3_symmetry()
