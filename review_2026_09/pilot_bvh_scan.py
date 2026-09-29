"""Pilot scan (2026-09-29) for BVH-like disorder in the Stavskaya model (NumPy port of stavskaya_step!).

epsilon = eps_u with probability p_bad (inactive step), eps_l = eps_u/lower_div otherwise, held for
block_len steps (per sample). Prints Q = 1 - rho and the slope of 1/Q vs ln t in the middle and last
thirds of the window: roughly constant slope = BVH-like critical; falling = active; growing = inactive.

    python3 review_2026_09/pilot_bvh_scan.py <lower_div> <p_bad> <block_len> <eps_u> [<eps_u> ...]
    python3 review_2026_09/pilot_bvh_scan.py 20 0.2 1 0.59 0.60 0.61

Results (L = 2000, t <= 2000, 300 samples; only brackets the critical point):
    block_len = 1: eps_u,c ~ 0.595-0.60   (0.59 active, 0.61 inactive)
    block_len = 6: eps_u,c ~ 0.455-0.46   (0.45 active, 0.47 inactive)
"""
import numpy as np, sys, time
def run(eps_u, lower_div, p_bad, b, L=2000, T=2000, S=300, seed=0):
    rng = np.random.default_rng(seed)
    eps_l = eps_u/lower_div
    cur = rng.random((S, L)) < 0.5            # True = 1 = inactive (absorbing value), as in the Julia code
    rec = np.unique(np.round(np.logspace(0, np.log10(T), 25)).astype(int))
    Q = []; k = 0
    for t in range(1, T+1):
        if (t-1) % b == 0:
            eps = np.where(rng.random(S) < p_bad, eps_u, eps_l)[:, None]
        cur = (rng.random((S, L), dtype=np.float32) < eps) | (cur & np.roll(cur, 1, axis=1))
        if t == rec[k]:
            Q.append(1 - cur.mean()); k += 1
            if k == len(rec): break
    return rec, np.array(Q)
if __name__ == "__main__":
    lower_div, p_bad, b = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    for eps_u in map(float, sys.argv[4:]):
        t0 = time.time(); t, Q = run(eps_u, lower_div, p_bad, b)
        invQ = 1/np.maximum(Q, 1e-9)
        lt = np.log(t); n = len(t)
        s1 = np.polyfit(lt[n//3:2*n//3], invQ[n//3:2*n//3], 1)[0]; s2 = np.polyfit(lt[2*n//3:], invQ[2*n//3:], 1)[0]
        print(f"b={b} eps_u={eps_u:.3f} eps_l={eps_u/lower_div:.4f}  Q(t=100)={Q[np.searchsorted(t,100)]:.4f} Q(T)={Q[-1]:.5f}  "
              f"d(1/Q)/dlnt mid={s1:8.3f} late={s2:8.3f}  ({time.time()-t0:.0f}s)", flush=True)
