# Project history: control transition with temporal randomness

*Long-term record of everything done on the referee-conflict work, oldest first. The active task list is `REFEREE_CONFLICT_REVIEW.md`; finished items move from there into this file.*

*Started 2026-09-25. Sections 0–6 are the original review (2026-09-24); §7 is the follow-up work on your annotations; §7.11 onward are later rounds.*

---

## Original review: the referee's objection, the follow-up numerics, and a code review

*Written 2026-09-24 after reading `ControlTransitionsPaper.pdf` (arXiv:2606.09297v1), Barghathi–Vojta–Hoyos, "Contact process with temporal disorder" (arXiv:1603.08075v2, PRE 94, 022111), and the scripts/notebooks listed at the end.*

---

## 0. Summary

1. **The conflict.** The paper says the spin-chain control transition, and a Stavskaya automaton with temporal randomness, share a new "temporally-random DP" universality class with ordinary power-law exponents ($\delta\approx0.11,\ \nu_t\approx2.25,\ z\approx1.35$–$1.45$). Barghathi, Vojta and Hoyos (BVH) show that temporal disorder sends 1+1D DP to an **infinite-noise critical point**. That point has no power-law exponents at all:
   - the density decays like $1/\ln t$;
   - $z=1$ with logarithmic corrections;
   - $\ln\xi_t\sim r^{-1/2}$;
   - the distribution of $\ln\rho$ broadens without bound.

   They argue that finite, non-universal power-law exponents like Jensen's (the paper's ref. [37]) come from a **slow crossover**. They are not a real fixed point. The paper's central claim is a new instance of that same situation.

2. **Why your numbers match neither DP nor BVH:** they are almost certainly **effective (running) exponents in a crossover window**. For every exponent whose direction is well defined, your value sits *between* clean DP and the infinite-noise limit:

   | | clean DP | spin chain | Stavskaya (paper) | infinite-noise limit |
   |---|---|---|---|---|
   | $\delta$ (power-law decay) | 0.159 | 0.11 | 0.105 | → 0 (log decay) |
   | $z$ | 1.58 | 1.34 | 1.45 | → 1 (+ log corrections) |
   | $\nu_t$ | 1.73 | 2.24 | 2.25 | → ∞ (activated scaling) |

   A pure $\rho\propto1/\ln t$ decay, fed through the paper's estimator $\log_{10}[\rho(t/10)/\rho(t)]$, gives $\delta_{\rm eff}=0.114\to0.075$ over the Fig. 4 window ($t\approx2\times10^4$–$2\times10^6$). That is exactly the size of the "$\delta=0.105$" reported. So the Stavskaya value alone cannot tell the two pictures apart (§3.1).

3. **Your own follow-up data already point toward BVH.** On the sliding-$p$ Stavskaya model, the log-scaling fit gives $\bar\delta\approx1.0$, which is BVH's prediction. The two disorder distributions give clearly different FSS exponents, which fits non-universal crossover behaviour. On the spin chain over $t=50$–$5000$, a pure power law and the BVH form $1/S=A+B\ln t$ fit about equally well (§3.3). The data cannot yet discriminate, and that is the main problem.

4. **Code.** I found no bug that would, by itself, turn a DP or infinite-noise result into the reported exponents. There are real problems, though:
   - The spin-chain random initial state is **not uniform on the sphere**: every spin is in the first octant.
   - Several analysis steps **assume power laws**, which biases $a_c$ and $\delta$.
   - Time sampling and system sizes put most of the FSS data **in the finite-size tail**.
   - Several notebooks are in a **non-reproducible, out-of-order state**, and scripts on disk no longer match the data they are read against.

   §4 lists everything with file and line references.

5. **What would settle it:** the tests BVH use are cheap and mostly doable with data you already have. They are the $1/\rho$ vs $\ln t$ line, $1/\delta_{\rm eff}$ vs $\ln t$, $\ln t_x$ vs $r^{-1/2}$, and the width of $P(\ln\rho)$ vs $\ln t$. Add a knob for disorder strength and block length. See §5.

---

## 1. What the project is

### 1.1 Spin-chain model (the thesis and paper)
- **Chain:** classical Heisenberg chain of $L$ unit spins, periodic boundary conditions, $H=-\sum_i(J_xS^x_iS^x_{i+1}+J_yS^y_iS^y_{i+1}+S^z_iS^z_{i+1})$.
- **Temporal randomness:** the signs of $J_x$ and $J_y$ are redrawn independently as $\pm1$ every unit time step. This was added to kill solitons (Supp. Fig. S5). It is the model's only source of temporal randomness.
- **One time step:** Hamiltonian evolution for $\tau=1/J$, then a contractive control map toward the period-4 spiral $\mathbf S^0_j=(0,\cos\frac{\pi j}{2},\sin\frac{\pi j}{2})$: $\mathbf S_j\to\frac{(1-a)\mathbf S^0_j+a\mathbf S_j}{|\cdot|}$.
- **Order parameter:** the activity $S_{\rm diff}=\frac1L\sum_i\langle|\mathbf S_i-\mathbf S^0_i|\rangle$.
- **Other observable:** the leading Lyapunov exponent, computed with Benettin's method.
- **Paper's claims:**
  - A controlled/chaotic transition at $a_c\approx0.758$.
  - A continuous activity transition with $\beta=0.23(2)$, $\nu_t=2.24(15)$, $z=1.34(15)$.
  - A discontinuous jump in the Lyapunov exponent.
  - A Stavskaya automaton with a time-random healing probability $\varepsilon(t)=0.5X^n$ gives the same exponents ($\beta=0.236$, $\nu_t=2.25$, $z=1.45$). Therefore both models sit in a "temporally random DP" class that satisfies the temporal Harris/CCFS bound $\nu_t\ge2$ without saturating it.

### 1.2 Stavskaya model (the `stavskya_mc/` code)
- **Update rule:** $\eta_i(t+1)=1$ with probability $\varepsilon(t)$, otherwise $\eta_i(t)\eta_{i-1}(t)$.
  - $\eta=1$ is healthy and all-healthy is absorbing.
  - The activity is $1-\rho$, where `calculate_avg_alive` returns $\rho$, the healthy fraction.
  - Clean critical point: $\varepsilon_c=0.2945$.
- **Post-rejection disorder variants:**
  - **"upper/lower binary"** (`time_rand_window_binary`): each step, $\varepsilon=\varepsilon_u$ with probability $p=0.8$, otherwise $\varepsilon_l=\varepsilon_u/20$. The control parameter is $\bar\varepsilon=p\varepsilon_u+(1-p)\varepsilon_l=0.81\,\varepsilon_u$. This mirrors BVH's binary distribution $W(\lambda)=p\,\delta(\lambda-\lambda_h)+(1-p)\,\delta(\lambda-\lambda_h/20)$ with $p=0.8$, but here the *healing* rate is the random quantity.
  - **"sliding $p$"** (`time_rand_slidding_p`): $\varepsilon_u=0.43$ and $\varepsilon_l=0.0215$ are fixed, and the control parameter is $p$, the probability of an inactive (high-healing) step. There is also an older variant with $\varepsilon_u=0.42$, $\varepsilon_l=0.042$.

---

## 2. The peer-review conflict

### 2.1 What BVH (arXiv:1603.08075) establish
- **Relevance:** clean DP has $\nu_\parallel=1.73<2$, so by Kinzel's temporal Harris criterion temporal disorder is **relevant**.
- **Real-time strong-noise RG** (a temporal analogue of Fisher's RG): the flow is of Kosterlitz–Thouless type, toward an **infinite-noise fixed point**. In 1D:
  - $\rho_{\rm av}(t)\sim(\ln t)^{-\bar\delta}$ with $\bar\delta=1$, and survival $P_s\sim(\ln t)^{-1}$;
  - $\ln\xi_t\sim|r|^{-\bar\nu_\parallel}$ with $\bar\nu_\parallel=1/2$ (activated, not a power law);
  - stationary density $\rho_{st}\sim|r|^{\beta}$ with $\beta=1/2$;
  - $z=1$ with log corrections: $R\sim t(\ln t)^{-y_R}$, $N_s\sim t(\ln t)^{-y_N}$, with $y_R\approx1.7$ and $y_N\approx3.6$ numerically;
  - $P(x=-\ln\rho)$ broadens without limit, with width $\propto\ln t$;
  - **temporal Griffiths phase:** on the active side the lifetime is $\tau\sim N^{1/\kappa}$, and $\kappa\to d$ at criticality.
- **Monte Carlo:** strong disorder ($\Delta t=6$, $\lambda_l=\lambda_h/20$, $p=0.8$) matches the predictions over about 4 decades. Weaker disorder ($\Delta t=1$, $\lambda_l=\lambda_h/10$) first follows clean DP and crosses over at $t\sim10^3$. Even weaker disorder does not reach the asymptotic regime at all.
- **On Jensen's results** ("temporally disordered DP" with continuously varying power-law exponents, the paper's ref. [37]): BVH attribute them to this slow crossover. Some of Jensen's $\nu_\parallel$ values even violate $\nu_\parallel>2$.

### 2.2 Why that contradicts the paper
The paper's universality class is essentially Jensen's picture: finite power-law exponents with $\nu_t\approx2.25$ that "satisfy but do not saturate" the bound. BVH's point is that a finite-disorder power-law fixed point is not where the flow ends. The asymptotic behaviour is logarithmic and activated.

The two models agreeing is weaker evidence than it looks. Both are temporally random DP-like systems, both were analysed with the same power-law estimators, and both over similar log-time windows. They will drift through similar effective exponents.

---

## 3. What the follow-up analysis shows

### 3.1 The estimators assume power laws and are biased under BVH
- **$\delta$ from flatness of $\log_b[\rho(t/b)/\rho(t)]$** (Figs. 2b and 4; notebook cells "determining_ac_delta"). At the true infinite-noise critical point this ratio is **never flat**: it drifts to zero as $\approx\log_{10}[\ln t/\ln(t/10)]$.
  - Slightly **inactive (controlled)** curves first follow the critical curve and then turn upward (BVH Fig. 5 inset). So the flattest curve in a finite window sits on the **controlled side** of the true critical point. The $\delta$ read off from it is just the running value in that window.
  - Worked numbers for pure $1/\ln t$:

    | $t$ | $10^3$ | $10^4$ | $10^5$ | $10^6$ | $10^7$ |
    |---|---|---|---|---|---|
    | $\delta_{\rm eff}$ | 0.176 | 0.125 | 0.097 | 0.079 | 0.067 |

    The Fig. 4 window ($\ln(1/t)\in[-14.5,-10]$) gives 0.11–0.075.
- **$\nu_t$ from the collapse $t^{1/\nu_t}\Delta$.** If the true variable is $\Delta(\ln t)^{2}$, a power-law collapse over a finite window gives an effective $\nu_t$ that keeps growing as you go later in time. Any measured value just above 2 fits that drift.
- **$z$ from FSS at fixed $t=L^{z}$, or from $t/L^z$ collapses.** With temporal disorder the finite-size lifetime is a power law $L^{1/\kappa}$ even in the active phase (temporal Griffiths). So an FSS "$z$" measured at a slightly misplaced critical point measures $1/\kappa$, which is non-universal. On top of that, BVH's $z=1$ carries $(\ln t)^{-y_R}$ corrections. With $y_R\approx1.7$ the effective $z$ is about 1.3 at $t\sim10^3$ and about 1.2 at $t\sim10^4$, the same range as the reported 1.34–1.45. This is suggestive only, since $y_R$ was measured for the contact process.
- **$\beta$ is never measured independently.** Table I reports $\beta=\delta\nu_t$, the product of an exponent that drifts to 0 and one that drifts to ∞. It carries no independent information. BVH predict $\beta=1/2$ from the stationary density.

### 3.2 Stavskaya follow-up results (from notebook outputs)
| model / analysis | result | comment |
|---|---|---|
| upper/lower, FSS at $t=L^{1.45}$, $L=1000$–$16000$ (`..._rho_per_ep.ipynb`) | $\bar\varepsilon_c=0.27092$, $\nu_\perp=1.97$, $\beta=0.596$ | **reduced $\chi^2=406$**; $\beta$ is pinned at the upper bound 0.6. The collapse failed. |
| upper/lower, time series, $L=35000$ (`..._rho_per_time.ipynb`) | $\bar\varepsilon_c=0.27033$ assumed; log fit $\alpha\equiv\bar\delta=1.33$ | $\varepsilon_c$ is not consistent with the FSS value; the $\alpha$ extrapolation is poorly controlled (§4.3) |
| sliding $p$, FSS at $t=L^{1.45}$ (`analyze_random_slidding_p.ipynb`) | $p_c=0.47912$, $\nu_\perp=1.46$, $\beta=0.245$, reduced $\chi^2=0.95$ | $\nu_t=z\nu_\perp\approx2.1$ |
| sliding $p$, time series, $L=35000$, 20000 samples (`..._rho_per_time2.ipynb`) | power law $\delta\approx0.094$ at $p_c\approx0.479$; **log fit $\bar\delta=0.997$** | matches BVH's $\bar\delta=1$ |
| sliding $p$ collapse in $\ln t$ | trial $\bar\nu_\perp=0.21$, giving "$\beta=0.21$" | BVH predict $\bar\nu=1/2$ (variable $r(\ln t)^2$) and $\beta=1/2$; the $\ln t$ range (≈7.6–15) is far too short to fix this exponent |

The two disorder distributions give **different** FSS exponents: $\nu_\perp$ 1.97 vs 1.46 and $\beta$ 0.6 vs 0.25. That is not what a single universal power-law fixed point looks like. It is what non-universal crossover looks like.

The distributions also differ in strength. In upper/lower, the inactive steps ($\varepsilon_u\approx0.334$) are only about 13% above the clean $\varepsilon_c$, so the disorder is weaker. In sliding $p$ they are about 46% above ($\varepsilon_u=0.43$), so the disorder is stronger. The stronger-disorder model is the one that gives $\bar\delta\approx1$, which is the BVH trend.

### 3.3 Spin chain: re-analysis of the local pickled means
I recomputed running exponents from `data/s_diff_per_time/N4/a*/IC1700/L2000/*mean.pickle`: $L=2000$, 1700 samples, $t\le10^4$, the same data `analyze_sdiff_per_time3.ipynb` uses.

$\delta_{\rm eff}(T)=\log_{10}[S(T/10)/S(T)]$:

| $a$ | T=50 | 200 | 800 | 3200 | 9990 |
|---|---|---|---|---|---|
| 0.7555 | 0.179 | 0.145 | 0.151 | 0.228 | 0.375 |
| 0.7574 | 0.169 | 0.131 | 0.108 | 0.113 | 0.160 |
| 0.7580 | 0.163 | 0.128 | 0.100 | 0.095 | 0.100 |
| 0.7582 | 0.163 | 0.122 | 0.097 | 0.094 | 0.095 |
| 0.7595 | 0.161 | 0.105 | 0.074 | 0.048 | 0.046 |
| 0.7615 | 0.144 | 0.086 | 0.050 | 0.034 | 0.027 |

- At every $a$ near $a_c$, $\delta_{\rm eff}$ **starts at the clean-DP value (0.16) at early times and drifts down**. That is the textbook signature of a crossover away from clean DP.
- Whether it levels off at about 0.10 (paper) or keeps sinking cannot be decided from $10^{2.5}$ decades.
- Weighted fits over $t\in[50,5000]$ at $a=0.758$–$0.7582$:
  - power law $\delta\approx0.095$–$0.10$, reduced $\chi^2\approx0.9$–$1.5$;
  - BVH form $1/S=A+B\ln t$, reduced $\chi^2\approx0.8$–$1.2$.

  They are statistically indistinguishable. These $\chi^2$ values ignore the strong correlations between times and are for comparison only.
- The notebook's $\ln\ln t$ fit gives $\bar\delta\approx0.69$. It extrapolates $1/\ln\ln t$ from $[0.45,0.65]$ to 0, which is not reliable (§4.3).

---

## 4. Code review: errors and issues

Ordered roughly by how much they could matter.

### 4.1 Spin chain (`heisen_spin_chain/`)
1. **Random initial state is not uniform on the sphere.** In `utils/make_spins.jl:8`, `make_random_state` uses `normalize(rand(3))`. `rand(3)` is uniform in $[0,1)^3$, so every spin lies in the **positive octant**, with mean $\approx(0.52,0.52,0.52)$ and net magnetisation $|m|\approx0.89$. The paper says spins are "uniformly distributed on the unit sphere". The same bug is in `make_random_spin` (line 33), which may also feed the Lyapunov perturbations.
   - **Fix:** `normalize(randn(3))`.
   - It probably does not change the universality class, but it contradicts the methods, changes the transient, and could shift $a_c$ slightly. All spin-chain data would need regenerating anyway if you change it. (I FIXED, PLEASE CONFIRM) → **confirmed, §7.2 (4.1.1)**
2. **Off-by-one in the saved time column.** `lyapunov_exponents/get_sdiff_data_severalL0.jl:75,81-82`: `states_evolve_func(...)[1:step_size:n]` holds states at times $0,\,s,\,2s,\dots$, because index 1 is the initial state. The CSV labels them `t = 1:step_size:n`, i.e. $1,\,s+1,\dots$.
   - `analyze_sdiff_per_time3.ipynb` ignores that column and rebuilds $t=0,1,2,\dots$, so the figures are right.
   - Anyone using the CSV `t` column is off by one step. The script also stops before time $n$ because `1:step:n` excludes index $n+1$. (SUGGEST FIX) → **proposal in §7.2 (4.1.2–4.1.4); implemented 2026-09-25, §7.9**
3. **Whole trajectory held in memory.** `utils/dynamics.jl:126-142` (`random_global_control_evolve`) allocates and stores all $n+1$ states and then subsamples. For $L=2000$ and $n=20L$ that is about 2 GB per sample before the `unflatten_state` copies.
   - It is not a correctness bug. But it is why samples are limited to 500 per job, and under temporal disorder the number of disorder realizations is what limits you.
   - **Fix:** compute $S_{\rm diff}$ on the fly. ((SUGGEST FIX) in more detail) → **proposal in §7.2 (4.1.2–4.1.4); implemented 2026-09-25, §7.9**
4. **Minor points.**
   - The time step is passed as `J` (line 75), so changing `J` silently changes the step. The variable `tau` is unused. (DC)
   - `J_vec` is a global array mutated in place by `L_J_vec[1] *= ±1` (`dynamics.jl:132-133`). This is statistically fine (a uniform random sign times anything is a uniform random sign) but fragile. (SUGGEST FIX) → **proposal in §7.2 (4.1.2–4.1.4); implemented 2026-09-25, §7.9**
   - `make_spiral_state` builds the spiral with SymPy. It is correct but slow. (DC)
5. **Duplicate job scripts.** `get_sdiff_data_severalL0–3.jl` differ only in `init_cond_name_offset` (0/500/1000/1500), which is fine. With SLURM `--requeue`, a preempted job silently regenerates its samples; that is harmless, but worth knowing. (DC)

### 4.2 Stavskaya (`stavskya_mc/`)
1. **Disorder correlation time is one step.** `utils/dynamics.jl:134-158` redraws $\varepsilon$ every step, so the disorder base interval is $\Delta t=1$. BVH needed $\Delta t=6$ (or 3) with a factor-20 contrast to see asymptotic behaviour quickly. With $\Delta t=1$ they saw a crossover at about $10^3$ even for a factor-10 contrast.
   - This is a design choice, not a bug. But it means your runs are exactly in the regime where BVH expect long crossovers.
   - **Suggestion:** add a `block_len` parameter. (FIX - make it like the other nobs and part of the created file's name) → **done, §7.3 and §7.4**
2. **Time sampling is linear and starts late.** `time_step = 20000`, `150000` or `2000`, with the first sample at `t = time_step`.
   - In the $z$ runs (`..._time_data2.jl`, `analyze_*_z*.ipynb`), $L=1250$ has its first point at $t=150{,}000\approx120L$. The collapse sees only the finite-size tail, and there is no data in the $t\ll L^z$ regime.
   - **Fix:** use log-spaced output times from $t=1$. (FIX) → **done, §7.4**
3. **Run length far beyond the light cone.** Examples: `T_f = 100L` (upper/lower), `4500L` (z2), `10^4 L` (`slidding_p_time_data.jl`).
   - In Stavskaya, information travels one site per step. For $t<L$, the disorder-averaged $\rho(t)$ of the periodic chain is **exactly** that of the infinite chain.
   - For bulk decay you want $L\gtrsim t_{\max}$ and no more. Because the disorder is global, a larger $L$ does **not** average over disorder; only the number of samples does. BVH used $3\times10^4$–$10^6$ disorder realizations.
   - The paper's Fig. 4 ($L=20000$, $t$ up to about $2\times10^6\approx100L$) extends into the finite-size regime in its last decade. (SUGGEST FIX) → **proposal in §7.3 (4.2.3); revisited in §7.10 after your question**
4. **Type instability.** `new_state = [0.0 ...]` is Float64 while `state` is Int, and the pointer swap alternates types every step. The results are correct but slow. `rand(L)` and `[upper_ep, lower_ep][choice+1]` also allocate every step. A typed, allocation-free kernel (e.g. `BitVector` with `rand!` into a preallocated buffer) should be several times faster. (FIX) → **done, §7.3 (4.2.4)**. *Correction after your test run: allocations fell 313 MiB → 0.5 MiB, but the speed-up is only ×1.07, so the old code was not badly slowed by them.*
5. **Stale comment.** `get_time_random_uppper_lower_binary_time_data*.jl:22` says "$\bar\varepsilon=(\varepsilon_u+\varepsilon_l)/2=0.55\varepsilon_u$". The code correctly uses $0.81\varepsilon_u$. (FIX) → **done, §7.3 (4.2.5)**
6. **Scripts on disk no longer match the data the notebooks read.**
   - `get_time_random_slidding_p_time_data.jl` has $\varepsilon_u=0.42$, `lower_div=10`, $p_c=0.4928$, `time_prefact=10000`. `analyze_random_slidding_p_rho_per_time_z2.ipynb` reads $\varepsilon_u=0.43$, `/20`, $p=0.47882$, `timepref4500`.
   - The `_time_data2.jl` script produces 4000 samples at offset 2000, but `..._rho_per_time2.ipynb` loads 20000.
   - In `get_time_random_slidding_p.jl:33-34`, `p_vals` is assigned twice, so the notebook's coarse grid must come from an earlier run.
   - Nothing records which parameters produced which files. **Fix:** write a JSON sidecar (parameters, git hash, seed) next to every data directory. (DC - this is an unfortunate side effect of doing many things)

### 4.3 Analysis notebooks
1. **`analyze_random_slidding_p_rho_per_time2.ipynb` was executed out of order.** Execution counts run cell 12→30, 13→31, 8→128, 10→157. Cell 2 defines 5 $p$ values (0.47842–0.47922, spacing 0.0002), but cell 8's saved output lists **7** values (0.47805–0.47985, spacing 0.0003).
   - The saved outputs, including $\bar\delta=0.997$ and the collapse plots, cannot be guaranteed to come from the code as saved.
   - `p_vals = set(sorted(...))` makes a set, which is unordered. So `zip(control_vals, ..., p_vals)` in cell 3 pairs mismatched values. It is only a display problem, since loading uses `control_vals`. (FIX) → **done, §7.5 (4.3.1)**
   - **Restart and run all** before trusting any number.
2. **`analyze_random_upper_lower_binary_rho_per_ep.ipynb`, cell 3:** `round(p*u+(1-p)*l)` without `ndigits` gives 0 for every entry. It only works because cell 4's plotting loop overwrites `epsilon_vals` before cell 8 uses it. (FIX) → **done, §7.5 (4.3.2)**
3. **Collapse fits (`DataCollapse`):**
   - lmfit's standard errors are meaningless here: for example $\sigma(p_c)=7\times10^{-9}$. The loss comes from sorting and nearest-neighbour interpolation, so it is piecewise and its numerical Jacobian is unreliable. Use the `bootstrapping` helper that is already in the cell but never called.
   - The 10001-point scan over initial $p_c$ records the *initial guess*, not the fitted $p_c$, and then hard-codes `best_pc`.
   - Parameters hitting their bounds (β = 0.596 against a bound of 0.6) should be treated as a failed fit.
   - FSS at a fixed $t=L^{1.45}$ bakes a power-law $z$ into the analysis.
4. **The $\ln\ln t$ extrapolation for $\bar\delta$** (all `*_rho_per_time*` notebooks and `analyze_sdiff_per_time3` cell 9) fits $\ln\rho/\ln\ln t$ against $X=1/\ln\ln t$ and reads $-\bar\delta$ from the intercept at $X=0$.
   - Over the data, $X$ spans only about 0.37–0.44 (Stavskaya) or 0.45–0.65 (spin chain). The extrapolation is 5–10× the data range, so $\bar\delta$ is extremely sensitive to noise and to any additive constant.
   - BVH's form is $\rho^{-1}=A+B\ln t$. It is linear in $\ln t$, needs no extrapolation, and absorbs the non-universal time scale. (SUGGEST IMPLEMENTATION) → **proposal in §7.5 (4.3.4); implemented 2026-09-25, §7.9**
   - In the Stavskaya cells, `positive` used for `crit_key` is the mask left over from the last loop iteration. (FIX) → **done, §7.5 (4.3.4)**
5. **`analyze_sdiff_per_time3.ipynb`, cells 10–11** (stale; execution count `None`):
   - The ratio uses `data[T//b-1]/data[T-1]`, i.e. $S(T/b-1)/S(T-1)$ with off-by-one indices. At $T\to0$ it hits `data[-1]`, the last element. (FIX) → **done, §7.5 (4.3.5)**
   - The x-axis uses `time_vals`, which is not defined in those cells and has a different length from the y data. (FIX) → **done, §7.5 (4.3.5)**
   - Cell 16, which makes the figure, does it correctly. Delete 10–11. (FIX) → **cell 10 fixed and kept, cell 11 deleted (your choice), §7.5 (4.3.5)**
6. **Collapses by eye.** The "± 0.15", "± 0.0003" etc. in the markdown of `analyze_sdiff_per_time3` are eyeball estimates from manual collapses (cells 12–16). There is no fit and no error propagation. (DC)
7. **Mislabelled axes and legends.** (FIX) → **done, §7.5 (4.3.7)**
   - `ylabel("1/log(1-ρ)")` where $1/(1-\rho)$ is plotted.
   - `xlabel("1/t")` where $\ln(1/t)$ is plotted.
   - Legends say "p =" for $\bar\varepsilon$.
   - `analyze_*_z` notebooks take $\log(0)$ once all samples are absorbed.

---

## 5. Recommended path to resolve the conflict

Framing: the question is not which exponents you get. It is **whether the effective exponents drift with scale, and in which direction**. BVH give sharp, parameter-free tests.

1. **Running exponents over as many decades as possible** (Stavskaya). Use log-spaced output, $L\ge t_{\max}$, and $\ge10^4$ disorder realizations.
   - Plot $1/\delta_{\rm eff}$ vs $\ln t$. It is linear at infinite-noise criticality (BVH Fig. 5 inset) and constant at a power-law point.
   - Plot $1/\rho$ vs $\ln t$. It is a straight line at criticality (BVH Fig. 6).
2. **Crossing times off criticality.** Define $t_x(r)$ as the time where an off-critical $1/\rho$ curve leaves the critical line. BVH predict $\ln t_x\propto r^{-1/2}$ (Fig. 7 inset); a power-law point gives $\ln t_x\propto\nu_t\ln(1/r)$. This test is independent of $\delta$.
3. **Distribution width.** At each $t$, take the per-sample values $x=-\ln(1-\rho)$ (Stavskaya) or $x=-\ln S_{\rm diff}$ (spin chain) and plot ${\rm std}(x)$ vs $\ln t$.
   - Infinite noise gives linear growth. A finite-disorder fixed point gives a width that saturates.
   - For large $L$ each sample is effectively one disorder realization. **You can do this now with the existing per-sample CSVs** on the external drive.
4. **Spreading runs from a single seed.** Measure $P_s(t)$, $N_s(t)$ and $R(t)$. For DP-type problems these are the cleanest. They are free of finite-size effects and test $z=1$ with log corrections directly.
5. **Tune disorder strength.** Vary the block length $\Delta t$ and the contrast $\varepsilon_u/\varepsilon_l$ (Stavskaya). For the spin chain, hold the random $J$-signs fixed over blocks of $\Delta t>1$, or make $a(t)$ binary-random.
   - If the "exponents" move with disorder strength, they are crossover values.
   - If strong disorder produces clean logarithmic scaling earlier, that is BVH directly.
6. **Temporal Griffiths.** On the chaotic side, measure the lifetime $\tau(L)$. BVH predict $\tau\sim L^{1/\kappa}$ with $\kappa$ varying continuously, not exponential growth. This would also explain the FSS $z$.
7. **Rewrite the claim.** The robust, novel content is:
   - a classical Hamiltonian control transition with DP-like activity;
   - a **discontinuous Lyapunov exponent**;
   - critical behaviour that is **consistent with, and crossing over toward, the temporally-disordered infinite-noise fixed point**.

   The Lyapunov jump story does not depend on the universality class. It may even be more natural in the infinite-noise picture, where rare, strongly chaotic temporal regions dominate. Cite BVH, and replace "new universality class with $\nu_t\approx2.25$" with an explicit crossover analysis.

---

## 6. Files reviewed

- **`stavskya_mc/` scripts:**
  - `get_time_random_uppper_lower_binary{,_time_data,_time_data2}.jl`
  - `get_time_random_slidding_p{,_time_data,_time_data2}.jl`
  - `utils/{general,calculations,dynamics}.jl`
- **`stavskya_mc/` notebooks:**
  - `analyze_random_upper_lower_binary_rho_per_{ep,time,time_z}.ipynb`
  - `analyze_random_slidding_p{,_rho_per_time2,_rho_per_time_z2}.ipynb`
- **`heisen_spin_chain/`:**
  - `lyapunov_exponents/get_sdiff_data_severalL0.jl` (and L1–3, which differ only in offset)
  - `utils/{make_spins,general,dynamics,lyapunov}.jl`
  - `analytics/spin_diffrences.jl`
- **`python_code/`:** `analyze_sdiff_per_time3.ipynb`
- **Data re-analysed (§3.3):** `data/s_diff_per_time/N4/*/IC1700/L2000/*mean.pickle`. Per-time Stavskaya data live on `/Volumes/ExternalData` and were not accessible, so the §3.2 numbers come from the notebooks' saved outputs.

---

## 7. Follow-up on your annotations to §4 (2026-09-25)

What each annotation meant (confirmed with you before starting):
- **FIX**: I edited the project files.
- **SUGGEST FIX / SUGGEST IMPLEMENTATION**: a tested proposal is written below, and the project files are unchanged.
- **DC**: left alone.

§4.3.3 and two bullets of §4.3.4 had no annotation, so I left the old notebooks alone on those points. The new notebooks do use bootstrap errors (§7.4).

### 7.0 Summary

| § | annotation | what was done | where |
|---|---|---|---|
| 4.1.1 | I FIXED, PLEASE CONFIRM | Confirmed correct. One leftover copy in an unused old script. | `heisen_spin_chain/utils/make_spins.jl` |
| 4.1.2–4.1.4 | SUGGEST FIX | Proposal: `random_global_control_sdiff`, which keeps O(L) memory, records true times starting at 0, and leaves the caller's `J_vec` alone. A Julia equivalence test is included. | §7.2; **implemented 2026-09-25 (§7.9)** |
| 4.1.4 (1st, 3rd bullet), 4.1.5 | DC | untouched | – |
| 4.2.1 | FIX | `block_len` keyword (default 1 = old model) on every time-random evolve function. It is a looped knob, and it appears in every new file name. | `stavskya_mc/utils/dynamics.jl`, `stavskya_mc/block_disorder/` |
| 4.2.2 | FIX | New log-time generators, submit scripts and analysis notebooks. `time_log` is in every file name, and the data go in their own directory. | `stavskya_mc/block_disorder/`, data → `stavskya_mc/data/time_log/` |
| 4.2.3 | SUGGEST FIX | Proposal plus a numerical check of the light-cone statement. | §7.3 |
| 4.2.4 | FIX | Type-stable, allocation-free kernel, bit-identical to the old one (test provided). | `stavskya_mc/utils/dynamics.jl` |
| 4.2.5 | FIX | Stale comment corrected. | `get_time_random_uppper_lower_binary_time_data{,2}.jl:22` |
| 4.2.6, 4.3.6 | DC | untouched | – |
| 4.3.1 | FIX | `set(sorted(...))` → `sorted(set(...))` | `analyze_random_slidding_p_rho_per_time2.ipynb` cell 2 |
| 4.3.2 | FIX | `round(..., ndigits=6)` | `analyze_random_upper_lower_binary_rho_per_ep.ipynb` cell 3 |
| 4.3.4 | SUGGEST IMPLEMENTATION | Drop-in BVH fit cell, tested on your spin-chain data. | §7.5; **implemented 2026-09-25 (§7.9)** |
| 4.3.4 | FIX | `positive` mask now computed for `crit_key` itself. | upper/lower `rho_per_time` cell 8; sliding-p `rho_per_time2` cell 13 |
| 4.3.5 | FIX | Cell 10 rewritten as a correct α_eff plot; cell 11 deleted. | `python_code/analyze_sdiff_per_time3.ipynb` |
| 4.3.7 | FIX | Axis and legend labels; `log(0)` masked in the `_z` notebooks. | 4 Stavskaya notebooks, listed in §7.5 |

All notebook edits were made by one script, `review_2026_09/apply_review_4_3_notebook_fixes.py`. It keeps a record of every change and asserts that each old string appears exactly as expected before replacing it. Only cell sources changed. The saved outputs of the edited cells are the old ones, except cell 10 of `analyze_sdiff_per_time3`, which was cleared. Re-run the notebooks to refresh them. Diffs: `git diff -- '*.ipynb'`.

### 7.1 How things were verified, and what could not be run here

**Julia could not be run in my sandbox.** The julialang.org download and package servers are blocked by the egress policy, both in the cloud sandbox and in your computer's VM. Your Terminal could only be granted click-only access, so I could not type into it either. The Julia work was checked in three ways:
1. Every new or edited `.jl` file parses cleanly (tree-sitter Julia grammar).
2. The logic was checked against a line-by-line Python port of the new kernel and recorder.
3. **Julia test files** are included. **You ran them on 2026-09-25 (Julia 1.12.6, macOS arm64): all 678 checks passed, and the local generator smoke run completed** (see §7.7). The commands were:

```bash
cd ~/research/senior_thesis
julia --project=. stavskya_mc/block_disorder/tests/test_dynamics.jl        # kernel refactor, block_len, recorder, log grid
STAV_LOCAL_TEST=1 julia --project=. stavskya_mc/block_disorder/generators/get_slidding_p_time_log.jl   # smoke-runs a generator
python -m pytest stavskya_mc/block_disorder/tests/test_time_log_tools.py   # Python estimators (9 tests, all pass here)
```
Every generator has the `STAV_LOCAL_TEST=1` mode. It uses 2 local workers, tiny L and 4 samples, writes to the git-ignored `block_disorder/_local_test_output/` (changed from a temp directory on 2026-09-25, see §7.7), and doesn't need Slurm or SlurmClusterManager. If `test_dynamics.jl` fails its "bit-identical" test set, the refactor changed results, and you should not use it for production until we look at it. The most likely cause would be `rand!(u)` and `rand(L)` drawing the random stream differently in your Julia version.

**Python** (the new notebooks, `time_log_tools.py`, and the edited notebook cells) was actually executed:
- `test_time_log_tools.py`: 9/9 pass. It checks exact recovery of δ from a power law, α from $(\ln t)^{-\alpha}$, and $(a,B)$ from $1/A=a+B\ln t$; the $1/\ln t$ running-δ numbers quoted in §3.1; the path strings against `naming.jl`; and the loader.
- All three new notebooks were run end to end with papermill on synthetic data in the new file layout. The data came from the Python port, at L = 32–512 and 60–200 samples. No errors.
- Edited old cells were executed against synthetic inputs, and against your spin-chain pickles for cell 10 (see §7.5).

### 7.2 Spin chain (`heisen_spin_chain/`)

**4.1.1 (confirmed).** Commit `f764600` changed `make_random_state` and `make_random_spin` to `normalize(randn(3))`. A normalized isotropic Gaussian vector is exactly uniform on the sphere, so the fix is correct. Three side notes:
- `make_random_spin(epsilon)` is also used for the Lyapunov perturbation of the middle spin (`get_good_data_severalL*.jl:78`). Before the fix, the kick always pointed into the (+,+,+) octant. Now it has a uniformly random direction and magnitude ε. The Supplemental text ("randomize each direction within the window [−ε, ε]") describes neither version and should be reworded.
- `heisen_spin_chain/my_own_numerics.jl:35` still has `normalize(rand(3))`. Nothing includes that file, so it does not matter. Left alone.
- The same commit set `init_cond_name_offset = 0` in `get_good_data_severalL{,2,3,4,5}.jl`. That is safe because each script uses a different L (8, 16, 32, 64, 128 unit cells). L5 and L6 share L = 128 but have offsets 0 and 500, so their files do not collide either.

**4.1.2–4.1.4 (proposal, not applied).** A single new function fixes the off-by-one time column (4.1.2), the O(L·T) memory (4.1.3) and the mutation of the global `J_vec` (4.1.4). It draws the same random numbers in the same order as `random_global_control_evolve`, so results are identical for a fixed seed.

Memory for comparison: the current path stores n+1 flattened states and then an unflattened copy. For L = 2000 and n = 20L that is about 2 GB of Float64 plus about 80 million small 3-vectors, several GB per sample. The proposal keeps one state.

```julia
# add to heisen_spin_chain/utils/dynamics.jl
"""
    random_global_control_sdiff(L_J_vec, original_state, a_val, T, t_step, s_0; record_every=1)

Same dynamics as `random_global_control_evolve`: the same random numbers are drawn in the
same order, so a fixed seed gives an identical trajectory. The difference is that only the
current state is kept, and S_diff is measured on the fly at t = 0, record_every*t_step, ...
Returns `(times, s_diff)`, where `times` are the true times of the recorded states
(t = 0 is the initial state). Memory is O(L) instead of O(L*T).

`L_J_vec` is copied, so the caller's vector is never modified (review 4.1.4).
"""
function random_global_control_sdiff(L_J_vec, original_state, a_val, T, t_step, s_0;
                                     record_every::Integer=1)
    J_signs = copy(L_J_vec)
    current_u = flatten_state(original_state)
    n_steps = div(T, t_step)            # the old loop ran for t = t_step, 2t_step, ..., <= T
    times = Float64[]
    s_diff = Float64[]
    push!(times, 0.0)
    push!(s_diff, weighted_spin_difference(unflatten_state(current_u), s_0))
    t = t_step
    for k in 1:n_steps
        J_signs[1] *= (rand() > 0.5) ? -1 : 1   # random signs of J_x, J_y (removes solitons)
        J_signs[2] *= (rand() > 0.5) ? -1 : 1
        current_u = evolve_spin(J_signs, current_u, (t, t + t_step))
        current_u = flatten_state(global_control_push(unflatten_state(current_u), a_val, s_0))
        t += t_step
        if k % record_every == 0
            push!(times, k * t_step)
            push!(s_diff, weighted_spin_difference(unflatten_state(current_u), s_0))
        end
    end
    return times, s_diff
end
```
In `get_sdiff_data_severalL{0,1,2,3}.jl`, lines 75–82 then become:
```julia
                times, current_sdiffs = random_global_control_sdiff(J_vec, spin_chain_A, a_val, n, J, S_NAUGHT;
                                                                     record_every=step_size)
                sample_filepath = ...                      # unchanged
                make_path_exist(sample_filepath)
                CSV.write(sample_filepath, DataFrame(t = times, s_diff = current_sdiffs))
```
The CSV `t` column then holds true times (0, s, 2s, …, up to n), so the notebook's own reconstruction and the column agree.

Test (run from `senior_thesis/` after adding the function):
```julia
using Test, Random, LinearAlgebra, Statistics, DifferentialEquations, SymPy
include("heisen_spin_chain/utils/make_spins.jl"); include("heisen_spin_chain/utils/general.jl")
include("heisen_spin_chain/utils/dynamics.jl");   include("heisen_spin_chain/analytics/spin_diffrences.jl")
L, T, step, a = 16, 60, 7, 0.75
S0 = make_spiral_state(L, 0.5)
@testset "on-the-fly S_diff == old store-everything path" begin
    for seed in 1:3
        Random.seed!(seed); s = make_random_state(L)
        Random.seed!(100 + seed)
        old = weighted_spin_difference_vs_time(random_global_control_evolve([1, 1, 1], s, a, T, 1, S0), S0)
        Random.seed!(100 + seed); J = [1, 1, 1]
        times, new = random_global_control_sdiff(J, s, a, T, 1, S0; record_every=step)
        @test J == [1, 1, 1]                      # caller's vector untouched
        @test times == collect(0:step:T)          # true times, starting at 0
        @test new == old[1:step:end]              # old[k] is the state at t = k - 1
    end
end
```
New data would not match old data sample for sample. Old runs shared one mutated `J_vec` across samples on a worker, so each sample started from whatever sign the previous one ended with. The statistics are the same, because a random sign times anything is a random sign.

### 7.3 Stavskaya core code (`stavskya_mc/utils/`, old generators)

**4.2.4 + 4.2.1: `utils/dynamics.jl` rewritten.** The pre-refactor file is frozen as `block_disorder/tests/reference_dynamics_pre_refactor.jl`.
- One kernel, `stavskaya_step!(new, cur, u, eps)`, is shared by all seven evolve functions. `u` is a preallocated Float64 buffer filled with `rand!`, the state keeps its element type (Int), and nothing is allocated per step.
- Each step draws the per-site uniforms with `rand!(u)` into the buffer. `rand(L)` fills a fresh vector the same way, and the scalar ε draw happens first, in the same order as before. So for a fixed seed every function is **bit-identical** to the old code. `test_dynamics.jl` checks this for all functions, several L, T and seeds, and also checks that the RNG state afterwards is identical.
- Every time-random function gained `block_len::Integer=1`. ε is redrawn only when `(t-1) % block_len == 0`. `block_len = 1` is the old model, so every old generator and notebook is unaffected.
- New `time_random_p_record_rho(state, record_times, ε_u, ε_l, p; block_len)` evolves without interruption and records ρ at arbitrary sorted times, t = 0 included. Chunked calls would restart the block phase at every chunk; this one keeps disorder blocks aligned to absolute time. Once the chain is absorbed it fills the remaining entries with 1.0 and stops early, which saves time in the finite-size tail.
- Only other behaviour change: the returned state is always the input's element type. The old code alternated between Int and Float64 depending on whether `time_steps` was odd or even.
- **Measured on your machine** (L = 20000, T = 2000): old 0.103 s and 313 MiB allocated, new 0.096 s and 0.5 MiB. That is a ×1.07 speed-up, much less than the "several ×" I first estimated. The main gains are the removed allocations (less GC pressure with many workers per node) and a type-stable state. Bit-identity was confirmed: 678/678 tests passed.

**`utils/general.jl`:** added `make_log_times(t_max; points_per_decade=20)`. It returns 0, then all integers up to about 10, then about 20 log-spaced times per decade, ending exactly at `t_max` (111 times for `t_max = 10^6`).

**4.2.5:** the comment at line 22 of both `get_time_random_uppper_lower_binary_time_data*.jl` now reads $\bar\varepsilon = p\varepsilon_u+(1-p)\varepsilon_l = (p+(1-p)/\text{lower\_div})\,\varepsilon_u = 0.81\,\varepsilon_u$.

**4.2.3 (proposal, not applied): run length versus L.**
- **Proposal:** for bulk decay runs, choose `time_prefact` ≤ 1, i.e. $t_{\max}\le L$. Spend the compute saved on more disorder realizations, $\gtrsim10^4$ at each control value. For finite-size scaling, use the dedicated `*_fss` generators and keep their late times.
- **Check of the underlying claim.** I simulated the upper/lower model at $\bar\varepsilon_c=0.27033$ with the Python port of the kernel, 20,000 samples per size, comparing ⟨A(t)⟩:
  - L = 8 vs L = 512: |difference|/SE ≤ 1.6 for all t ≤ 2L (t = 2, 4, 6, 7, 8, 12, 16), then −11.5 at t = 4L and −36 at t = 8L.
  - L = 32 vs L = 512: |difference|/SE ≤ 1.3 up to t = 5L at this small correlation length.

  So t < L is always safe, as the light-cone argument says. How far beyond L you can go depends on how far the correlation length has grown, which is why the paper's last decade (t ≈ 100L) is at risk.
- **Built-in check:** the new FSS notebook tests light-cone agreement between sizes automatically.

### 7.4 New log-time / block-disorder pipeline (4.2.1, 4.2.2): `stavskya_mc/block_disorder/`

```
block_disorder/
  generators/   get_{upper_lower_binary,slidding_p}_time_log.jl       several control values, one L (successors of *_time_data.jl)
                get_{upper_lower_binary,slidding_p}_time_log_fss.jl   one control value, several L   (successors of *_time_data2.jl / the z2 data)
                get_{upper_lower_binary,slidding_p}_rho_per_ep_block.jl  rho at t=L^z scans with block_len (successors of get_time_random_*.jl)
                setup_workers.jl   Slurm workers, or local test mode with STAV_LOCAL_TEST=1
                naming.jl          every path and file name (mirrored in analysis/time_log_tools.py)
  submit/       submit_*.sh        same SBATCH headers as the old ones; submit from this directory
  analysis/     analyze_time_log.ipynb          decay: running delta, 1/delta vs ln t, 1/A vs ln t, alpha_eff, fit comparison,
                                                width of P(-ln A) over disorder, crossing times (BVH tests of section 5)
                analyze_time_log_fss.ipynb      several L: light-cone check, power-law vs BVH (z=1) collapses, lifetimes -> z_eff = 1/kappa
                analyze_rho_per_ep_block.ipynb  rho(t=L^z) collapse with bootstrap errors and a parameter-at-bound check
                time_log_tools.py               loaders + estimators that work on any (log) time grid
                data_collapse.py                your DataCollapse class, copied verbatim (only applymap -> map for pandas >= 2.1)
  tests/        test_dynamics.jl, reference_dynamics_pre_refactor.jl, test_time_log_tools.py,
                run_all_tests.sh + check_local_test_output.py (added 2026-09-25, see 7.7)
  _local_test_output/   written by STAV_LOCAL_TEST=1 runs (git-ignored)
```
- **Knobs:** the generators keep the old knobs, names and default values. Added knobs are `block_len_vals` (looped over, like `L_vals`) and `points_per_decade`. `time_prefact` is still $T_f/L$.
- **The critical point depends on `block_len`.** The defaults are the block_len = 1 estimates; rescan before trusting a critical estimate at block_len > 1.
- **File names:** every log-time file contains `blocklen<b>` and `time_log`:
  - log time: `data/time_log/<model>/rho_per_time/IC1/L<L>/epsilonu<u>/epsilonl<l>/pval<p>/blocklen<b>/IC1_L<L>_epsilonu<u>_epsilonl<l>_pval<p>_blocklen<b>_timepref<tp>_ppd<ppd>_time_log_sample<k>.csv`
  - block rho-per-ep: `data/block_rho_per_ep/<model>/rho_per_epsilon/IC1/L<L>/IC<n>_L<L>_..._blocklen<b>_z<z>.csv`
- **Output paths are absolute**, built from the script's own location. They land in `<repo>/stavskya_mc/data/...` on the cluster, not in the doubled `stavskya_mc/stavskya_mc/data` the old scripts produce when launched from `stavskya_mc/`. Point `ssh_transfer` at `.../senior_thesis/stavskya_mc/data/time_log`.
- **Number formatting:** Julia's `string(Float64)` and Python's `str(float)` only agree between 1e-4 and 1e5 (see the note in `naming.jl`). All current parameters are inside that range, and `float_str` in Python warns if one is not.
- **Notebooks** take their parameters from one tagged cell, so they also run headless:

  ```bash
  papermill analyze_time_log.ipynb out.ipynb -y "{MODEL: slidding_p, L: 35000, ...}"
  ```

  They read the new data only. The old linear-time notebooks are unchanged apart from §7.5.

### 7.5 Existing analysis notebooks (4.3)

Made by `review_2026_09/apply_review_4_3_notebook_fixes.py`, with the same result on your files as on my tested copies (identical md5).

- **4.3.1** `analyze_random_slidding_p_rho_per_time2.ipynb` cell 2: `p_vals = sorted(set([...]))`, plus the two commented variants. Now `zip(control_vals, …, p_vals)` pairs correctly, and `p_vals` is an ordered list.
- **4.3.2** `analyze_random_upper_lower_binary_rho_per_ep.ipynb` cell 3: `round(..., ndigits=6)`, which now gives `[0.27033, …]` instead of `[0, …]`.
- **4.3.4 (FIX)** `positive` recomputed for `crit_key` in `analyze_random_upper_lower_binary_rho_per_time.ipynb` cell 8 and `analyze_random_slidding_p_rho_per_time2.ipynb` cell 13.
  - Test: synthetic data where the critical curve is absorbed from point 50 on but the last control value never is. The old cell prints `alpha = nan`; the fixed cell prints `alpha = 1.0000` for input $A\propto1/\ln t$.
- **4.3.5** `python_code/analyze_sdiff_per_time3.ipynb`: cell 10 rewritten, cell 11 deleted, so the old cells 12–16 are now 11–15.
  - The new cell 10 computes $\alpha_{\rm eff}(T)=\ln[S(T/b)/S(T)]/\ln[\ln T/\ln(T/b)]$ with correct indices (`S[k]` is the value at $t=k\,\Delta t$). It only uses $T/b\ge e$, so the denominator never approaches 0 and `data[-1]` is never reached, and it plots against its own T.
  - It was executed on your local pickles (L = 2000, 1700 samples). It matches a brute-force loop exactly and matches `time_log_tools.running_alpha` to $2\times10^{-15}$.
  - At a = 0.758–0.7582, α_eff rises slowly, from about 0.36 at t = 10 to 0.76–0.95 at t ≈ 10⁴. On the controlled side (a = 0.7535) it climbs past 10. On the chaotic side (a = 0.7615) it falls to 0.1–0.3.
- **4.3.7** labels and `log(0)`:
  - `1/log(1-ρ)` → `1/(1-ρ)` in both `rho_per_time` notebooks.
  - `1/t` → `ln(1/t)` where the x data are `ln(1/t)`.
  - Upper/lower legends `p = …` → $\bar\varepsilon$ = … (`control_label`).
  - Sliding-p collapse x-labels $(\varepsilon-\varepsilon_c)$ → $(p-p_c)$.
  - In `analyze_random_upper_lower_binary_rho_per_time_z.ipynb` cells 6 and 9 and `analyze_random_slidding_p_rho_per_time_z2.ipynb` cells 5, 6 and 9, the mask is now `(times > 0) & (1 - mean_rhos[key] > 0)`. Re-run with fully absorbed samples, these cells raise no divide-by-zero warnings.
  - The α-ratio cells (upper/lower cell 9, sliding-p cell 14) still divide by zero once every sample is absorbed. They were not in your list, so they are unchanged.
- **Not done:** "Restart and run all" (4.3.1, third bullet, unannotated). The per-time data are on `/Volumes/ExternalData`, which I cannot reach. Please do it before quoting any number from these notebooks again.

**4.3.4 (SUGGEST IMPLEMENTATION): BVH fit cell for the old notebooks.** Paste it after the cell that builds `time_vals`, `mean_rhos` and `sem_rhos`. It fits $1/A=a+B\ln t$ over a window, with no extrapolation, next to a power law over the same window, and plots the local slope $d(1/A)/d\ln t$:
```python
fit_tmin, fit_tmax = 2e4, 2e6          # a window inside the data (ideally t < L)
rows = []
for L_val in L_vals:
    for control_val in control_vals:
        key = (L_val, control_val)
        t = np.asarray(time_vals[key], float); A = 1 - np.asarray(mean_rhos[key]); sA = np.asarray(sem_rhos[key])
        m = (t >= fit_tmin) & (t <= fit_tmax) & (A > 0) & (sA > 0)
        x = np.log(t[m])
        y, sy = 1 / A[m], sA[m] / A[m]**2                          # BVH: 1/A, error sigma_A/A^2
        (B, a), cov = np.polyfit(x, y, 1, w=1/sy, cov="unscaled")
        chi2_bvh = np.sum(((y - (a + B*x)) / sy)**2) / (m.sum() - 2)
        (slope, c), _ = np.polyfit(x, np.log(A[m]), 1, w=A[m]/sA[m], cov="unscaled")   # power law, same window
        chi2_pow = np.sum(((np.log(A[m]) - (c + slope*x)) / (sA[m]/A[m]))**2) / (m.sum() - 2)
        rows.append({"L": L_val, "control": control_val, "B": B, "B_err": np.sqrt(cov[0, 0]), "ln t0 = a/B": a / B,
                     "chi2r BVH": chi2_bvh, "delta (power law)": -slope, "chi2r power": chi2_pow, "points": int(m.sum())})
print("chi2r ignores correlations between times: compare the two columns, do not read them absolutely")
display(pd.DataFrame(rows).round(4))

fig, ax = plt.subplots(figsize=(8, 5))                 # local slope over a factor 2 in t
for L_val in L_vals:
    for control_val in control_vals:
        key = (L_val, control_val)
        t = np.asarray(time_vals[key], float); A = 1 - np.asarray(mean_rhos[key])
        ok = (t > 0) & (A > 0); t, A = t[ok], A[ok]; T = t[t / 2 >= t[0]]
        ax.plot(np.log(T), (1/A[np.searchsorted(t, T)] - np.interp(np.log(T/2), np.log(t), 1/A)) / np.log(2), label=f"{control_val}")
ax.set_xlabel(r"$\ln t$"); ax.set_ylabel(r"$d(1/A)/d\ln t$"); ax.legend(fontsize=8); plt.show()
```
Tests:
- **Synthetic data:** it recovers $B$ exactly (0.300 and 0.250 for exact BVH inputs) and δ = 0.1000 with $\chi^2_\nu=0$ for an exact power law.
- **Your spin-chain pickles, window $t\in[50,5000]$.** For the spin chain, use `collected_sdiffs` as $A$ and `collected_sdiff_stds/sqrt(1700)` as the SEM:

  | a | B | ln t0 | χ²ν BVH | δ (power law) | χ²ν power |
  |---|---|---|---|---|---|
  | 0.7574 | 0.214 | 2.35 | 1.60 | 0.114 | 0.65 |
  | 0.7578 | 0.203 | 2.61 | 1.21 | 0.110 | 0.44 |
  | 0.7580 | 0.183 | 3.51 | 1.43 | 0.100 | 0.88 |
  | 0.7582 | 0.166 | 4.35 | 0.83 | 0.092 | 0.73 |
  | 0.7595 | 0.096 | 10.86 | 1.59 | 0.056 | 1.90 |

  This agrees with §3.3: on this window the two forms cannot be told apart.

### 7.6 Housekeeping

- **Stale git lock (my fault).** One of my read-only `git status` calls left a stale, empty `.git/index.lock` in your repository. My session cannot delete files, so I moved it to `_to_delete/stale_git_index.lock`; after that, `git status` works. You have since deleted the `_to_delete/` folder. I now use `git --no-optional-locks` for read-only calls.
- **Nothing committed.** All changes are uncommitted in your working tree. `git status` shows them, and `git diff` shows exactly what changed in existing files.

### 7.7 Test results, and how I can run the Julia tests myself next time (2026-09-25)

**Your run (Julia 1.12.6, macOS arm64):**
- `test_dynamics.jl`: **678/678 passed.** The new kernel is bit-identical to the old one for every evolve function when both run on the same Julia version. Switching the cluster runs to the new code therefore changes nothing in the physics or the statistics. The block_len, recorder and log-grid tests passed too.
- **Speed:** ×1.07, with allocations down from 313 MiB to 0.5 MiB (numbers in §7.3).
- **Local generator run:** `get_slidding_p_time_log.jl` completed for p = 0.4787 / 0.479 / 0.4793 and block_len 1 and 3, with 40 output times at $T_f=256$. That matches `make_log_times(256)`.
- The only other output was juliaup's "1.13 is available" notice. Julia does not promise the same random-number stream across versions. So reproducing an old data file *sample for sample* needs the cluster's 1.11.6; statistically, the version makes no difference.

**Why I could not run it myself:**
1. **Julia downloads are blocked.** Your account's network-egress policy blocks `julialang-s3.julialang.org` (Julia binaries) and `pkg.julialang.org` (packages), both in my cloud sandbox and in the Linux VM on your computer. By policy I don't route around a block, for example by fetching Julia from a mirror.
2. **Terminal is click-only.** Computer use can see Terminal and IDEs and click in them, but it cannot type or press keys there. Typing a command is exactly what running Julia needs.

**Option A: allow the Julia domains (lets me run everything myself).** In the Claude settings where network egress is configured, add the Julia hosts to the allowed domains. On an individual plan that is Settings → Capabilities; on a Team or Enterprise plan it is Admin settings → Capabilities, owner only. The menu wording may differ slightly. Hosts:

```
julialang-s3.julialang.org      # Julia binaries (and juliaup's version list)
pkg.julialang.org               # package server
*.pkg.julialang.org             # regional package servers it redirects to (e.g. us-east.pkg.julialang.org)
```
Next session I would then download Julia and run `test_dynamics.jl`, the local generator runs and the Python checks myself, reading and writing your files directly. I'd install Julia inside my sandbox, never into your `~/.julia`. The sandbox is wiped between sessions, so each session pays a one-off setup cost:
- about 1–2 minutes for Julia, `Distributions`, `CSV` and `DataFrames` (all the Stavskaya tests need);
- noticeably longer, roughly 10+ minutes of precompilation, for `DifferentialEquations` (spin-chain tests).

This is the only option where I can check new Julia code *before* handing it to you.

**Option B: one command that writes a log I can read (works now, no settings change).** I added `stavskya_mc/block_disorder/tests/run_all_tests.sh`:
```bash
bash stavskya_mc/block_disorder/tests/run_all_tests.sh
```
It runs, in order:
1. `test_dynamics.jl`;
2. **all six** generators in local mode;
3. `tests/check_local_test_output.py`. This new check has the Python loaders look for exactly the files Julia wrote, using the parameters the notebooks rebuild, so it tests the Julia↔Python file-name hand-off for real, which nothing has tested so far;
4. the Python unit tests, if `pytest` is installed.

Everything is saved to `stavskya_mc/block_disorder/tests/last_test_run.log`, ending in `OVERALL: PASS/FAIL`. Just tell me you ran it: I can read the log and the generated files straight from your folder, so no copy-pasting. To support this, local-test output now goes to `stavskya_mc/block_disorder/_local_test_output/` instead of a macOS temp directory I can't see. That folder and the log are git-ignored through the new `block_disorder/.gitignore`.

How the pieces were checked here:
- **The runner:** I ran it with a stand-in `julia` that exits with chosen codes. Exit codes, the PASS/FAIL lines and the log all behave; one failing step makes the overall result `FAIL`.
- **The checker, on a complete set of files:** I generated all 96 expected local-test files in the new layout with a Python port and ran the checker. It reports 96/96 found and passes.
- **The checker, on bad files:** it catches one missing file and one mis-named file (a `4p79e-1` float format) and exits with an error.
- **On your computer:** the checker runs with your Python and, as expected before any Julia run, reports the local-test files missing.

**Recommendation:** use B now, since it already removes the copy-paste step. Add A if you want me to verify Julia changes end to end before you see them.

### 7.8 Open questions (answered 2026-09-25)

1. **DataCollapse (§4.3.3).** You said it doesn't matter: those collapses only help you guess $p_c$, and they assume power-law scaling anyway. Left unchanged.
2. **α-ratio cells.** You asked what this meant.
   - **Which cells.** Cell 9 of `analyze_random_upper_lower_binary_rho_per_time.ipynb` and cell 14 of `analyze_random_slidding_p_rho_per_time2.ipynb`. Both plot the running log-exponent $\ln[Q(t/b)/Q(t)]\,/\,\ln[\ln t/\ln(t/b)]$, where $Q=1-\rho$ is the mean activity.
   - **What goes wrong.** If every sample at some control value is absorbed before $t_{\max}$, then from that time on $Q(t)=0$. The ratio $Q(t/b)/Q(t)$ becomes $x/0$ (then $0/0$), NumPy prints "divide by zero" warnings, and the curve runs to $\infty$ or NaN at late times.
   - **When it happens.** This does not happen at the near-critical values with $L=35000$ and $t\le100L$. It can happen on the inactive side, or at small $L$.
   - **The fix I'd suggest.** It is the same one already applied to the `_z` notebooks (§7.5, 4.3.7): compute the ratio only where $Q>0$ at both $t/b$ and $t$, so the curve simply ends at the last time with surviving activity. The new `analyze_time_log.ipynb` already does this.
   - **Question for you:** should I add it to those two cells? → **Yes; done 2026-09-25 (§7.11).**
3. **Spin-chain proposal.** Implemented; see §7.9.

### 7.9 Spin chain: proposals implemented, `record_every` knob, notebooks (2026-09-25)

Your decisions (asked before starting):
- regenerated data go to new `_v2` folders;
- with `record_every > 1`, the `lambda` column holds the **block average**;
- `analyze_sdiff_per_time3.ipynb` is updated too.

All old spin data are superseded because of the initial-condition bug, so nothing below tries to stay compatible with them.

**Julia code** (`heisen_spin_chain/`):

| file | change | why |
|---|---|---|
| `utils/dynamics.jl` | `random_global_control_evolve`: docstring states that element k is time (k-1)·t_step; it now works on a **copy** of `L_J_vec`. Trajectory unchanged for a fixed seed. | 4.1.2, 4.1.4 (you asked me to check it for the same problems) |
| `utils/dynamics.jl` | **new** `random_global_control_sdiff(...; record_every)`: same random numbers as above, but S_diff is measured on the fly (O(L) memory). Returns true times t = 0, k, 2k, …, n. | 4.1.2, 4.1.3 |
| `utils/dynamics.jl` | `random_evolve_spins_to_time`: sign randomization now keeps \|J_x\|, \|J_y\|. Before, the entries were *set* to ±1, which silently dropped J ≠ 1. Identical for your J = 1. | same kind of issue as 4.1.4, found while checking |
| `utils/lyapunov.jl` | **new** `benettin_lambda_sdiff(...; record_every)`: the Benettin loop that used to be inline in `get_good_data_severalL*.jl`, moved into a function so it can be tested. Works on a copy of `J_vec`. | 4.1.4, testability |
| `lyapunov_exponents/get_good_data_severalL{,2,…,6}.jl` | Call `benettin_lambda_sdiff`. **New knob** `@everywhere record_every = 1` (linear time, every k-th step written), which appears in the file name as `_timestep<k>`. Output to `data/spin_dists_per_time_v2/`. Includes made robust (`@__DIR__`). New `SPIN_LOCAL_TEST=1` mode. | your request; v2 folders |
| `lyapunov_exponents/get_sdiff_data_severalL{0,1,2,3}.jl` | Use `random_global_control_sdiff(...; record_every=step_size)`. The CSV `t` column now holds true times 0, step, …, n (n itself included). Output to `data/s_diff_per_time_v2/`. Same local mode and include change. | 4.1.2–4.1.4 |

**File format of the new `get_good_data` output:**
- columns `t, lambda, delta_s`;
- `t = k, 2k, …, ⌊n/k⌋·k`, with n = round(L^z);
- `lambda` is the mean of ln(\|d\|/ε) over the k steps ending at t, so any average over a window of whole blocks equals the `record_every = 1` result;
- `delta_s` is S_diff at t.

With `record_every = 1` the rows are exactly as before (t = 1…n). File names: `data/spin_dists_per_time_v2/N4/a<a>/IC1/L<L>/N4_a<a>_IC1_L<L>_z<z>_timestep<k>_sample<i>.csv`.

**Notebooks.** All changed by `review_2026_09/apply_review_spin_notebooks.py`, which checks every replacement.

In **`analysis_lyapunov_fixed.ipynb`, `s_diff_analysis_python.ipynb`, `s_diff_analysis_logsdiff_plot.ipynb` and `capture_time_distrabution.ipynb`**:
- **New parameters:** `DATA_DIR = "spin_dists_per_time_v2"` and `RECORD_EVERY = 1`. Set `RECORD_EVERY` to the generator's `record_every`.
- **Helpers:** `t_index(L, t)` gives the row nearest to time t; `t_window(L, t_min, t_max)` gives a row mask.
- **Loaders** read the `t` column and fill `collected_times[L]`.
- **Downstream indexing:** the old index arithmetic is replaced wherever it assumed row i = time i+1. That covers `[t_idx]` with t_idx = L^z − 1, `[t_min-1:t_max]`, `np.arange(n)` as a time axis, `i*time_steps`, and `values[num_skip:n]`.
- **Caches:** the `.npz` cache names get `_timestep<k>`, and a `_times.npz` is saved next to them. `analysis_lyapunov_fixed` writes its summary CSVs to `../data/spin_chain_lambdas_v2/`.
- **Two latent bugs fixed on the way:** the zoomed log(S_diff) figures used the fit of the last `L` of an earlier loop instead of `L_plot`; the fit-check plots drew the data against row index but the fit against 1-based time.

In **`analyze_sdiff_per_time3.ipynb`**:
- data folders → `s_diff_per_time_v2`;
- the loader checks the new `t` column and keeps the first T_f/time_step rows, so row i ↔ t = i·time_step and every later cell sees the same array length as before;
- a BVH cell is appended (4.3.4, below).

**4.3.4 implemented.** The BVH fit cell ($1/A=a+B\ln t$ next to a power law over the same window, plus the local slope $d(1/A)/d\ln t$) is appended, with a markdown header, to `analyze_random_upper_lower_binary_rho_per_time.ipynb`, `analyze_random_slidding_p_rho_per_time2.ipynb` (A = 1−ρ) and `analyze_sdiff_per_time3.ipynb` (A = S_diff).

**Verification.**
- **Every notebook was executed here**, original vs edited, with overrides for data location, number of ICs and a reduced L set. The inputs were synthetic CSVs, identical in content between the old format and the new `_v2`/`_timestep1` format; a `_timestep5` set was built exactly the way `benettin_lambda_sdiff` does it. Results:
  - `s_diff_analysis_python`: with `RECORD_EVERY=1`, the `collected_S_diffs`, all fitted slopes, offsets and slope errors, and ξ_t are **identical** to the original. With `RECORD_EVERY=5`, ξ_t agrees within 3.5% (fewer points), and S(t = L^z) is read at the nearest recorded time.
  - `s_diff_analysis_logsdiff_plot`: slopes and offsets identical.
  - `analysis_lyapunov_fixed`: λ window averages identical at k = 1. At k = 5 they differ by ≤ 1.3×10⁻³, because the window edges fall inside a block; choose windows that are multiples of k to remove this.
  - `capture_time_distrabution`: averaged arrays identical. **t\* from both methods is now exactly 1 larger than before.** The old code returned the row index (t − 1) as the time. Your capture-time histograms and Poisson fits shift by one step.
  - `analyze_sdiff_per_time3`: `collected_sdiffs` identical after the trim.
  - The BVH cells recover exact inputs: B = 0.300 and 0.250 with χ²ν = 0 for exact BVH data, and δ = 0.1000 for an exact power law. On your local spin pickles they reproduce the §7.5 table.
- **Cells that failed here** failed identically in the original and edited notebooks:
  - `s_diff_analysis_python` cell 14 uses Python ≥ 3.12 f-string syntax; it's fine on your Mac.
  - `s_diff_analysis_python` cell 29 plots a = 0.69, which isn't in `a_vals` (pre-existing KeyError).
  - A few cells hard-code an L or sample index outside my reduced test set.
- **Julia.** Not run here. The edits parse cleanly. I also checked, with the Python test data, that `check_spin_local_output.py` accepts correct output and that the eight untested generator scripts differ from the two tested ones only in settings lines. **Please run:**
  ```bash
  bash heisen_spin_chain/tests/run_spin_tests.sh      # writes heisen_spin_chain/tests/last_test_run.log
  ```
  It runs, in order:
  1. `test_spin_refactor.jl`, which compares against frozen copies of the old functions in `tests/reference_pre_2026_09.jl`. It checks: the evolve trajectory is unchanged and the caller's J is untouched; `random_global_control_sdiff` equals S_diff of the stored trajectory at true times; `benettin_lambda_sdiff(k=1)` is bit-identical to the old inline loop; k > 1 gives block means and subsampled S_diff; \|J\| is preserved.
  2. Local runs of `get_good_data_severalL.jl` and `get_sdiff_data_severalL0.jl`, launched from the repo root like your submit scripts.
  3. `check_spin_local_output.py`, which checks the files against the notebook path templates and t columns, and checks that the other eight scripts differ only in settings.

  Tell me when it has run and I'll read the log. → **You ran it 2026-09-25: OVERALL PASS (§7.11).**

**Found, not changed.** Flagged for you:
- `capture_time_distrabution.ipynb`, `find_t_star_dist`: `t_stars.append(round(pre_b - post_b / (post_m - pre_m)))`. Operator precedence makes this $b_{pre} - b_{post}/(m_{post}-m_{pre})$. The crossing of the two fitted lines is $(b_{pre}-b_{post})/(m_{post}-m_{pre})$, so the parentheses are probably missing. It changes the t\* values. I left it for you to confirm. → **Your answer (2026-09-25): leave it. The line-matching method did not give what you are after and is kept only for bookkeeping; the Lyapunov-exponent fitting is the method that produced the correct results (§7.11).**
- The log(S_diff) fits use `xs` = 1, 2, … starting at `t_min`, so the stored intercept is relative to `t_min`. The zoomed figures then plot `offset + slope * t` against absolute t, which is why a hand-tuned `+0.2` shift is there. It is unchanged, since I kept the intercept convention; say if you want absolute intercepts instead.

### 7.10 Your question on 4.2.3: is `time_prefact < 1` wise? (2026-09-25)

No, not as a blanket rule; I overstated it. What the run length should be depends on what the run is for.

**Decay runs at criticality** (δ, the running exponents, the BVH tests):
- **The initial condition doesn't have to be lost.** The quantity being measured is the relaxation *from* it. A finite system at criticality has no stationary state to forget it into; it eventually falls into the absorbing state. What you need is to be past the short, non-universal transient of the particular random initial state (tens to hundreds of steps), and still before finite-size effects set in.
- **Finite size sets the limit.** Finite-size effects begin when the correlation length $\xi_\perp(t)\sim t^{1/z}$ reaches $L$, i.e. around $t\sim L^z$. Within $t<L$ nothing can be affected at all (the light-cone argument).
- **Why 100L works for clean DP.** With $z=1.58$ and $L=20000$, $L^z\approx6.4\times10^6$, so $t_{\max}=100L=2\times10^6$ is about $0.3\,L^z$, which is safe enough. I can't check what [40] used; if it was 100L, it is consistent with this.
- **Why it's riskier with temporal disorder.** The effective $z$ is smaller. Your own 1.45 gives $L^z\approx1.8\times10^6$, so 100L ≈ L^z. At BVH's $z\to1$ (with log corrections) the safe window shrinks toward $t\sim L$. My small-L test in §7.3 saw deviations at 2–4L.

**Stationary quantities** (β from the steady-state activity in the active phase) do need to lose the initial condition: $t\gg\xi_t$ with $L\gg\xi_\perp$. That is a different run from the decay runs.

**Recommendation:**
- Keep `time_prefact = 100`; the defaults are unchanged.
- Let the data decide which late times to trust: for the critical estimate, run the log-time generator at $L$ and $L/2$ (or $L/4$), and use only the times where the two agree within errors. `analyze_time_log_fss.ipynb` has this check built in.
- With log-spaced output the extra decades cost almost nothing in stored rows; compute cost still grows linearly in $t_{\max}$.

### 7.11 Ratio-mask fix, spin test results, and the t* note (2026-09-25)

**Spin-chain tests passed.** You ran `bash heisen_spin_chain/tests/run_spin_tests.sh` (Julia 1.12.6, 2026-09-25 15:19 EDT). `heisen_spin_chain/tests/last_test_run.log`:
- `test_spin_refactor.jl`: **187/187 passed**. The new functions reproduce the frozen pre-2026-09 code exactly at `record_every = 1`, give block means / subsampled S_diff for k > 1, and leave the caller's J untouched.
- Local runs of `get_good_data_severalL.jl` and `get_sdiff_data_severalL0.jl`: PASS.
- `check_spin_local_output.py`: 16/16 expected files found, none unexpected; the other eight generator scripts differ only in settings. **OVERALL: PASS.**
- The only other output was juliaup's "1.13 available" notice.

**α-ratio / δ-ratio divide-by-zero fix (the §7.8 question; you said yes).** Applied by `review_2026_09/apply_review_ratio_mask.py`, which asserts every replacement:

| notebook | cell | quantity | change |
|---|---|---|---|
| `analyze_random_upper_lower_binary_rho_per_time.ipynb` | 9 | α-ratio $\ln[Q(t/b)/Q(t)]/\ln[\ln t/\ln(t/b)]$ | keep only times with $Q(t/b)>0$ and $Q(t)>0$ |
| `analyze_random_slidding_p_rho_per_time2.ipynb` | 14 | α-ratio | same |
| `analyze_random_slidding_p_rho_per_time2.ipynb` | 9 | δ-ratio $\log_{10}[Q(t/10)/Q(t)]$ | same mask |

The inserted lines (α-ratio version):
```python
ok = (data_to_plot_numerator1 > 0) & (data_to_plot_numerator2 > 0)   # 2026-09: only times with Q(t/b), Q(t) > 0 (no log of 0)
l_time_vals = np.array(l_time_vals)[ok]
data_to_plot = np.log(data_to_plot_numerator1[ok]/data_to_plot_numerator2[ok]) / np.log(data_to_plot_denomenator1[ok]/data_to_plot_denomenator2[ok])
```
Tested on synthetic data with one fully absorbed control value: the divide-by-zero warning is gone, the absorbed curve now ends at its last surviving time (350 → 160 points), and curves that never hit Q = 0 are unchanged. Only cell sources changed; re-run the notebooks to refresh outputs.

**`find_t_star_dist` (capture-time line matching): kept as is, for bookkeeping only.** You said this method of matching two fitted lines did not produce what you are after; the fitting with Lyapunov exponents is what produced the correct results. The suspected missing parentheses in `pre_b - post_b / (post_m - pre_m)` (§7.9) are therefore left alone and don't matter for the results.

**Documentation restructured.** This file (`PROJECT_HISTORY.md`) now holds the full record. `REFEREE_CONFLICT_REVIEW.md` was rewritten as a short active-task list with plans for the future.

### 7.12 Full Stavskaya test runner passed (2026-09-25)

You ran `bash stavskya_mc/block_disorder/tests/run_all_tests.sh` (Julia 1.12.6, Python 3.13.7, 2026-09-25 17:08 EDT). `stavskya_mc/block_disorder/tests/last_test_run.log`:
- `test_dynamics.jl`: **678/678 passed**. Speed at L = 20000, T = 2000 was 0.116 s → 0.099 s (×1.18 this run, ×1.07 last time), with allocations down from 313 MiB to 0.5 MiB.
- All six generators passed their local smoke runs (`STAV_LOCAL_TEST=1`).
- `check_local_test_output.py`: **96/96 expected files found, 0 unexpected**. The Julia file names and the Python loaders agree.
- `test_time_log_tools.py` was skipped because pytest isn't installed for that Python (`pip install pytest` to include it). It passed 9/9 in my sandbox earlier.
- **OVERALL: PASS.** The only other output was juliaup's "1.13 available" notice.

This closes the last test task; the new Stavskaya pipeline is cleared for production runs.

