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


### 7.13 Cluster launch timeout (2026-09-25)

Your first cluster submission of `get_sdiff_data_severalL1.jl` (Julia 1.11.6) failed inside `addprocs(SlurmManager())` with `launch_timeout exceeded`.
- **What failed:** SlurmClusterManager waits a default 60 s for all srun'd workers to report back, and 250 workers did not report in time. This happened before any project code ran on the workers.
- **What didn't cause it:** the call is unchanged from the old scripts. The cluster tests (`test_spin_refactor.jl` and the local run on 1.11.6) passed.
- **Change:** the ten spin generators and `stavskya_mc/block_disorder/generators/setup_workers.jl` now use `SlurmManager(launch_timeout=600.0)`. `check_spin_local_output.py` still passes, since the scripts still differ only in settings. The older Stavskaya scripts were not changed.

### 7.14 BVH reproduction with the Stavskaya model (2026-09-29)

**Parameters and where they come from.**
- **BVH 1D strong disorder.** The active rate is used with probability 0.8, the inactive rate is the active rate/20, and disorder is held for Δt = 6 (and 3). Decay runs used L = 200,000 with 50,000 realizations; spreading runs used 5×10⁴–10⁶ runs up to t ≈ 10⁵.
- **Mendonça (arXiv:1011.1489), clean Stavskaya.** ε* = 0.29450(5), δ = 0.155(5), z = 1.6.
- **Pilot scan** (`review_2026_09/pilot_bvh_scan.py`, NumPy port, L = 2000, t ≤ 2000, 300 samples). With p = 0.2 and ε_l = ε_u/20, ε_u,c ≈ 0.60 at block_len 1 and ≈ 0.46 at block_len 6. With ε_l = ε_u/10 the critical point barely moves. In this model the disorder strength is set by p and by how far ε_u lies above ε*, so your existing p = 0.8 (upper/lower) and sliding-p models are the weaker rungs.

**First version, then stripped back.** The first version used a presets file (`bvh_presets.toml`), preset-driven generators and an overview notebook. You asked for plain knob copies in your usual style, so all of that was deleted. `analyze_time_log.ipynb`, `analyze_time_log_fss.ipynb`, `time_log_tools.py` and `.gitignore` were restored to their committed versions.

**What remains.**

Decay runs are copies of your generators with only the knobs and header changed. Each has its own submit script. Data go to the usual `data/time_log/time_rand_window_binary/...`, told apart by `pval`, `L100000` and `timepref1p0`.

| copy of `get_upper_lower_binary_time_log*.jl` | p_val | average_epsilon_c / rate | ε_u values | block_len | L, time_prefact | samples |
|---|---|---|---|---|---|---|
| `_bvh_b1` | 0.2 | 0.144 / 0.0012 | 0.585–0.615 (7) | 1 | 10⁵, 1.0 | 20,000 |
| `_bvh_b6` | 0.2 | 0.1104 / 0.0012 | 0.445–0.475 (7) | 6 | 10⁵, 1.0 | 20,000 |
| `_clean` | 1.0 | 0.2945 / 0.0001 | 0.2942–0.2948 (7) | 1 | 10⁵, 1.0 | 5,000 |
| `_fss_bvh_b1` | 0.2 | 0.144 (0:0) | 0.6 (set after bvh_b1) | 1 | 1000–16000, 100 | 5,000 |

ε_l = ε_u/20 throughout, and ε̄ = 0.24 ε_u at p = 0.2. With `time_prefact = 1` every output time is t ≤ L, where the periodic chain is statistically identical to the infinite one, so there are no finite-size effects.

The spreading runs are the only genuinely new code. The existing generators always start from a random half-filled chain and record only ρ, so they cannot give single-seed survival P_s(t), number of active sites N(t) or radius R(t).
- `utils/dynamics.jl` gained two functions:
  - `time_random_spreading` runs one seed on the infinite chain, updating only the active window. Active sites stay within x0 ≤ i ≤ x0 + t, centred on x0 + t/2.
  - `spreading_chunk` sums over many runs.
- `generators/get_upper_lower_binary_spreading_{bvh_b1,bvh_b6,clean}.jl` use knob style like the time-log generators. Each has 5 ε values, t_max = 10⁵, and 200 chunks × 500 runs per value; each chunk writes one CSV of sums, and all chunks share one `pmap` queue. `naming.jl` gained `spreading_chunk_path`. Submit scripts: `submit_upper_lower_spreading_*.sh`.
- `analysis/spreading_tools.py` loads the chunks and computes P_s, N and R² with bootstrap errors, local slopes, and `best_log_exponent`.
- `analysis/analyze_spreading.ipynb` reproduces BVH Figs. 4–7: 1/P_s vs ln t, 1/δ_eff, (N/t)^(−1/y_N) and (R/t)^(−1/y_R), and crossing times. It overlays the clean run, and its parameters use the generator's knob names. It was rewritten on 2026-09-29 as an explanatory notebook: the model and light cone, the stored sums and the observables built from them, DP scaling, the Harris/Kinzel criterion and crossover time, the BVH predictions with their finite-time forms (1/δ_eff = ln t + a/B has slope 1; θ_eff ≈ 1 − y_N/ln t; 1/z_eff ≈ 1 − y_R/ln t), how to read each figure, sanity checks (P_s(0) = N(0) = 1, R ≤ t/2), and a closing verdict table. On synthetic clean data it gave θ_eff = 0.322 and 1/z_eff = 0.640 (DP: 0.314, 0.633).

Tests: `tests/test_spreading.jl` and `tests/test_spreading_tools.py`. `check_local_test_output.py` now covers the 7 new generators (200 files), and `run_all_tests.sh` runs everything.

**Verified here** (no Julia in the sandbox):
- every `.jl` file parses;
- a Python port of the spreading kernel matches a full-lattice simulation (|z| < 1) and the exact cases;
- the checker passes on emulated local output (200/200) and catches a missing or misnamed file;
- Python tests: 12/12;
- `analyze_spreading.ipynb`, and your unmodified `analyze_time_log.ipynb` pointed at the `_bvh_b1` knobs, both ran end to end on small synthetic data.

No production-size simulation was run.

**Cost estimate:**
- each p = 0.2 decay copy takes a few hours on 500 tasks (the clean copy about a quarter of that) and writes ~140k small CSVs (~0.4 GB);
- each spreading copy takes ~1–2 h and writes 1000 chunk files.

### 7.15 Snapshot of the task list as it stood before 2026-10-05

*Moved here verbatim from `REFEREE_CONFLICT_REVIEW.md` on 2026-10-05, when that file was rewritten around the first BVH-reproduction results. The spin-chain v2 regeneration is still in progress (the cluster jobs had failed because the account ran out of memory; fixed by you).*

#### Where things stand

- **The question.** Are the measured exponents ($\delta\approx0.11$, $z\approx1.35$–$1.45$, $\nu_t\approx2.25$) a new fixed point? Or are they effective exponents crossing over toward the infinite-noise point of Barghathi–Vojta–Hoyos (BVH)?
  - Working hypothesis: crossover. Every exponent sits between clean DP and the infinite-noise limit.
  - A pure $1/\ln t$ decay gives the running estimate $\delta_{\rm eff}=0.114\to0.075$ over the paper's window, so the Stavskaya δ alone cannot decide (H §0, §3).
- **Stavskaya code.** Done and tested: 678/678 checks, all six generators run locally, 96/96 files found by the Python loaders (H §7.7, §7.12):
  - kernel refactor, bit-identical to the old one;
  - `block_len` disorder blocks;
  - log-time generators, submit scripts and analysis notebooks in `stavskya_mc/block_disorder/`.
- **Spin-chain code.** Done and tested (187/187 checks, local runs and file checks all pass, H §7.9, §7.11):
  - initial-condition bug fixed;
  - new `random_global_control_sdiff` and `benettin_lambda_sdiff`;
  - `record_every` knob;
  - output to `_v2` folders, with the analysis notebooks updated.

  **All old spin data are superseded.**
- **Notebook fixes.** All annotated fixes are done, including the α/δ-ratio divide-by-zero mask. Only cell sources changed, so the saved outputs are stale until you re-run them.
- **Git.** Nothing has been committed. About 33 changed or new paths are sitting uncommitted in the working tree.

---

#### Active tasks (rough order)

1. **Regenerate the spin-chain data (v2).**
   - **Before submitting:**
     - The new code is not committed yet. Commit and push it, then `git pull` in the cluster copy (`~/senior_thesis2/senior_thesis`).
     - The tests ran on Julia 1.12.6, but the cluster uses 1.11.6. Do a one-minute check there first:
       ```bash
       ~/julia-1.11.6/bin/julia heisen_spin_chain/tests/test_spin_refactor.jl
       SPIN_LOCAL_TEST=1 ~/julia-1.11.6/bin/julia heisen_spin_chain/lyapunov_exponents/get_good_data_severalL.jl
       ```
     - The notebooks' parameter cells (L, a values, number of ICs, and for `analyze_sdiff_per_time3` also `time_prefact` and `time_step`) still hold the old runs' values. Match them to what you submit.
     - The transfer commands in `ssh_transfer.txt` point at the old folders. Use `data/spin_dists_per_time_v2` and `data/s_diff_per_time_v2`.
   - Lyapunov/S_diff runs:
     - set `@everywhere record_every = …` in `heisen_spin_chain/lyapunov_exponents/get_good_data_severalL{,2..6}.jl`; it appears in file names as `_timestep<k>`;
     - submit with `sbatch submit_severalL_data_job{,2..6}.sh`;
     - output goes to `data/spin_dists_per_time_v2/`.
   - S_diff-per-time runs:
     - submit with `sbatch submit_sdiff_time_data_number.sh <0–3>`;
     - output goes to `data/s_diff_per_time_v2/`.
     - Note: the older `submit_sdiff_time_data.sh` calls `get_sdiff_data_severalL.jl`, which no longer exists. Use the `_number` version.
   - **Then re-run the analysis notebooks:**
     - `analysis_lyapunov_fixed`, `s_diff_analysis_python`, `s_diff_analysis_logsdiff_plot`, `capture_time_distrabution`: set `RECORD_EVERY` to match the generator;
     - `analyze_sdiff_per_time3`.
   - Tips:
     - choose λ averaging windows that are multiples of `record_every`;
     - capture times t\* will come out exactly one step later than in the old results (the old code returned the row index).

2. **BVH reproduction with the Stavskaya model: ready to run** (details H §7.14). These are copies of your upper/lower generators with only the knobs changed, plus new spreading generators.
   - **Local test on your Mac first:** `bash stavskya_mc/block_disorder/tests/run_all_tests.sh`. It now also runs `test_spreading.jl` and every new copy at tiny sizes.
   - **Cluster**, after `git pull` in `~/senior_thesis3/senior_thesis` (the Stavskaya clone). Run these from `stavskya_mc/`, where both `block_disorder/submit/` and `data/` are visible. The Slurm logs land in `stavskya_mc/`:
     ```bash
     cd ~/senior_thesis3/senior_thesis/stavskya_mc
     sbatch block_disorder/submit/submit_upper_lower_time_log_clean.sh
     sbatch block_disorder/submit/submit_upper_lower_time_log_bvh_b1.sh
     sbatch block_disorder/submit/submit_upper_lower_time_log_bvh_b6.sh
     sbatch block_disorder/submit/submit_upper_lower_spreading_clean.sh
     sbatch block_disorder/submit/submit_upper_lower_spreading_bvh_b1.sh
     sbatch block_disorder/submit/submit_upper_lower_spreading_bvh_b6.sh
     ```
     - clean: DP control, ε = 0.2942–0.2948
     - bvh_b1: p = 0.2, ε_u = 0.585–0.615, block_len 1
     - bvh_b6: p = 0.2, ε_u = 0.445–0.475, block_len 6
     - The three spreading scripts use the same models, starting from one active site.
     - Later, once `bvh_b1` has pinned ε_u,c, set `average_epsilon_c` in `get_upper_lower_binary_time_log_fss_bvh_b1.jl` to 0.24 × ε_u,c and run `sbatch block_disorder/submit/submit_upper_lower_time_log_fss_bvh_b1.sh`.
   - **Data** (from `stavskya_mc/`):
     - decay: `data/time_log/time_rand_window_binary/rho_per_time/IC1/L100000/epsilonu<u>/epsilonl<l>/pval0p2/blocklen<b>/`; the clean run uses `pval1p0`.
     - spreading: `data/spreading/time_rand_window_binary/spreading/tmax100000/epsilonu<u>/epsilonl<l>/pval<p>/blocklen<b>/`
   - **Getting the data home:** `stavskya_mc/block_disorder/ssh_transfer.txt`. `time_log/` (GB-sized) goes to `/Volumes/ExternalData/stavskya_mc/data/time_log/`; `spreading/` and `block_rho_per_ep/` (small) go to the repo's git-ignored `stavskya_mc/data/`. The `analyze_time_log*.ipynb` notebooks read the external drive unless `TESTING = True`.
   - **Analysis:**
     - `analyze_time_log.ipynb` with `MODEL = "window_binary"`, `L = 100000`, `TIME_PREFACT = 1.0`, `AVG_EPS_C` / `AVG_EPS_RATE` / `P_VAL` / `BLOCK_LEN` / `N_SAMPLES` set to the generator's knobs;
     - `analyze_spreading.ipynb` (same knob names). It is written as a guide: the model, the observables, the DP and BVH predictions with formulas, what to look for in each figure, and a closing verdict table.
     - `analyze_time_log_fss.ipynb` for the FSS copy.

3. **Stavskaya log-time production runs** (`stavskya_mc/block_disorder/submit/*.sh`).
   - Start with `block_len = 1`.
   - The critical control values in the generators were found for `block_len = 1` only. Rescan them with the `*_rho_per_ep_block` generators before using `block_len > 1`.

4. **Decide which late times to trust.**
   - Keep `time_prefact = 100` (H §7.10).
   - Run the `*_time_log_fss` generators at $L$ and $L/2$ (or $L/4$).
   - Use only the times where the sizes agree within errors. `analyze_time_log_fss.ipynb` has the light-cone/agreement check built in.

5. **Run the BVH tests on the new data.** The notebooks already contain all of these: `analyze_time_log.ipynb`, plus the BVH cells appended to the two `rho_per_time` notebooks and `analyze_sdiff_per_time3`.
   - $1/A$ against $\ln t$ is a straight line at infinite-noise criticality; the local slope $d(1/A)/d\ln t$ is flat.
   - Running $\delta_{\rm eff}$ drifts toward 0 (it is constant at a power-law point); $1/\delta_{\rm eff}$ against $\ln t$ is linear.
   - Crossing times off criticality: $\ln t_x\propto r^{-1/2}$ (BVH) vs $\nu_t\ln(1/r)$ (power law).
   - Width of $P(\ln A)$ over disorder realizations: it grows linearly in $\ln t$ under BVH and saturates at a finite-disorder fixed point. This can already be tried on the existing per-sample Stavskaya CSVs.

6. **Re-run the edited old Stavskaya notebooks** to refresh their saved outputs. These are the upper/lower and sliding-p `rho_per_time` notebooks, the `_z` notebooks and `rho_per_ep` (list in H §7.5, §7.11).

---

#### Plans for later

- **Disorder-strength scans.**
  - Stavskaya: vary `block_len` and the contrast $\varepsilon_u/\varepsilon_l$.
  - If the "exponents" move with disorder strength, they are crossover values. If strong disorder gives clean log scaling sooner, that is BVH directly.
  - Spin chain: this needs a new knob (not implemented yet). Options are J-signs held fixed over blocks $\Delta t>1$, or a binary-random $a(t)$.
- **Spreading runs from a single seed** ($P_s(t)$, $N_s(t)$, $R(t)$): the cleanest test of $z=1$ with log corrections, free of finite-size effects. Not implemented yet.
- **Temporal Griffiths:** lifetime $\tau(L)$ on the active/chaotic side. BVH predict $\tau\sim L^{1/\kappa}$ with $\kappa$ varying continuously.
- **Rewrite the paper's claim** (H §5 item 7). Keep the Hamiltonian control transition and the discontinuous Lyapunov exponent, drop "new universality class", cite BVH and show the crossover analysis.
- **Optional, "Option A":** allowlist `julialang-s3.julialang.org`, `pkg.julialang.org` and `*.pkg.julialang.org` so I can run the Julia tests myself before handing code to you (H §7.7).

---

#### Known issues deliberately left alone

- **`find_t_star_dist` (capture-time line matching).** It is kept for bookkeeping only. The Lyapunov-exponent fitting is the method of record, so the suspected missing parentheses there don't matter.
- **Pre-existing notebook failures.**
  - `s_diff_analysis_python` cell 29: KeyError, because a = 0.69 is not in `a_vals`.
  - `s_diff_analysis_python` cell 14: needs Python ≥ 3.12, which your Mac has.
- **log(S_diff) fit intercepts.** They are relative to `t_min`, hence the hand-tuned `+0.2` shift in the zoomed figures. Say if you want absolute intercepts.
- **DataCollapse and the other items marked DC** in H §4.

---

#### How to run the tests

```bash
cd ~/research/senior_thesis
bash stavskya_mc/block_disorder/tests/run_all_tests.sh   # Stavskaya: kernel tests, 6 local generator runs, file-name check, pytest
bash heisen_spin_chain/tests/run_spin_tests.sh           # spin chain: refactor tests, 2 local runs, file/script check
```
Each writes `last_test_run.log` next to itself, ending in `OVERALL: PASS/FAIL`. Tell me you ran one and I'll read the log from your folder. Local test output goes to git-ignored folders (`_local_test_output/`).

### 7.16 First BVH-reproduction results: decay (time_log) and spreading (2026-10-04/05)

Data: L = 10⁵, t ≤ L decay runs (`analyze_time_log.ipynb` clean, `analyze_time_log3.ipynb` block_len 6) and t ≤ 10⁵ spreading runs (`analyze_spreading.ipynb` clean, `analyze_spreading2.ipynb` block_len 1, `analyze_spreading3.ipynb` block_len 6). The block_len 1 decay data were still being finished (493 empty files at ε_u = 0.615). Claude also read the raw files directly (decay aggregated in the session VM; summary in `stavskya_mc/block_disorder/analysis/_claude_scratch/summary.json`).

**Clean control: textbook DP, which validates the pipeline.**
- Decay: ε* = 0.2945 (Mendonça 0.29450(5)); fitted δ = 0.156 (DP 0.1595); crossing-time ν_t ≈ 1.79 (DP 1.73); over t = 10³–10⁵ the power law gives χ²ᵣ = 1.1 and BVH 1591.
- Spreading at 0.2945: δ_eff 0.15–0.16, θ_eff 0.31–0.32, 1/z_eff 0.63–0.64 at every time; P_s fit χ²ᵣ 0.09 (power law) vs 84 (BVH).
- Spread of ln A over samples grows like t^0.315: intrinsic finite-chain noise √(ξ⊥/L) ∝ t^(1/(2z)) = t^0.316, not disorder. (Claude had wrongly said it "saturates" for DP; that only applies to the disorder part at a conventional fixed point.)

**Block_len 6 (p = 0.2, ε_u/ε_l = 20).**
- *Block-phase artifact.* Output times fall at arbitrary steps inside the 6-step disorder blocks; the disorder-averaged activity has a sawtooth of about −2.2% (first step of a block) to +2.8% (last step). Residuals were ~4× the statistical error; one global phase correction brings them to ~1×. It drives the choppiness of δ_eff, 1/δ_eff and α_eff. Spreading output is affected the same way, less visibly.
- *Decay at ε̄ = 0.1104:* 1/A = 0.999 + 0.121 ln t, i.e. B(ln t + c) with offset c ≈ 8.3. BVH with this offset reproduces the measured δ_eff (0.068/0.062/0.059 at t = 10³/10⁴/10⁵ vs predicted 0.072/0.061/0.054), α_eff (0.39/0.50/0.60 vs 0.41/0.49/0.56) and 1/δ_eff. The grey "1/ln t" curve and "α_eff → 1" in the notebooks assume c = 0, so they are the wrong references while ln t ≈ c. But a power law with δ ≈ 0.06 fits nearly as well: over the window, ln t + c only changes by a factor 1.5. Box–Cox straightness test, (A^(−κ) − 1)/κ vs ln t (κ = 0 power law, κ = 1 BVH): best κ ≈ 0.65–0.85; χ²ᵣ(κ = 1) = 1.1 vs χ²ᵣ(κ = 0) = 1.3–3.2: leans BVH, weakly.
- *Spreading at ε̄ = 0.1104 (much more decisive):* d(1/P_s)/d ln t = 0.108, 0.111, 0.113, 0.125 at ln t = 6, 8, 10, 11.4 (flat: BVH Fig. 4); best κ = 0.80 with χ²ᵣ 0.15; κ = 1 gives 0.42 and κ = 0 gives 5.6. Straightest log exponents y_N = 4.3, y_R = 2.5 (BVH contact process: 3.6, 1.7; the clean run prefers a pure power law).
- *Not DP at any time:* even at t = 50–400, before off-criticality matters and for every ε, the local exponents are δ ≈ 0.07, θ ≈ 0.67, 1/z ≈ 0.84 (DP: 0.16, 0.31, 0.63). Late values at 0.1104: δ 0.06, θ 0.72–0.77, 1/z 0.86–0.87.
- *Critical point:* ε̄_c ≈ 0.1100–0.1104 (0.1092 active, 0.1116 inactive; at 0.1104 the spreading 1/δ_eff peels down and B_eff rises slightly at the end, the inactive-side signature).
- *Crossing times (decay, from 0.1104):* local slope of ln t_x vs ln(1/r) rises 1.7 → 2.7 as r shrinks (the activated-scaling signature), but only three points and ε̄_c uncertain.
- *Width of ln A (decay):* at 0.1104 grows ~0.1 per decade (0.85, 0.96, 1.05 at 10³, 10⁴, 10⁵); the active side saturates (0.67 at 0.1068).

**Block_len 1 (p = 0.2).**
- The critical point falls between grid values: ε̄ = 0.1428 is active and 0.1440 inactive (ε_u 0.595–0.600); estimated ε̄_c ≈ 0.1432–0.1436. No grid value is close enough for critical tests.
- Local exponents sit between clean and block_len 6 at all times: early δ ≈ 0.10, θ ≈ 0.55, 1/z ≈ 0.75. These are close to the paper's Stavskaya values (δ ≈ 0.105, z ≈ 1.45, i.e. 1/z ≈ 0.69).

**Interpretation so far.** The effective exponents move monotonically with disorder strength, from clean through block_len 1 to block_len 6, toward the infinite-noise limit (δ → 0, θ and 1/z → 1). A single new fixed point would give the same critical exponents for block_len 1 and 6. At block_len 6 the spreading survival probability prefers the BVH logarithmic form by the same test that cleanly identifies DP in the clean model. This supports the crossover/BVH reading; the paper's exponents look like an intermediate-disorder crossover value.

### 7.17 Block_len 1 decay data complete (2026-10-05)

All 7 × 20,000 files present (the 493 empty files at ε_u = 0.615 were refilled). `analyze_time_log2.ipynb`:
- surviving fraction at t = 10⁵ is 1.0000 / 0.9996 / 0.977 / 0.398 / 0.003 at ε̄ = 0.1416 / 0.1428 / 0.1440 / 0.1452 / 0.1464, so ε̄_c lies between 0.1428 and 0.1440.
- Interpolating ln A linearly in ε̄ between those two curves:
  - near 0.1430 the BVH form is preferred: best κ ≈ 1.2; χ²ᵣ 1.4 for BVH vs 10.8 for a power law; per-decade δ 0.092 / 0.070 / 0.065;
  - at 0.1432 a power law is preferred: κ ≈ 0.1; δ 0.097 / 0.082 / 0.100.

  The verdict flips within Δε̄ = 0.0002, so the grid is too coarse to decide.
- The crossing-time plot used `CONTROL_C` = 0.1440, an inactive curve, so its ν_eff = 1.68 is not meaningful.
- The spread of ln A at 0.1428 saturates near 0.79 (active side).
- Decade-by-decade power-law δ (clean constant at 0.155–0.158; disordered runs drifting), and the plan built on it, are in §7.18 (snapshot); the assessment of both is §7.19.

### 7.18 Snapshot of the Stavskaya status and next-steps proposal as they stood on 2026-10-05

*Moved verbatim from `REFEREE_CONFLICT_REVIEW.md` ("Where things stand", "Stavskaya next steps") on 2026-10-05, when they were replaced by the plan revised after the assessment in §7.19.*

#### Where things stand

- **The question.** Are the measured exponents (δ ≈ 0.11, z ≈ 1.35–1.45, ν_t ≈ 2.25) a new fixed point? Or are they effective exponents crossing over toward the infinite-noise point of Barghathi–Vojta–Hoyos (BVH)?
- **Stavskaya, first results (H §7.16):**
  - **Clean control:** textbook DP. This validates the pipeline.
  - **Effective exponents move with disorder strength:** clean → block_len 1 → block_len 6, toward the infinite-noise limit:

    | | δ_eff | θ_eff | 1/z_eff |
    |---|---|---|---|
    | clean | 0.16 | 0.31 | 0.63 |
    | block_len 1 | ≈ 0.10 | ≈ 0.55 | ≈ 0.75 |
    | block_len 6 | ≈ 0.06–0.07 | ≈ 0.7 | ≈ 0.85 |
  - **Block_len 6 spreading** prefers the BVH log form for P_s by the same test that identifies DP in the clean run.
  - **Block_len 6 decay** is consistent with BVH plus a non-universal offset, but cannot yet exclude a power law.
  - **Block_len 1 decay (complete, 2026-10-05):** ε̄_c lies between the grid values 0.1428 (active) and 0.1440 (inactive). Interpolating ln A between them puts it near 0.1430–0.1432, and the verdict there is ambiguous:
    - at 0.1430 the BVH form is preferred (best κ ≈ 1.2; χ²ᵣ 1.4 for BVH vs 10.8 for a power law);
    - at 0.1432 a power law is (κ ≈ 0.1);
    - the grid (Δε̄ = 0.0012) is about 6× too coarse.

    Near ε̄_c the power-law δ fitted on 10²–10³ is ≈ 0.09–0.10, matching the paper's δ ≈ 0.105.
  - **Crossover signature:** the decade-by-decade power-law fit of the decay δ is constant only for the clean run.

    | power-law δ fitted per decade | 10²–10³ | 10³–10⁴ | 10⁴–10⁵ |
    |---|---|---|---|
    | clean, ε = 0.2945 | 0.155 | 0.158 | 0.154 |
    | block_len 6, ε̄ = 0.1104 | 0.070 | 0.061 | 0.058 (BVH with offset predicts 0.072 → 0.054) |
    | block_len 1, interpolated 0.1430 | 0.092 | 0.070 | 0.065 |
    | block_len 1, interpolated 0.1432 | 0.097 | 0.082 | 0.100 |

    No disordered ε̄ gives a constant DP-like δ.
  - **Working reading:** crossover toward BVH. The paper's exponents look like an intermediate-disorder value, very close to block_len 1's.
- **Spin chain:** regenerating v2 data. The earlier cluster failures were the account running out of memory, now fixed. Run instructions are in H §7.15, item 1.


#### Stavskaya next steps (proposed 2026-10-05, waiting for your go-ahead)

**What the data can and cannot say yet.**
- *Clear:*
  - the clean model is DP;
  - the effective exponents move monotonically with disorder strength (clean → block_len 1 → block_len 6);
  - block_len 6 spreading prefers the log form.
- *Not yet:*
  - block_len 1 at criticality: the grid is too coarse;
  - crossing times: too few inactive-side values;
  - finite-size z: needs ε_c first.

**New data, in priority order.** Each item is a knob change in a copy of an existing generator and submit script.
1. **N1. Finer ε grids around both critical points, decay and spreading, at the same ε values.**

   | | ε_u values | ε̄ range | covers |
   |---|---|---|---|
   | block_len 1 | 0.5955–0.5985 in steps of 0.0005 (7 values) | 0.14292–0.14364 | decay estimate ≈ 0.1430–0.1432 and spreading estimate ≈ 0.1432–0.1436 |
   | block_len 6 | 0.4580–0.4598 in steps of 0.0003 (7 values) | 0.10992–0.11035 | ε̄_c ≈ 0.1100–0.1104 |

   - Same sizes and sample counts as before.
   - Cost: a few hours per decay copy and 1–2 h per spreading copy.
   - This is what makes the critical-point tests (straightness/κ, offset-corrected δ_eff, and per-decade δ) meaningful for block_len 1.
2. **N2. Extra inactive-side values for crossing times:** about 4–5 values spanning r ≈ 0.002–0.03 above each ε_c. Spreading is the cheapest place to do this.
3. **N3. Block-end output times for block_len 6:** round the log grid to multiples of `block_len` in the block_len 6 copies (decay and spreading) used for N1/N2. This removes the ±2.5% block-phase sawtooth at the source.
4. **N4. Finite-size runs** (`_fss_bvh_b1`, plus a `_fss_bvh_b6` copy) once N1 pins ε_c. The lifetime τ ∝ L^z is a test that doesn't depend on the offset: z = 1 (BVH) vs 1.58 (DP).
5. **Not planned: longer runs.** Doubling ln t + c would need ~500× longer runs and larger L.

**Analysis changes** (no new data; best done before N1 lands so the new runs are read correctly):
- **A1. Block-phase correction for block_len > 1.** Divide out one global phase pattern, or use block-end times only. Needed for the existing block_len 6 data; N3 makes it unnecessary for new runs.
- **A2. BVH reference curves with the fitted offset.**
  - For δ_eff and α_eff, use the fitted 1/A = B(ln t + c) at `CONTROL_C` instead of the grey 1/ln t curve and the "α_eff → 1" line.
  - Use the same offset for the spreading slope-1 guide and the 1 − y/ln t guides.
  - Relabel the fit-table column `ln t0 = a/B` as the offset c (= −ln t₀).
- **A3. Box–Cox straightness scan,** (A^(−κ) − 1)/κ against ln t, per ε: κ = 0 is a power law, κ = 1 is BVH. Also scan between adjacent grid values by interpolating ln A. Apply to decay A and spreading P_s.
- **A4. Decade-by-decade power-law δ table** (as above) for decay and spreading. It is the most intuitive crossover indicator: a fixed point gives a constant δ.
- **A5. Early- vs late-time local exponents** of δ, θ and 1/z for all ε. Early times show each model's character before off-criticality matters.
- **A6. Notebook text:**
  - give the regime (active, critical, inactive; "BVH only once ln t ≫ c") for every statistic;
  - explain that the clean spread of ln A is finite-chain noise ∝ t^(1/(2z)).

**Spreading specifically.**
- The block_len 6 run already gives the strongest evidence; N1–N3 sharpen it (ε_c, crossing times, no sawtooth).
- The block_len 1 spreading run needs N1 before it can say anything at criticality.
- No changes are needed for the clean run.

**Lessons for the spin chain, which we expect to be in the crossover regime too.**
- **Don't rely on any single fit.** A power law over 2–3 decades can be found almost anywhere. Report:
  - the drift of the running exponents (decade-by-decade δ);
  - the κ straightness indicator;
  - early-time local exponents.
- **Placing the spin chain on the ladder.** Its reported δ ≈ 0.11 and z ≈ 1.34 (1/z ≈ 0.75) sit close to Stavskaya block_len 1 (δ ≈ 0.10, 1/z ≈ 0.75).
- **The convincing spin-chain test is a disorder-strength knob** (see "Plans for later"). The exponents should move with disorder strength in the same direction as Stavskaya's.
- **A spreading-type observable for the spin chain.** If the existing OTOC data (`get_OTOC_data.jl`) give a light cone, its front could play the role of R(t).

### 7.19 Assessment of the Stavskaya next steps, with checks on the data (2026-10-05)

You asked whether the takeaways in the review file are justified, which observables are most promising, and whether N1–N4/A1–A6 (snapshot in §7.18) are the right next steps. Everything below was computed from the existing data. Decay means come from `_claude_scratch/summary.json` and spreading from the raw chunks. No notebook, generator or data file was changed, and nothing was submitted.
- **Checks:** `stavskya_mc/block_disorder/analysis/_claude_scratch/review_checks_2026_10_05.py`, sections [1]–[8], about 4 s:
  ```bash
  cd stavskya_mc/block_disorder/analysis && python3 _claude_scratch/review_checks_2026_10_05.py
  ```
- **Local Julia pilot:** `_claude_scratch/coupling_pilot_2026_10_05.jl`. Production kernel, L = 4000, 300 samples, about 40 s; nothing written.

**1. The decay mean and the spreading survival are the same observable [1].**
- **The identity.**
  - Site (j, t) is active iff a backward path of open (unhealed) sites leads to an active site at t = 0.
  - Reversing time turns that path into the forward cluster of a single seed with the disorder sequence reversed.
  - So per disorder history ρ_full(t) = (1 − ε_t) P_s^rev(t − 1). For i.i.d. steps (clean, block_len 1) the average is ⟨ρ_full(t)⟩ = (1 − ε̄)⟨P_s(t − 1)⟩, exactly.
  - The decay runs start half filled and reach the fully active curve once the backward cluster is large.
- **Data.**
  - A(t)/[(1 − ε̄)P_s(t − 1)] is 1 within ≈ 2% for t ≥ 10² at every clean and block_len 1 grid value. At block_len 1, ε̄ = 0.1428: 0.999 ± 0.004, 0.990 ± 0.005, 0.981 ± 0.006 at t = 10², 10³, 10⁵.
  - At t = 10 it is 0.93–0.98, from the half-filled start.
  - Block_len 6: a constant 1.02–1.04 from t = 10². The first reversed block shares its ε with the factor 1 − ε_t. The time dependence is the same.
- **Consequences.**
  - "Decay weak, spreading decisive" at block_len 6 compares two estimates of one curve. The difference comes from the decay's block-phase sawtooth and error model, so these are not two pieces of evidence.
  - Decay runs at the same ε as spreading runs duplicate the mean. Decay is needed only for what spreading can't give:
    - per-history distributions (width of ln A);
    - FSS;
    - coupled ε-differences (item 4).

**2. One observable alone cannot fix ε_c and the functional form at once [2], [3].**
- **The test.** ln A is interpolated between measured neighbours. For each form, the ε where it fits best is found (κ = 0 power law, κ = 1 BVH).

| | window | power law: best ε̄, χ²ᵣ | BVH: best ε̄, χ²ᵣ |
|---|---|---|---|
| clean decay | 10³–10⁵ | 0.29451, 0.97 | 0.29430, 102 |
| clean P_s | 10³–10⁵ | 0.29450, 0.06 | 0.29430, 0.78 |
| block_len 1 decay | 10³–10⁵ | 0.14305, 1.78 | 0.14295, 1.45 |
| block_len 1 P_s | 10³–10⁵ | 0.14305, 1.08 | 0.14300, 0.67 |
| block_len 6 P_s | 10³–10⁵ | 0.11045, 0.27 | 0.11020, 0.12 |
| block_len 6 P_s | 10²–10⁵ | 0.11050, 1.81 | 0.11030, 0.16 |
| clean decay | 10²–10⁵ | 0.29450, 43.7 | 0.29420, 3641 |

- **On 10³–10⁵ each form fits at its own ε, 0.00005–0.00025 apart.**
  - A slightly inactive curve near a BVH point bends like a power law. A slightly active curve near a power-law point bends like a log.
  - P_s alone cannot even reject BVH for clean DP (χ²ᵣ 0.78). Only the clean decay, with its tiny errors, can.
- **The BVH preference comes only from 10²–10³.** There even the clean decay fails its own correct power law (χ²ᵣ 44, from the half-filled start).
- **A finer grid measures the same two curves.** So the block_len 1 "ambiguity" is not a matter of grid resolution.
- **The interpolated block_len 1 κ values depend on method and window.** At ε̄ = 0.1430:
  - linear, 10²–10⁵: κ = 1.20 (reproduces §7.17);
  - quadratic: κ = 1.75;
  - linear, 10³–10⁵: κ = 0.45 (χ²ᵣ 1.80 power law vs 1.92 BVH).

  Don't quote them.

**3. Joint test across observables: no conventional critical point on 10³–10⁵ [4], [8].**
- **The test.** A conventional critical point needs one ε_c where P_s, N and R are all power laws. No curve interpolation is needed:
  - fit ln O = a + b u + q u² (u = ln t − ⟨ln t⟩) at every measured grid value;
  - find where the curvature q crosses zero in ε̄;
  - bootstrap over chunks.

| zero-curvature ε̄ (10³–10⁵) | P_s | N | R | A (decay) |
|---|---|---|---|---|
| clean | 0.29450(1) | 0.29451(1) | 0.29452(3) | 0.29451 |
| block_len 1 | 0.14306(1) | 0.14332(1) | 0.14388(3) | 0.14303 |
| block_len 6 | 0.11043(1) | 0.11061(1) | 0.11108(2) | 0.11048 |

- **The pattern holds in every window tried.** On 3×10³–10⁵ and 10³–3×10⁴ the gap between P_s and R is:
  - 0.0008 at block_len 1;
  - 0.0005–0.0009 at block_len 6;
  - ≤ 0.0001 for clean, at Mendonça's 0.29450(5).
- **Where P_s is a power law, R is not.**
  - As a power law R has χ²ᵣ 89 (block_len 6) and 225 (block_len 1); with a log correction 0.5 and 2.8.
  - Its local exponent 1/z_eff rises between 10³ and 10⁵: 0.828 → 0.872 (block_len 6, 0.1104) and 0.760 → 0.859 (block_len 1, 0.1428). Clean: 0.636 → 0.634 ± 0.003.
- **Conversely, R is straight where P_s is clearly inactive.** At block_len 1, ε̄ = 0.1440, 1/z_eff = 0.745–0.751 ± 0.002 over 10³–3×10⁴, i.e. z = 1.34.
  - That is the spin chain's reported z. It shows how convincing an effective exponent can look at a slightly wrong ε.
- **Where P_s takes BVH's form, all three curvatures are positive.**
  - y_R = 1.56 (block_len 6) and 2.00 (block_len 1); BVH found 1.7 for the contact process.
  - y_N = 3.1 at block_len 1 (BVH 3.6). At block_len 6, N fits poorly (χ²ᵣ 25), probably the block-phase sawtooth, so its y_N = 2.6 is not reliable.
- **Reading.**
  - The exponents measured in this window are not fixed-point exponents.
  - BVH's forms fit all observables at a common ε, with log exponents close to BVH's.
  - **Regime:** this says nothing about ln t ≫ c.
  - **Loophole:** a conventional point with large, observable-specific corrections to scaling. That would still make the paper's exponents effective rather than universal.
- **Assumption.** The zeros are interpolated in q between grid values 0.0012 apart.
  - At block_len 1 this is safe. q_R at the P_s zero is +9 ± 0.3 (×10⁻³), and the P_s curvature at the measured 0.1440 is −24 ± 0.5.
  - A measured fine grid removes the assumption.

**4. The ε-response exponent: the paper's ν_t, measured locally in time [5] and in the pilot.**
- **Definition.** χ(t) = −∂ ln A/∂ε̄, from central differences of neighbouring curves. Its local exponent d ln χ/d ln t is:
  - 1/ν_∥ for a power law (scaling variable r t^(1/ν));
  - 2/(ln t + c′) for BVH (variable r (ln t)²). Then ν_eff = (ln t + c′)/2, growing with slope ½ in ln t.
- **Clean check.** 0.53–0.62 at every t from 10² to 10⁵ at ε = 0.2945 (DP 0.577).
- **Disordered, on the present grid.** It is linear only up to ≈ 3×10³, where the forward/backward ratio is ≲ 1.4.
  - Block_len 6 P_s at 0.1104: 0.59, 0.53, 0.44, 0.41 at t = 10², 3×10², 10³, 3×10³.
  - Block_len 1 P_s at 0.1428 / 0.1440: 0.62 / 0.58, 0.55 / 0.55, 0.46 / 0.52, 0.43 / 0.51.
  - So ν_eff rises from DP-like values to ≈ 2.4 and passes the paper's 2.25 near t ≈ 10³. That is slower than BVH's asymptotic slope (½; observed ≈ 0.2).
  - Later times are nonlinear on the 0.0012 grid (ratio 2–4). So it is open whether ν_eff keeps growing (crossover) or levels off (fixed point).
- **Why this observable.**
  - It compares O(1) numbers: 0.44 (paper) against ≲ 0.2 (BVH at t ≳ 10⁴).
  - Relative to that gap, an error in ε_c moves it ~1/(ν δ_eff) ≈ 7× less than it moves δ_eff.
- **Why it needs coupled runs.** Linearity needs a fine grid, and independent samples can't resolve differences that small.
- **Pilot.** Block_len 6, Δε̄ = 0.00012, every sample run at all three ε with one seed vs independent seeds.
  - The coupled samples are pathwise monotone in ε (Stavskaya is attractive).
  - Errors of χ are 17× smaller at t = 4000, 38× at 10³ and 70–300× earlier: 30.1 ± 2.5 vs 67 ± 172 at t = 272.
  - The coupled pilot gives 1/ν_eff = 0.52, 0.45, 0.48 over t ≈ 10²–4×10³.
- **No kernel change is needed.** The decay kernel draws exactly L uniforms per step plus one per block, independent of ε. One `Random.seed!` per sample in the generator suffices.

**5. Which ladder numbers are robust [6].** Early times (t = 50–400), across each whole grid:

| | 1/z_eff | θ_eff | δ_eff |
|---|---|---|---|
| clean | 0.638–0.639 | 0.305–0.315 | 0.152–0.158 |
| block_len 1 | 0.732–0.760 | 0.416–0.575 | 0.080–0.158 |
| block_len 6 | 0.824–0.841 | 0.603–0.708 | 0.054–0.103 |

- **1/z_eff, and roughly θ_eff, give a ladder that doesn't depend on ε_c.**
- **Early δ_eff does depend on ε_c.** At block_len 1's inactive edge it equals the DP value.
- **§7.16's "not DP at any time … for every ε"** holds for 1/z and θ, not for δ.

**6. Per-decade δ [7].**
- **No sawtooth in the end-point ratios.** All decade end points 10^k have block phase (t − 1) mod 6 = 3.
- **Errors are ±0.0013–0.0018 (P_s) and ±0.002–0.003 (decay).** Block_len 6 P_s at ε̄ = 0.1104 gives 0.0701, 0.0610, 0.0572, so the last step is ≈ 1.7σ.
- **Decades 2–3 can show a constant δ at one ε in each rung:**
  - block_len 6: ≈ 0.062 at ε̄ ≈ 0.1104–0.1105;
  - block_len 1: ≈ 0.076 at ε̄ ≈ 0.1431.

  Only the excess in decade 1 holds at every ε, and corrections to scaling could also produce it.

**7. Width of ln A (summary.json).** Its slope over 10³–10⁵ rises monotonically through each grid: −0.001 to 0.149 at block_len 6, 0.001 to 0.20 at block_len 1. It has the same degeneracy as δ_eff.

**8. Crossing times.**
- Clean P_s, with ε* known to ±0.00005, gives ν = ln(79433/17783)/ln 2 = 2.16 against DP's 1.73.
- Moving the assumed ε_c by ±0.00005 turns the r ratio 2 into 3 or 1.67, which gives ν = 1.36 or 2.93.
- ε_c depends on the hypothesis at the 0.0002–0.0003 level (item 2). So crossing times at r ≈ 0.002 are undefined.
- The response exponent carries the same information without a threshold or a reference curve.

**Verdicts on the takeaways (as of §7.18):**

| takeaway | verdict |
|---|---|
| clean = DP | justified; items 3 and 4 confirm it |
| exponents move monotonically with disorder strength | justified for 1/z and θ; the δ column depends on ε_c. On its own this shows non-universality, not BVH: a line of disorder-dependent fixed points would also do it |
| block_len 6 P_s "prefers BVH by the same test as clean" | not justified: with ε_c free the test is degenerate on 10³–10⁵; the clean test worked because ε* was known independently |
| block_len 6 decay "consistent with BVH, can't exclude a power law" | true, but it is the same observable as P_s (item 1) |
| block_len 1 "ambiguous; grid 6× too coarse" | the numbers reproduce, the diagnosis doesn't: the degeneracy is intrinsic, and the interpolated κ depend on method and window |
| per-decade δ "constant only for clean" | partly: a constant δ over decades 2–3 exists at one ε in each rung; only the decade-1 excess holds at every ε |
| working reading: crossover toward BVH | better supported than stated, but by item 3, not by P_s straightness |
| paper's exponents ≈ block_len 1's | weak: δ depends on ε and window; the paper's disorder distribution differs; the spin chain starts at DP (§3.3) and block_len 1 doesn't |

**Verdicts on the proposed steps.** The revised plan is in `REFEREE_CONFLICT_REVIEW.md`.
- **N1: replace.**
  - Its purpose fails (item 2).
  - The decay half duplicates the spreading mean (item 1).
  - Uncoupled fine grids can't give the response exponent (item 4).
  - Its ranges sit on the ε_c favoured by BVH. They miss where N and R are power laws (block_len 1 up to 0.1440, block_len 6 up to 0.1111), which item 3 needs.
- **N2: drop** (item 8).
- **N3: keep;** it is free.
- **N4: drop or defer.**
  - Under BVH, FSS sees an effective z ≈ 1 + y_R/ln L, well above 1 at L ≤ 1.6×10⁴; the test is not "1 vs 1.58".
  - Lifetimes near ε_c are dominated by temporal Griffiths effects, τ ~ L^(1/κ(ε)). So the fitted exponent depends on ε_c at first order.
  - Spreading R already gives z_eff, without finite-size effects.
- **Longer runs:** agreed.
- **A1:** optional.
- **A2:** a presentation fix only. "BVH with the fitted offset reproduces δ_eff and α_eff" is the log-derivative of the same 1/A fit, not independent evidence.
- **A3:** drop the interpolation and the per-ε verdicts; the joint test replaces them.
- **A4:** keep, with errors and the decade-2–3 caveat.
- **A5:** keep, with claims restricted to 1/z and θ.
- **A6:** keep.

### 7.20 Bringing the spin chain and Stavskaya together (2026-10-05)

You reread the paper, Mendonça (arXiv:1011.1489) and BVH (arXiv:1603.08075), and asked how to bring the two stories together for the resubmission, whether the spin chain needs a block length (which might not solve the soliton issue), and what the next best step is. The proposal is in `REFEREE_CONFLICT_REVIEW.md`. No notebook, generator or data file was changed, and nothing was submitted.
- **Checks:**
  - Stavskaya: new section [9] of `stavskya_mc/block_disorder/analysis/_claude_scratch/review_checks_2026_10_05.py`, about 3 s:
    ```bash
    cd stavskya_mc/block_disorder/analysis && python3 _claude_scratch/review_checks_2026_10_05.py 9
    ```
  - Spin chain: `heisen_spin_chain/_claude_scratch/spin_checks_2026_10_05.py`, sections [1]–[3], under 1 s:
    ```bash
    python3 heisen_spin_chain/_claude_scratch/spin_checks_2026_10_05.py
    ```
    Sections [1]–[2] read the old local means (`data/s_diff_per_time/N4/a*/IC1700/L2000`: L = 2000, 1700 samples, every step to t = 10⁴, first-octant initial states).

**1. The paper's own Stavskaya model is not on the ladder yet.**
- Fig. 4 and Table I use ε(t) = 0.5Xⁿ with X uniform on [0, 1], redrawn every step (`time_random_n_evolve_state`; `get_time_random_data_n_time_data2.jl`). With ε̄_c = 0.253725, n ≈ 0.97.
- Its disorder is weak. By the spread of ε per step (std/mean, all block_len 1):

  | model | std/mean of ε |
  |---|---|
  | paper's model | 0.57 |
  | W1 (p = 0.8, ratio 20) | 0.47 |
  | block_len 1 (p = 0.2, ratio 20) | 1.58 |

  So it is the weak rung W1 was meant to supply. Its reported δ = 0.105 and 1/z = 0.69 (from FSS) lie between clean and block_len 1.
- It has not been through the spreading runs or the joint test. Its decay data (L = 20000, t to 2×10⁶, every 2000 steps) give P_s by duality, but only from t = 2000.

**2. The J-sign randomness of the spin chain is a sequence of random symmetry kicks (exact) [3].**
- Each sign pair's step equals the (+,+) step conjugated by a site-wise symmetry G that fixes the target spiral S⁰ and commutes with the push:
  - (−,−): G = π rotation about z on odd sites;
  - (+,−): G = K∘R∘Y, with Y = π rotation about y on odd sites, R = reflection j → −j, and K: S_j → −S_{j+2}. K is anti-canonical, so it turns the antiferromagnet's forward step into the ferromagnet's;
  - (−,+): G = K∘R∘Y∘Z.
- Checked on L = 12 with the code's equations of motion: the identity holds to 5×10⁻¹⁶, while each step differs from the (+,+) step by O(1).
- So the dynamics is one deterministic map (Heisenberg ferromagnet for τ = 1, then the push), with a random target-fixing symmetry kick between steps. The kicks never change a, the energy scale, the target or S_diff; they only scramble the state relative to the Hamiltonian. Two of the four draws give no kick.
- **Consequences:**
  - The chain's temporal disorder couples to the distance from criticality only indirectly, through stretches of coherent versus scrambled evolution. That is weak disorder in BVH's sense, consistent with δ_eff starting at the DP value (§3.3).
  - Holding the signs fixed for b steps means fewer kicks, i.e. longer stretches of the fixed Heisenberg chain, the setting where the solitons of Fig. S5 appear. It moves the model toward the soliton-supporting clean chain, not toward stronger disorder. In Stavskaya, block_len lengthens excursions of the distance to criticality itself; the J signs have no such role.
  - A spin-chain knob that maps one-to-one onto Stavskaya's ε(t) is a time-random push strength a(t) (binary a_u/a_l with probability p, held for block_len steps), with the J signs still redrawn every step so that solitons stay suppressed.

**3. The joint test still separates in short windows [9].** The ε̄ at which ln O is straight in ln t, with chunk bootstrap errors, and the spread of the three values as a fraction of ε_c:

| window | clean | block_len 1 | block_len 6 |
|---|---|---|---|
| 10²–10³ | 0.05% | 0.25% | 1.22% |
| 10²–3×10³ | 0.06% | 0.41% | 0.76% |
| 3×10²–3×10³ | 0.06% | 0.41% | 1.18% |
| 3×10²–10⁴ | 0.04% | 0.58% | 0.86% |
| 10³–10⁵ | 0.01% | 0.57% | 0.59% |

- The disordered splits are 4–30× the clean one in every window. The individual errors are 0.00003–0.0003.
- In block_len 1, the point where P_s is a power law moves toward the active side as the window moves later: 0.14382, 0.14360, 0.14339 and 0.14306 for the windows starting at 10², 10², 3×10² and 10³. The clean value doesn't move beyond ±0.0001.
- **For the spin chain:** the joint test works in windows a spin-chain run can reach (t ≲ 10⁴). But a split of 0.3–0.6% of a_c is ±0.002–0.005 in a, so the a grid must be about 10× wider than the v2 S_diff grid (±0.0009).

**4. The spin chain's a-response exponent is measurable, but not yet decisive [1].** χ(t) = d ln⟨S_diff⟩/da is a weighted linear fit across a, and 1/ν_eff = d ln χ/d ln t over a factor √10.

| a values | t = 10³ | 3×10³ | 9×10³ | linear in a? |
|---|---|---|---|---|
| narrow, 0.7570–0.7595 (9) | 0.42 ± 0.12 | 0.52 ± 0.08 | 0.40 ± 0.06 | yes (χ²ᵣ ≤ 2.5) |
| wide, 0.7535–0.7615 (13) | 0.50 ± 0.04 | 0.41 ± 0.03 | 0.32 ± 0.02 | no (χ²ᵣ 8–63) |
| outer pair 0.7555/0.7595 | 0.57 ± 0.10 | 0.53 ± 0.06 | 0.52 ± 0.04 | secant |

- References: DP 0.577; the paper's ν_t = 2.24 gives 0.446.
- The narrow (valid) set is comparable to Stavskaya block_len 1 at the same times (0.46–0.52 at 10³, 0.43–0.51 at 3×10³, §7.19). It fits both a constant near the paper's value and a slow fall; errors must shrink about 3× to decide.
- The v2 S_diff grid (7 values, ±0.0009, 2000 samples) would give errors about 1.5× larger than the narrow set.
- Caveat: these data have the first-octant initial states.

**5. Sample spread suggests a global component, not yet attributable [2].**
- std(S_diff)/⟨S_diff⟩ at a = 0.758 is 0.21, 0.31 and 0.39 at t = 30, 100 and 300. The finite-chain estimate √(ξ/L), with ξ = t^0.75, is 0.08, 0.13 and 0.19.
- The excess is a spatially coherent fluctuation. It could come from the shared disorder history, which coupled samples would cancel. It could also come from the old initial states' net magnetization |m| ≈ 0.89. A coupled pilot decides.

**6. What the running v2 spin data can and can't give.**
- **The v2 S_diff copy** (L = 2000, every 200 steps, t ≤ 4×10⁴):
  - δ_eff(T) = log₁₀[S(T/10)/S(T)] is available only from T = 2000, so the early departure from DP isn't visible in it;
  - its last decade passes L^z ≈ 2.7×10⁴ (z = 1.34).
- **The v2 Lyapunov copies** (`get_good_data_severalL*`, every step, t ≤ L^1.7, 1000 samples at each L = 32–512 sites; L = 512 is split over L5 and L6) supply what the S_diff copy can't:
  - early-time δ_eff at L = 256 and 512;
  - a response estimate from their near-critical a values (0.7550–0.7610) at about the old data's precision.

**Superseded text:** the "Lessons for the spin chain" block of `REFEREE_CONFLICT_REVIEW.md` (written 2026-10-05 after §7.19), as it stood before this section:

> **Lessons for the spin chain, which we expect to be in the crossover regime too:**
> - **One observable's straightness is degenerate with a_c**, as shown above for Stavskaya. Drop the κ indicator. Report the drift of the running exponents, but don't use it as the test.
> - **Response exponent.** ν_eff(t) from ∂ ln S_diff/∂a between neighbouring a values of the v2 runs tests ν_t = 2.24 directly.
>   - Sharing the J-sign sequence and initial state across a values (one seed per sample) may cut its noise. Chaos decorrelates the microscopic states, so measure the gain first.
> - **A second observable makes J1 possible:** a single-site perturbation spreading run, i.e. the OTOC/decorrelator front (`get_OTOC_data.jl`).
> - **Placing the spin chain on the ladder.** Match on 1/z, which is robust (z ≈ 1.34 ↔ block_len 1's 1/z_eff ≈ 0.75), not on δ.
>   - Its δ_eff starts at the DP value, so it sits at weaker effective disorder than block_len 1, near W1.
> - **The convincing spin-chain test is still the disorder-strength knob** (Plans for later).

### 7.21 What the last Stavskaya runs show directly, and the runs that make the story plottable (2026-10-05)

You asked for the final Stavskaya takeaway, and which new runs would give plots that show it on measured data rather than on extrapolation. You also asked to save the spin-chain plans (SC1, SC2) until the running v2 data arrive. The figure plan and runs are in `REFEREE_CONFLICT_REVIEW.md`. No notebook, generator or data file was changed, and nothing was submitted.
- **Checks:** new sections [10] and [11] of `stavskya_mc/block_disorder/analysis/_claude_scratch/review_checks_2026_10_05.py`, about 10 s:
  ```bash
  cd stavskya_mc/block_disorder/analysis && python3 _claude_scratch/review_checks_2026_10_05.py 10 11
  ```
- **Notebooks:** the saved figures of `analyze_spreading{,2,3}.ipynb` and `analyze_time_log{,2,3}.ipynb` (last run 2026-10-04) were reread.

**1. What the figures show.**
- **Block_len 6 spreading** (`figs/spreading_p0.2_bl6_*`):
  - At ε̄ = 0.1104, 1/P_s is nearly a straight line in ln t from ln t ≈ 3 to 11.5; its slope B_eff rises only from 0.11 to 0.125.
  - 1/δ_eff follows the BVH slope-1 guide from ln t ≈ 4 to 10, then flattens near 17.
  - 0.1116 is clearly inactive from ln t ≈ 6; 0.1092 and 0.1080 are active.
  - At 0.1104, 1/z_eff rises slowly (0.82 → 0.87) and θ_eff is noisy, around 0.65–0.8.
- **Block_len 1 spreading** (`figs/spreading_p0.2_bl1_*`):
  - No grid value is critical: 0.1428 is active and 0.1440 inactive.
  - At 0.1440, R is a clean power law (1/z_eff ≈ 0.745 from ln t ≈ 4 to 11.5) while P_s and N bend down. This is why the notebook's verdict looked "qualitatively different from both DP and BVH".
- **Block_len 6 decay** (`figs/window_binary_L100000_bl6_*`): the block-phase sawtooth dominates every local-slope panel (±0.1 in d(1/A)/d ln t). These panels are not usable as figures without N3, and by duality spreading P_s is the same curve without the sawtooth.
- **Block_len 1 decay:** the width of ln A saturates at 0.6–0.8 on the active side and grows on the inactive side. As found in §7.19, it moves monotonically through the grid, so it can't fix the critical point.

**2. What is measured and what is interpolated.**
- **The interpolation.** In each disordered rung, all three zero-curvature points of the joint test lie inside a single grid interval:
  - block_len 1: [0.1428, 0.1440];
  - block_len 6: [0.1104, 0.1116].

  The positions in §7.19 item 3 come from interpolating the curvature linearly across that interval.
- **The measured version (section [10]).** At every grid value: the local exponent of P_s, N and R over 10³–10⁴ and over 10⁴–10⁵, and its drift (zero for a power law). Chunk bootstrap errors.

  | | ε̄ | δ_eff drift | θ_eff drift | 1/z_eff drift |
  |---|---|---|---|---|
  | clean | 0.2944 | +0.025 ± 0.002 | +0.056 ± 0.004 | +0.010 ± 0.003 |
  | clean | 0.2945 | −0.000 ± 0.003 | +0.005 ± 0.004 | +0.002 ± 0.003 |
  | clean | 0.2946 | −0.022 ± 0.003 | −0.042 ± 0.004 | −0.007 ± 0.003 |
  | block_len 1 | 0.1428 | +0.030 ± 0.001 | +0.126 ± 0.003 | +0.056 ± 0.001 |
  | block_len 1 | 0.1440 | −0.111 ± 0.003 | −0.161 ± 0.005 | −0.007 ± 0.002 |
  | block_len 6 | 0.1104 | +0.004 ± 0.001 | +0.034 ± 0.003 | +0.024 ± 0.001 |
  | block_len 6 | 0.1116 | −0.117 ± 0.002 | −0.192 ± 0.004 | −0.019 ± 0.001 |

  The δ_eff column is the drift of the slope of ln P_s, so positive means P_s flattens.
  - **Block_len 6, measured without interpolation:** at 0.1104, P_s is nearly a power law (δ_eff drift +0.004), while N and R still drift by 11σ and 24σ. Clean at ε* has all three drifts below 0.005.
  - **Block_len 1:** the measured statement is weaker. At 0.1440, R is a power law while P_s and N are far off. Whether R is still drifting at the P_s point needs measured values between 0.1428 and 0.1440 (N1-S).
  - R's exponent is much less sensitive to ε than P_s's or N's, in clean as well. So a figure has to show the zeros themselves, not one drift at one ε.

**3. The response exponent: a correction (section [11]).**
- **Method.** χ = −∂ln P_s/∂ε̄ is taken at a point from a quadratic in ε̄ through three grid values at each time. Then 1/ν_eff = d ln χ/d ln t per half decade.
- **Clean check:** 0.57–0.62 at 10⁴–10⁵ (DP: 0.577). Early times are noisy, since the grid is narrow.

| | t = 3×10² | 10³ | 3×10³ | 10⁴ | 3×10⁴ | 10⁵ |
|---|---|---|---|---|---|---|
| block_len 1 at 0.14306 | 0.55 | 0.47 | 0.44 | 0.41 | 0.39 | 0.42 |
| block_len 1 at 0.1434 | 0.54 | 0.48 | 0.45 | 0.45 | 0.43 | 0.46 |
| block_len 6 at 0.11043 | 0.53 | 0.44 | 0.41 | 0.39 | 0.39 | 0.42 |
| block_len 6 at 0.1108 | 0.57 | 0.44 | 0.43 | 0.42 | 0.44 | 0.48 |

- Statistical errors are 0.01–0.05. The systematic error dominates: the quadratic spans 0.0024, and late times are nonlinear (χ·Δε̄ ≈ 0.6–0.8 at 10⁵). The value also depends on where it is taken, e.g. 0.42 vs 0.48 at 10⁵.
- **Reading.** 1/ν_eff falls from DP-like values to ≈ 0.4–0.45 by t ≈ 10³–10⁴, then is roughly flat, or ticks up, to 10⁵. That is ν_eff ≈ 2.1–2.5, close to the paper's 2.25, in both rungs.
- **This corrects** §7.19 item 4 / the review's "ν_eff passes 2.25 near 10³ and is still moving". The earlier J2 numbers stopped at 3×10³.
- **BVH asymptotics** (χ ∝ (ln t + c′)²) give 1/ν_eff = 2/(ln t + c′): 0.29 → 0.17 over 10³–10⁵ for c′ = 0, and ≈ 0.13–0.10 with the P_s offset c ≈ 8.3. The window does not show activated scaling in the response.
- **Consequence for the story:** ν_t ≈ 2.25 is what the temporal-disorder crossover gives in this window, including at BVH's own parameters (block_len 6). So it is not evidence of a new fixed point. It is also not, by itself, evidence for BVH.
- **Error estimate for N1-S.** Central differences over 2δ = 0.0004 with 10⁵ runs give χ to ≈ 13–15% at 10³, 5–7% at 10⁴ and 2–2.5% at 10⁵. That is useful from about 3×10³; earlier times need the coupled N1-C.

**4. Plan** (review file): figures F1–F5, and runs N1-S (unchanged), N1-C (grids fixed; clean added as the control line) and P1.

**Superseded text:** the Stavskaya status of `REFEREE_CONFLICT_REVIEW.md` ("Where things stand", Stavskaya bullet) as it stood before this section:

> - **Stavskaya** (results H §7.16–7.17; checks H §7.19; script `stavskya_mc/block_disorder/analysis/_claude_scratch/review_checks_2026_10_05.py`):
>   - **Clean control: DP.**
>     - P_s, N, R and the decay all become power laws at one ε̄_c = 0.29450.
>     - The ε-response exponent is 1/ν = 0.53–0.62 at all times (DP: 0.577).
>   - **Non-universality, robust part.** Early-time (t = 50–400) 1/z_eff, across each whole ε grid:
>     - clean 0.638;
>     - block_len 1: 0.73–0.76;
>     - block_len 6: 0.82–0.84.
>
>     θ_eff behaves the same way. Early δ_eff does not: it depends on ε (block_len 1: 0.08–0.16 across its grid).
>   - **Decay mean = spreading survival.** ⟨A(t)⟩ = (1 − ε̄)⟨P_s(t − 1)⟩ exactly for block_len 1; the data agree within ≈ 2% from t = 10². Block_len 6 differs by a constant 1.03. So decay and spreading are one observable, not two pieces of evidence.
>   - **One observable can't decide.**
>     - With ε_c free, P_s (or A) fits both a power law and BVH on 10³–10⁵, at ε values 0.00005–0.00025 apart.
>     - The BVH preference comes only from 10²–10³.
>     - The interpolated block_len 1 κ values depend on the interpolation method and the window. Don't quote them.
>   - **Joint test (new, the strongest evidence).** The ε̄ at which each observable is a power law on 10³–10⁵:
>
>     | | P_s | N | R |
>     |---|---|---|---|
>     | clean | 0.29450 | 0.29451 | 0.29452 |
>     | block_len 1 | 0.14306 | 0.14332 | 0.14388 |
>     | block_len 6 | 0.11043 | 0.11061 | 0.11108 |
>
>     - Bootstrap errors are ≤ 0.00003, and the pattern is the same in every fit window tried.
>     - Where P_s is a power law, R is not: power-law χ²ᵣ 89–225, and 1/z_eff still rises by 0.04–0.10 over 10³–10⁵.
>     - Where P_s takes BVH's form, R needs y_R = 1.6–2.0 (BVH: 1.7).
>     - So no conventional critical point describes 10³–10⁵, while BVH's forms fit every observable at one ε.
>     - **Regime:** the critical region at ln t ≈ 7–11.5, not yet ln t ≫ c.
>   - **ν_t (preliminary).** The ε-response exponent 1/ν_eff falls in both rungs, from ≈ 0.6 at t ≈ 10² to 0.41–0.51 at t ≈ 3×10³. So ν_eff passes the paper's 2.25 near t ≈ 10³ and is still moving. Later times need a finer, coupled grid.
>   - **Working reading.**
>     - The paper's exponents are effective values in a crossover whose functional forms are BVH's.
>     - "Close to block_len 1" is weak: δ depends on ε and window, and the spin chain starts at DP while block_len 1 doesn't.

and its "Stavskaya next steps" section:

> ## Stavskaya next steps (revised 2026-10-05 after H §7.19; waiting for your go-ahead)
>
> **Verdict on the earlier proposal (H §7.18; reasons in H §7.19):**
>
> | item | verdict | main reason |
> |---|---|---|
> | N1 fine grids, decay + spreading | replace by N1-S, N1-C | finer grids can't break the ε–form degeneracy; decay duplicates the P_s mean; uncoupled grids can't give the response exponent; the ranges miss where N and R are power laws |
> | N2 crossing times | drop | needs ε_c better than the smallest r. Even clean DP with ε* known gives ν = 2.16 (true 1.73), and moving ε_c by ±0.00005 shifts it to 1.36–2.93 |
> | N3 block-end times | keep | free; use in every new block_len 6 copy |
> | N4 FSS lifetimes | drop or defer | under BVH, FSS sees z_eff ≈ 1 + y_R/ln L, not 1; temporal-Griffiths lifetimes make it depend on ε_c at first order; spreading R already gives z_eff |
> | longer runs | not planned (agree) | — |
> | A1 block-phase correction | optional | use P_s or the decade end points (all at the same block phase) instead |
> | A2 offset reference curves | presentation only | "BVH with offset reproduces δ_eff, α_eff" restates the same 1/A fit; it is not evidence |
> | A3 κ scan + interpolation | drop interpolation and per-ε verdicts | replaced by J1 |
> | A4 per-decade δ | keep, with errors and caveat | a constant δ over decades 2–3 exists at one ε in each rung |
> | A5 early vs late exponents | keep | restrict the "every ε" claims to 1/z and θ |
> | A6 notebook text | keep | — |
>
> **Analysis (no new data), first:**
> - **J1. Joint zero-curvature test** of P_s, N and R in the spreading notebooks, with the chunk bootstrap. Code: sections [4] and [8] of the script. It becomes the headline result.
> - **J2. ε-response exponent cell:** 1/ν_eff(t) from central differences, with the forward/backward linearity flag. Code: section [5].
>
> **New data, in priority order.** All are knob copies except the one seed line in N1-C.
> 1. **N1-S. Spreading on a fine, wider grid**, to confirm J1 on measured curves:
>
>    | | ε_u | ε̄ | knobs |
>    |---|---|---|---|
>    | block_len 1 | 0.59525–0.60125 in steps of 0.00075 | 0.14286–0.1443 | `average_epsilon_c` = 0.14358, rate 0.00018, `i in -4:4` |
>    | block_len 6 | 0.458–0.4652 in steps of 0.0009 | 0.1099–0.1117 | 0.110784, rate 0.000216, `i in -4:4`, plus N3 |
>
>    - Both grids avoid every existing ε_u, so no chunk file is overwritten (checked). A grid starting at 0.595 would overwrite the existing 0.595 chunks.
>    - 200 × 500 runs as before; about 2× an existing spreading copy each.
>    - If the three zero-curvature points merge on measured curves, the conclusion changes. That makes it a real test.
> 2. **N1-C. Coupled decay runs for the response exponent to t = 10⁵.**
>    - Copy `_bvh_b6` (then `_bvh_b1`). Add `Random.seed!(SEED_BASE + init_cond + num_init_conds_offset)` at the top of the sample loop, so each sample has the same seed at every ε.
>    - Use a separate data root and N3.
>    - 5–9 ε values with Δε̄ ≈ 0.0001–0.0002 across ε_c, 5000 samples: about ⅓ of an existing decay copy. An optional first look at L = 10⁴ costs 1/100 of that.
>    - The local pilot (H §7.19 item 4) shows the coupling works with the production kernel and cuts the error of χ 17–300×.
>    - This decides whether ν_eff keeps growing past 2.25 (crossover) or levels off (fixed point).
> 3. **P1. The paper's own model (ε = 0.5Xⁿ, Fig. 4) through the spreading runs. It replaces W1** (a p = 0.8 binary copy).
>    - It is the model in Table I, so a referee will ask about it.
>    - Its disorder is weak: std/mean of ε is 0.57, against 0.47 for W1 and 1.58 for block_len 1. So it is the weak rung W1 was meant to supply, and it should start near DP like the spin chain. Run J1 on it.
>    - Needs `time_random_spreading` to accept the Xⁿ draw: a draw keyword, about 10 lines plus a test. That is the only new code.
>    - Grid: about 7–9 ε̄ values in 0.2530–0.2555 (ε̄_c ≈ 0.2537). The binary rungs' N and R points sit up to 0.6% above ε_c, so the grid reaches that far. Clusters are clean-sized, so it is cheap.

### 7.22 Runs N1-S, N1-C and P1 implemented (2026-10-05)

*Superseded the same day by §7.23: everything below was removed except the two fine spreading generators, which were redone as plain knob copies.*

You asked to implement the next Stavskaya runs, with their analysis, submit scripts and transfer commands. Everything was written and tested locally. Nothing was submitted or committed, and no existing data file was touched.

**What was added, and why.**

| piece | file | what it does |
|---|---|---|
| block-end output times (N3) | `stavskya_mc/utils/general.jl`: `make_block_end_times(t_max, block_len)` | log times rounded to multiples of block_len, so every output sits at the end of a disorder block (no ±2.5% sawtooth at block_len 6). For block_len 1 it is exactly `make_log_times` |
| any ε distribution in spreading | `stavskya_mc/utils/dynamics.jl`: `_spreading_run(record_times, draw)` and `_spreading_chunk`; new `time_random_n_spreading`, `spreading_chunk_n`, `_n_draw` | the paper's ε = aXⁿ model (n = a/ε̄ − 1, as `time_random_n_evolve_state`) in the single-seed experiment. `time_random_spreading` and `spreading_chunk` are now thin wrappers |
| naming | `generators/naming.jl`: `spreading_n_chunk_path`; `analysis/spreading_tools.py`: its mirror and `load_spreading_n` (the loaders now share `_load_chunks`) | `<root>/time_rand_n/spreading/tmax<T>/epsilonbar<e>/a<a>/blocklen<b>/…` |
| N1-S | `generators/get_upper_lower_binary_spreading_fine_{b1,b6}.jl` | knob copies of the bvh spreading generators: 9 fine ε values; block-end times |
| N1-C | `generators/get_upper_lower_binary_time_log_coupled_{b1,b6,clean}.jl` | copies of the bvh/clean decay generators with `Random.seed!(seed_base + k + offset)` per sample, fine grids, 5000 samples, block-end times, own data root `data/time_log_coupled` |
| P1 | `generators/get_time_random_n_spreading.jl` | spreading of the paper's model, 9 values of ε̄ in 0.2532–0.2552, model dir `time_rand_n` |
| submit | `submit/submit_upper_lower_spreading_fine_{b1,b6}.sh`, `submit_upper_lower_time_log_coupled_{b1,b6,clean}.sh`, `submit_time_random_n_spreading.sh` | copies of the bvh scripts; only the generator name and job name differ |
| transfer | `block_disorder/ssh_transfer.txt`, section "2026-10-05" | code to the cluster (git or rsync), sbatch, completeness checks, copy back (spreading → repo, coupled → external drive) |
| analysis | `analysis/crossover_tools.py`, `analyze_spreading_fine.ipynb`, `analyze_time_log_coupled.ipynb` | F1 (exponent drift against ε̄ with zero crossings), F2, F3, F4 (response exponent: coupled decay, plus a late-time spreading check), F5 (ladder). The notebooks hold no outputs; run them once the data are in |
| tests | `tests/test_runs_2026_10.jl`, `tests/test_crossover_tools.py`, additions to `test_spreading_tools.py`, `check_local_test_output.py` and `run_all_tests.sh` | block-end grid; coupling (samples ordered in ε, and reproducible); aXⁿ draw mean and spreading vs a full-lattice simulation; naming literals; the analysis functions on synthetic data with known answers; the local output of all six generators, including the ordering of the coupled samples |

**Checks.**
- **Refactor is bit-identical.** The refactored `time_random_spreading`/`spreading_chunk` give the same runs as the 2026-09-29 versions for fixed seeds. Checked in 4 settings × 200 seeds, plus chunks, against a saved copy of the old file.
- **Rounding ties (a bug avoided).** With the grids first proposed (§7.21), 3 of the 46 new ε_l = ε_u/20 values were exact decimal ties at the 7th digit, and Julia's `round(x, digits=6)` and Python's `round(x, 6)` broke them differently (e.g. 0.59875/20 → 0.029938 vs 0.029937). The notebooks would not have found those files.
  - The final grids use ε_u that are multiples of 0.00002: block_len 1 N1-S ε_u = 0.5954–0.6018 in steps of 0.0008; coupled grids on the same values as N1-S; clean coupled ε = 0.29442–0.29458 in steps of 0.00004.
  - All 46 names agree between the two languages, there are no ties, and no existing spreading ε_u is reused.
- **Tests.** `bash stavskya_mc/block_disorder/tests/run_all_tests.sh` (Julia 1.12.6, 2026-10-05): **OVERALL: PASS**.
  - test_dynamics, test_spreading and the new test_runs_2026_10 (89/89) pass.
  - All 15 generators pass their local smoke runs, including the 6 new ones.
  - `check_local_test_output.py`: 296/296 expected files, none unexpected. This includes the check that the coupled samples are ordered in ε at every time.
  - pytest is not installed for this Python, so the suite skips the Python tests; they were run separately, 20/20 pass (old and new). The log is `stavskya_mc/block_disorder/tests/last_test_run.log`.
- **Dry runs** on the existing data (executed copies in the session scratchpad; repo notebooks left without outputs):
  - `analyze_spreading_fine.ipynb` with `GRID = "coarse"`, 7 s, no errors. It reproduces the zero crossings of §7.21 (clean 0.29450/0.29451/0.29452; block_len 1 0.14306/0.14333/0.14386; block_len 6 0.11044/0.11058/0.11107), and each disordered zero is bracketed by the same two grid values 0.0012 apart. That is what N1-S fixes.
  - `analyze_time_log_coupled.ipynb` with `DRY_RUN_UNCOUPLED = True` (1000 of the existing independent samples per value), 41 s, no errors. Its coupling fraction is 0.56–0.64, and 1/ν_eff has errors of ±0.1–0.5 for the disordered rungs. That is the problem the coupled runs remove; on N1-C the coupling fraction must be 1.0.

**Data size and location.**
- **Spreading** (N1-S, P1): 27 × 200 chunk files of ~8 KB, ~45 MB. Repo `stavskya_mc/data/spreading` (git-ignored), like the other spreading runs.
- **Coupled decay** (N1-C): 95,000 sample files of ~1.1 KB (~0.4 GB on disk). External drive, `/Volumes/ExternalData/stavskya_mc/data/time_log_coupled`, like all decay data.

**Cost on the cluster** (same 8 nodes / 500 tasks / 24 h requests as before):
- each N1-S copy is about 2× an earlier spreading copy (1–2 h each), and P1 is similar;
- each N1-C copy is about ¼ of an earlier p = 0.2 decay copy, with clean less.

### 7.23 Simplified to one knob-copy run (2026-10-05)

You said §7.22 was far too much new code (most of it needed only knob changes), that the coupled runs felt like cheating and made things too complicated, and restated the goal:
- confirm that Stavskaya is consistent with BVH (not being blind to it), with reasonable effective-exponent estimates;
- get effective exponents for the spin chain in the crossover.

**Removed** (`git checkout` of the eight shared files I had edited, which had no prior local changes, and deletion of the files I had added):
- the coupled decay runs N1-C (three generators, submit scripts, notebook);
- the paper's-model spreading run P1, with its kernel refactor, naming and loader;
- the block-end output times (N3) and the `make_block_end_times` helper. They are not needed for spreading: the decade end points that the joint test uses already share one block phase;
- `crossover_tools.py`, the figure notebooks, `test_runs_2026_10.jl`, `test_crossover_tools.py`, and the additions to `test_spreading_tools.py`, `run_all_tests.sh` and the transfer file.

`dynamics.jl`, `general.jl`, `naming.jl` and `spreading_tools.py` are back to their committed versions.

**Kept, as knob copies:**

| file | change from the copied file |
|---|---|
| `generators/get_upper_lower_binary_spreading_fine_{b1,b6}.jl` | `average_epsilon_c`, `average_epsilon_rate`, `i in -4:4`, and a header note |
| `submit/submit_upper_lower_spreading_fine_{b1,b6}.sh` | `NAME`, `--job-name` |
| `analysis/analyze_spreading{4,5}.ipynb` (copies of 2, 3; outputs cleared) | `AVG_EPS_C`, `AVG_EPS_RATE`, `EPS_STEPS`, plus one appended joint-test cell (~20 lines) |
| `tests/check_local_test_output.py` | two lines listing the copies' local-test outputs |
| `block_disorder/ssh_transfer.txt` | a short section: rsync, sbatch, completeness check |

The grids are the tie-free ones of §7.22: every ε_u is a multiple of 0.00002, so Julia and Python name every file the same.

**Checks.**
- Both notebook copies, executed with their knobs set back to the existing coarse runs, run without errors. The joint-test cell reproduces the coarse zero crossings: block_len 1 gives 0.143056/0.143326/0.143864, block_len 6 gives 0.110438/0.110578/0.111075.
- `bash stavskya_mc/block_disorder/tests/run_all_tests.sh` on the simplified code: **OVERALL: PASS**.
  - The kernel tests and all 15 generator smoke runs pass, including the two fine copies.
  - The checker finds 216/216 expected files, none unexpected. pytest is not installed, so the Python tests were skipped.

**Also dropped from the plan:**
- the spin-chain coupled a-response (SC1) and the spin-chain spreading run (SC2);
- the F1–F5 figure plan.

The spin chain's effective exponents come from the v2 data with the existing estimators, compared with block_len 1 and 6 at equal times.

**Superseded text** (`REFEREE_CONFLICT_REVIEW.md`, the run and joint-plan sections as they stood before this section):

> ## Stavskaya: figures and the runs that fill them (2026-10-05; waiting for your go-ahead)
>
> *The verdicts on N1–N4/A1–A6 and the earlier version of this section are archived in H §7.21. In short: N2 and N4 are dropped, A1–A3 are optional or replaced, and A4–A6 stay as notebook text items.*
>
> **Figures for the paper** (each from measured curves once the runs below land):
>
> | fig | what it shows | data now | needs |
> |---|---|---|---|
> | F1. Joint test | drift of δ_eff, θ_eff and 1/z_eff (late decade − early decade) against ε̄, for clean, block_len 1, block_len 6 and P1. Clean: the three zeros coincide. Disordered: they separate | clean done; block_len 6 has one measured point inside the split; block_len 1 has none | N1-S, P1 |
> | F2. Local exponents against ln t, at the P_s point and at the R point of each rung | at the P_s point δ_eff is flat while 1/z_eff rises; at the R point the reverse. Clean: one ε, all flat | block_len 6 P_s point only | N1-S |
> | F3. BVH forms at strong disorder | 1/P_s, (N/t)^(−1/y_N) and (R/t)^(−1/y_R) straight in ln t at one measured ε̄, with fitted y and errors; clean overlay curving | 0.1104, close to the point | N1-S |
> | F4. 1/ν_eff(t) | clean flat at 0.58; the disordered rungs and P1 against the paper's 0.444 | model-dependent (item 5) | N1-C, with N1-S as a late-time check |
> | F5. Disorder ladder | early and late 1/z_eff and θ_eff against disorder strength (clean, P1, block_len 1, block_len 6); the spin chain added later | done except P1 | P1 |
> | Appendix | the decay/spreading duality; the distribution of ln A at the block_len 6 point | done; the block_len 6 sawtooth | N1-C per-sample files with N3 |
>
> **Runs: implemented 2026-10-05, tested locally, not submitted** (details H §7.22). Commands: `stavskya_mc/block_disorder/ssh_transfer.txt`, section "2026-10-05" (code to the cluster, `sbatch`, completeness checks, copying back).
>
> | run | generator (`block_disorder/generators/`) | ε grid | size | data go to |
> |---|---|---|---|---|
> | N1-S block_len 1 | `get_upper_lower_binary_spreading_fine_b1.jl` | ε̄ 0.142896–0.144432, 9 values, step 0.000192 (ε_u 0.5954–0.6018) | 9 × 200 × 500 runs, ~15 MB | repo `data/spreading` |
> | N1-S block_len 6 | `..._spreading_fine_b6.jl` (+ N3 block-end times) | ε̄ 0.10992–0.111648, 9 values, step 0.000216 (ε_u 0.458–0.4652) | same | repo |
> | N1-C block_len 1 | `get_upper_lower_binary_time_log_coupled_b1.jl` | ε̄ 0.142896–0.144048, 7 values (the lower 7 of N1-S) | 7 × 5000 samples, L = 10⁵ | external `time_log_coupled` |
> | N1-C block_len 6 | `..._time_log_coupled_b6.jl` (+ N3) | ε̄ 0.110136–0.111432, 7 values (the middle 7 of N1-S) | same | external |
> | N1-C clean | `..._time_log_coupled_clean.jl` | ε 0.29442–0.29458, 5 values, step 0.00004 | 5 × 5000 | external |
> | P1 | `get_time_random_n_spreading.jl` | ε̄ 0.2532–0.2552, 9 values, step 0.00025 (a = 0.5) | 9 × 200 × 500 runs, ~15 MB | repo |
>
> - **Grids changed from the earlier proposal.** The block_len 1 N1-S grid and the N1-C grids were moved so every ε_u is a multiple of 0.00002. Otherwise ε_l = ε_u/20 lands on a rounding tie that Julia and Python break differently, and the notebooks would miss those files (3 of 46 values; checked).
> - The new grids still avoid every existing spreading ε_u, so nothing is overwritten. The coupled runs sit at the same ε as N1-S.
> - **Coupling:** `Random.seed!(seed_base + k + num_init_conds_offset)` per sample. The checker verifies, on the local test output, that samples are ordered in ε.
> - **Reading F4:** report whatever 1/ν_eff does next to the clean control. A plateau at 0.40–0.45 would not signal a fixed point, since the BVH-like block_len 6 rung gives it too; a continued fall would show activated scaling starting.
> - **P1 grid:** refine if its split turns out smaller than one step.
>
> **Analysis files** (`block_disorder/analysis/`; dry-run on the existing data without errors):
> - `analyze_spreading_fine.ipynb`: F1 (drift against ε̄ with zero crossings), F2, F3, F5 and the late part of F4. `GRID = "coarse"` reproduces the existing results.
> - `analyze_time_log_coupled.ipynb`: F4. Its first table must show `coupling_fraction` = 1.0.
> - Both call `crossover_tools.py`.
>
> ---
>
> ## Bringing the two models together (proposed 2026-10-05; deferred until the v2 spin data arrive; details H §7.20)
>
> **What the spin chain's randomness is (exact; checked numerically, H §7.20 item 2).**
> - Each J-sign pair's step is the (+,+) step conjugated by a site-wise symmetry that fixes the target spiral and commutes with the push. For example, (−,−) is (+,+) under a π rotation about z on odd sites.
> - So the chain runs one deterministic map with random target-fixing symmetry kicks between steps. The kicks never change a, the target or S_diff; they only scramble the state.
> - **Consequence 1:** the disorder reaches the distance from criticality only indirectly. That makes it weak in BVH's sense, which fits δ_eff starting at DP.
> - **Consequence 2: don't add a block length to the J signs.** It means fewer kicks, i.e. longer stretches of the fixed Heisenberg chain, which is where the solitons live (Fig. S5). It moves the model toward the soliton chain, not toward stronger disorder.
> - The spin-chain counterpart of Stavskaya's ε(t) is a **time-random push strength a(t)**, with block_len on a(t) and the J signs still redrawn every step (Plans for later).
>
> **Observable map.** Use the same discriminating tests in both models.
>
> | Stavskaya | spin chain | status | discriminates? |
> |---|---|---|---|
> | decay A(t) | S_diff(t) | v2 running | no: one observable is degenerate with the critical point |
> | ε-response 1/ν_eff(t) (J2, N1-C) | a-response, ∂ ln S_diff/∂a | old data: measurable, ±0.06–0.12 | yes, once precise; tests the paper's ν_t directly |
> | spreading P_s, N, R (J1) | one chaotic site seeded into the controlled spiral | none | yes |
> | early-time 1/z_eff (ladder position) | 1/z_eff from R of the same spreading run | FSS z only | places the chain on the ladder |
> | FSS lifetime, crossing times | FSS collapse, t\* | have | no (H §7.19) |
> | — | Lyapunov jump | v2 running | not about the class; stays the paper's main claim |
>
> - The old idea of using the OTOC front as the spreading run is dropped. OTOC fronts measure damage spreading between two copies, a different process; the counterpart of Stavskaya's single-seed run is an active seed in the absorbing state.
>
> **First step once the v2 data are in: SC1, the spin chain's a-response exponent, with coupled samples.**
> - **Why this test:**
>   - ν_t is the number the paper's argument rests on: 2.24 > 2, "satisfies the bound without saturating". BVH predict that ν_eff grows without bound.
>   - The estimator is the same as Stavskaya's F4 (N1-C), so every model goes on one 1/ν_eff(t) plot with DP (0.577) and the paper (0.446). Item 5 above: a spin-chain value near 0.45 would match the disordered Stavskaya rungs, not single out a fixed point.
>   - It reuses the existing S_diff code. Only a per-sample seed and knobs change.
>   - The old data already resolve χ(t): 1/ν_eff = 0.42 ± 0.12, 0.52 ± 0.08 and 0.40 ± 0.06 at t = 10³, 3×10³ and 9×10³, close to block_len 1's values. Errors must shrink about 3× to tell a constant from a fall.
> - **SC1a. Local pilot first, about 1 h on 8 cores, writes nothing.**
>   - L = 256, t ≤ 500, a = 0.757/0.758/0.759, 200 samples, coupled vs uncoupled, as in the Stavskaya pilot.
>   - Why first: a change of a by 0.0003 decorrelates the microscopic state within about 25 steps, so only the shared initial state and kick sequence can cancel noise.
>   - The sample spread is 2–3× the finite-chain estimate at t = 30–300, which points to a shared global part. The old initial states could also cause that, so measure the gain.
> - **SC1b. Production.** A knob copy of `get_sdiff_data_severalL0.jl`:
>   - add `Random.seed!(SEED_BASE + init_cond + init_cond_name_offset)` before `make_random_state`; it seeds the initial state and the kick sequence;
>   - data root `s_diff_per_time_v2_coupled`;
>   - `time_prefact = 5` (t ≤ 10⁴), `step_size = 10`;
>   - a = 0.758 ± 0.0012 in steps of 0.0003 (9 values; linear in a up to t ≈ 10⁴ per the old data);
>   - 2000 samples, or the pilot's number;
>   - cost: about ⅓ of the running v2 S_diff copy.
>
>   If coupling doesn't help, run the same copy without the seed and with about 8000 samples.
> - **Free meanwhile, from the v2 data:**
>   - take early-time δ_eff from the L = 256 and 512 Lyapunov runs, which record every step. The v2 S_diff copy records every 200 steps, so its δ_eff starts at t = 2000;
>   - a first 1/ν_eff from their a values in 0.7550–0.7610.
>
> **SC2, after SC1: a spin-chain spreading run.** This is the J1 counterpart, and it gives the chain's early 1/z_eff.
> - Start from the spiral and set one site (or one 4-site cell) to a random direction.
> - Record at log-spaced times:
>   - survival: alive while max_i δS_i is above a small threshold; ε-close states never revive (Supp. S4);
>   - N = Σ_i δS_i;
>   - R² = Σ_i (i − i₀)² δS_i / Σ_i δS_i.
> - Needs one new function, a variant of `random_global_control_sdiff`.
> - In Stavskaya the joint test still separates on t = 10²–3×10³: the splits are 0.4–0.8% of ε_c, against 0.06% for clean DP. So the a grid must span about ±0.004: for example 9 values in 0.754–0.762, 10⁴ runs each.
>
> **What each piece supplies to the resubmission:**
> - **Keep:** the control transition and the Lyapunov jump (Figs. 1–3 from v2).
> - **Table I:** becomes effective exponents, each with the window it was measured in.
> - **Fig. 4 becomes the Stavskaya ladder:**
>   - J1 (clean passes, disordered rungs fail);
>   - 1/ν_eff(t) for every model, including the spin chain (SC1);
>   - the early 1/z_eff ladder, with the paper's model (P1) and the spin chain (SC2) placed on it.
> - **New paragraph:** the chain's randomness as symmetry kicks. It replaces "random signs to remove solitons" with a precise statement and explains why the chain is a weak-disorder model.

