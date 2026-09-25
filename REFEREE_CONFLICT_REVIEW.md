# Control transition with temporal randomness: the referee's objection, the follow-up numerics, and a code review

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
   - It probably does not change the universality class, but it contradicts the methods, changes the transient, and could shift $a_c$ slightly. All spin-chain data would need regenerating anyway if you change it.
2. **Off-by-one in the saved time column.** `lyapunov_exponents/get_sdiff_data_severalL0.jl:75,81-82`: `states_evolve_func(...)[1:step_size:n]` holds states at times $0,\,s,\,2s,\dots$, because index 1 is the initial state. The CSV labels them `t = 1:step_size:n`, i.e. $1,\,s+1,\dots$.
   - `analyze_sdiff_per_time3.ipynb` ignores that column and rebuilds $t=0,1,2,\dots$, so the figures are right.
   - Anyone using the CSV `t` column is off by one step. The script also stops before time $n$ because `1:step:n` excludes index $n+1$.
3. **Whole trajectory held in memory.** `utils/dynamics.jl:126-142` (`random_global_control_evolve`) allocates and stores all $n+1$ states and then subsamples. For $L=2000$ and $n=20L$ that is about 2 GB per sample before the `unflatten_state` copies.
   - It is not a correctness bug. But it is why samples are limited to 500 per job, and under temporal disorder the number of disorder realizations is what limits you.
   - **Fix:** compute $S_{\rm diff}$ on the fly.
4. **Minor points.**
   - The time step is passed as `J` (line 75), so changing `J` silently changes the step. The variable `tau` is unused.
   - `J_vec` is a global array mutated in place by `L_J_vec[1] *= ±1` (`dynamics.jl:132-133`). This is statistically fine (a uniform random sign times anything is a uniform random sign) but fragile.
   - `make_spiral_state` builds the spiral with SymPy. It is correct but slow.
5. **Duplicate job scripts.** `get_sdiff_data_severalL0–3.jl` differ only in `init_cond_name_offset` (0/500/1000/1500), which is fine. With SLURM `--requeue`, a preempted job silently regenerates its samples; that is harmless, but worth knowing.

### 4.2 Stavskaya (`stavskya_mc/`)
1. **Disorder correlation time is one step.** `utils/dynamics.jl:134-158` redraws $\varepsilon$ every step, so the disorder base interval is $\Delta t=1$. BVH needed $\Delta t=6$ (or 3) with a factor-20 contrast to see asymptotic behaviour quickly. With $\Delta t=1$ they saw a crossover at about $10^3$ even for a factor-10 contrast.
   - This is a design choice, not a bug. But it means your runs are exactly in the regime where BVH expect long crossovers.
   - **Suggestion:** add a `block_len` parameter.
2. **Time sampling is linear and starts late.** `time_step = 20000`, `150000` or `2000`, with the first sample at `t = time_step`.
   - In the $z$ runs (`..._time_data2.jl`, `analyze_*_z*.ipynb`), $L=1250$ has its first point at $t=150{,}000\approx120L$. The collapse sees only the finite-size tail, and there is no data in the $t\ll L^z$ regime.
   - **Fix:** use log-spaced output times from $t=1$.
3. **Run length far beyond the light cone.** Examples: `T_f = 100L` (upper/lower), `4500L` (z2), `10^4 L` (`slidding_p_time_data.jl`).
   - In Stavskaya, information travels one site per step. For $t<L$, the disorder-averaged $\rho(t)$ of the periodic chain is **exactly** that of the infinite chain.
   - For bulk decay you want $L\gtrsim t_{\max}$ and no more. Because the disorder is global, a larger $L$ does **not** average over disorder; only the number of samples does. BVH used $3\times10^4$–$10^6$ disorder realizations.
   - The paper's Fig. 4 ($L=20000$, $t$ up to about $2\times10^6\approx100L$) extends into the finite-size regime in its last decade.
4. **Type instability.** `new_state = [0.0 ...]` is Float64 while `state` is Int, and the pointer swap alternates types every step. The results are correct but slow. `rand(L)` and `[upper_ep, lower_ep][choice+1]` also allocate every step. A typed, allocation-free kernel (e.g. `BitVector` with `rand!` into a preallocated buffer) should be several times faster.
5. **Stale comment.** `get_time_random_uppper_lower_binary_time_data*.jl:22` says "$\bar\varepsilon=(\varepsilon_u+\varepsilon_l)/2=0.55\varepsilon_u$". The code correctly uses $0.81\varepsilon_u$.
6. **Scripts on disk no longer match the data the notebooks read.**
   - `get_time_random_slidding_p_time_data.jl` has $\varepsilon_u=0.42$, `lower_div=10`, $p_c=0.4928$, `time_prefact=10000`. `analyze_random_slidding_p_rho_per_time_z2.ipynb` reads $\varepsilon_u=0.43$, `/20`, $p=0.47882$, `timepref4500`.
   - The `_time_data2.jl` script produces 4000 samples at offset 2000, but `..._rho_per_time2.ipynb` loads 20000.
   - In `get_time_random_slidding_p.jl:33-34`, `p_vals` is assigned twice, so the notebook's coarse grid must come from an earlier run.
   - Nothing records which parameters produced which files. **Fix:** write a JSON sidecar (parameters, git hash, seed) next to every data directory.

### 4.3 Analysis notebooks
1. **`analyze_random_slidding_p_rho_per_time2.ipynb` was executed out of order.** Execution counts run cell 12→30, 13→31, 8→128, 10→157. Cell 2 defines 5 $p$ values (0.47842–0.47922, spacing 0.0002), but cell 8's saved output lists **7** values (0.47805–0.47985, spacing 0.0003).
   - The saved outputs, including $\bar\delta=0.997$ and the collapse plots, cannot be guaranteed to come from the code as saved.
   - `p_vals = set(sorted(...))` makes a set, which is unordered. So `zip(control_vals, ..., p_vals)` in cell 3 pairs mismatched values. It is only a display problem, since loading uses `control_vals`.
   - **Restart and run all** before trusting any number.
2. **`analyze_random_upper_lower_binary_rho_per_ep.ipynb`, cell 3:** `round(p*u+(1-p)*l)` without `ndigits` gives 0 for every entry. It only works because cell 4's plotting loop overwrites `epsilon_vals` before cell 8 uses it.
3. **Collapse fits (`DataCollapse`):**
   - lmfit's standard errors are meaningless here: for example $\sigma(p_c)=7\times10^{-9}$. The loss comes from sorting and nearest-neighbour interpolation, so it is piecewise and its numerical Jacobian is unreliable. Use the `bootstrapping` helper that is already in the cell but never called.
   - The 10001-point scan over initial $p_c$ records the *initial guess*, not the fitted $p_c$, and then hard-codes `best_pc`.
   - Parameters hitting their bounds (β = 0.596 against a bound of 0.6) should be treated as a failed fit.
   - FSS at a fixed $t=L^{1.45}$ bakes a power-law $z$ into the analysis.
4. **The $\ln\ln t$ extrapolation for $\bar\delta$** (all `*_rho_per_time*` notebooks and `analyze_sdiff_per_time3` cell 9) fits $\ln\rho/\ln\ln t$ against $X=1/\ln\ln t$ and reads $-\bar\delta$ from the intercept at $X=0$.
   - Over the data, $X$ spans only about 0.37–0.44 (Stavskaya) or 0.45–0.65 (spin chain). The extrapolation is 5–10× the data range, so $\bar\delta$ is extremely sensitive to noise and to any additive constant.
   - BVH's form is $\rho^{-1}=A+B\ln t$. It is linear in $\ln t$, needs no extrapolation, and absorbs the non-universal time scale.
   - In the Stavskaya cells, `positive` used for `crit_key` is the mask left over from the last loop iteration.
5. **`analyze_sdiff_per_time3.ipynb`, cells 10–11** (stale; execution count `None`):
   - The ratio uses `data[T//b-1]/data[T-1]`, i.e. $S(T/b-1)/S(T-1)$ with off-by-one indices. At $T\to0$ it hits `data[-1]`, the last element.
   - The x-axis uses `time_vals`, which is not defined in those cells and has a different length from the y data.
   - Cell 16, which makes the figure, does it correctly. Delete 10–11.
6. **Collapses by eye.** The "± 0.15", "± 0.0003" etc. in the markdown of `analyze_sdiff_per_time3` are eyeball estimates from manual collapses (cells 12–16). There is no fit and no error propagation.
7. **Mislabelled axes and legends.**
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
