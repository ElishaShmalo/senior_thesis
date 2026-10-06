# Referee conflict: active tasks and plans

*Working file: current status, open tasks and plans only. Everything done so far is in **`PROJECT_HISTORY.md`** ("H §…"). Archived verbatim in H:*
- *the task list as it stood until 2026-10-05, with all the run instructions: H §7.15;*
- *the first BVH-reproduction results: H §7.16–7.17;*
- *the status and next-steps proposal as they stood before the assessment: H §7.18;*
- *the spin-chain lessons as they stood before the joint plan: H §7.20;*
- *the Stavskaya status and next steps as they stood before the figure plan: H §7.21;*
- *the figure plan, the N1-S/N1-C/P1 runs and the spin-chain joint plan before the simplification: H §7.23.*

*Last updated 2026-10-05 (plan simplified to one knob-copy run, H §7.23; final takeaway, H §7.21).*

---

## Where things stand

- **The question.** Are the measured exponents (δ ≈ 0.11, z ≈ 1.35–1.45, ν_t ≈ 2.25) a new fixed point? Or are they effective exponents crossing over toward the infinite-noise point of Barghathi–Vojta–Hoyos (BVH)?
- **Stavskaya: final takeaway** (results H §7.16–7.17; checks H §7.19–7.21; script `stavskya_mc/block_disorder/analysis/_claude_scratch/review_checks_2026_10_05.py`). **Regime:** the critical region at t = 10²–10⁵ (ln t ≈ 4.6–11.5), a crossover, not yet ln t ≫ c. Each item says whether it is measured or still interpolated.
  1. **Clean = DP on every test** [measured]:
     - P_s, N, R and the decay are power laws at one ε̄_c = 0.29450;
     - the ε-response gives 1/ν = 0.53–0.62 at all times (DP: 0.577).
  2. **Disorder makes the exponents depend on disorder strength** [measured]. Early 1/z_eff (t = 50–400) is 0.638 (clean), 0.73–0.76 (block_len 1) and 0.82–0.84 (block_len 6), at every ε of each grid; θ_eff behaves the same way. So there is no single universality class.
  3. **No conventional critical point describes 10³–10⁵.** In each disordered rung, P_s, N and R become power laws at different ε̄ (P_s, N, R: block_len 1 0.14306, 0.14332, 0.14388; block_len 6 0.11043, 0.11061, 0.11108; clean 0.29450–0.29452).
     - **Measured at block_len 6, ε̄ = 0.1104:** between the decades 10³–10⁴ and 10⁴–10⁵, P_s's exponent drifts +0.004 ± 0.001, while N's and R's drift +0.034 ± 0.003 and +0.024 ± 0.001. Clean at ε* drifts ≤ 0.005 in all three.
     - **Interpolated at block_len 1:** all three points lie between the measured 0.1428 and 0.1440. Measured at 0.1440: R is a power law (1/z = 0.745, drift −0.007 ± 0.002) while P_s and N bend down strongly (drifts −0.11 and −0.16).
     - N1-S removes the interpolation.
  4. **At block_len 6, BVH's logarithmic forms fit all three observables at one ε̄** [partly interpolated]:
     - 1/P_s is nearly linear in ln t over ln t ≈ 3–11.5 (slope 0.11–0.125);
     - at the measured 0.1104, the straightest log exponents are y_N ≈ 4.3 and y_R ≈ 2.5;
     - interpolated to the BVH point, y_R ≈ 1.6–2.0 (BVH: 1.7).
  5. **ν_eff ≈ 2.1–2.5 in both disordered rungs at t = 10³–10⁵** [model-dependent]. 1/ν_eff falls from ≈ 0.55 at 3×10² to 0.39–0.48, and is not yet seen falling further (section [11]).
     - **This corrects the earlier "ν_eff passes 2.25 and is still moving".**
     - It means the paper's ν_t ≈ 2.25 is what this crossover gives in this window, including at BVH's own parameters. So ν_t ≈ 2.25 is not evidence of a new fixed point.
     - The estimate is a quadratic in ε across 0.0024, nonlinear at late times; quote ν as an effective value with its window.
  6. **Decay mean = spreading survival** (exact for block_len 1; within 2% from t = 10²). So no new decay runs are needed.
  7. **Not tested:** the paper's own model (ε = 0.5Xⁿ, Fig. 4). Its spreading run would need new kernel code, so it was dropped in the simplification (H §7.23); the paper can use the binary ladder instead.
- **Spin chain:** regenerating v2 data. The earlier cluster failures were the account running out of memory, now fixed. Run instructions are in H §7.15, item 1.
  - Its J-sign randomness is exactly a sequence of random symmetry kicks that fix the target (H §7.20 item 2), so it is weak temporal disorder.
  - The old data give an a-response 1/ν_eff ≈ 0.4–0.5 at t = 10³–10⁴, close to block_len 1's, with errors too large to decide (H §7.20 item 4).

---

## Active tasks

1. **Finish the spin-chain v2 runs**, then re-run the spin notebooks (instructions: H §7.15, item 1).
2. **Submit the two fine spreading runs** (below; tested locally). Nothing has been submitted or committed.
3. **Spin chain:** effective exponents from the v2 data with the Stavskaya estimators (below).

---

## Stavskaya: the remaining run (simplified 2026-10-05; details H §7.23)

**Goal (yours):** show that Stavskaya is consistent with BVH (we are not blind to it), give reasonable effective-exponent estimates, and get effective exponents for the spin chain in the crossover.

**One run: fine spreading grids.** Knob copies of the bvh spreading generators; only `average_epsilon_c`, `average_epsilon_rate` and `i in -4:4` differ.
- Why: on the coarse grids the three power-law points of the joint test (item 3) sit inside one grid interval, so their positions were interpolated. With 9 values, measured curves fall inside the split.
- If the three zeros merge on the fine grid, item 3 is wrong; that makes it a real test.

| generator (`block_disorder/generators/`) | `average_epsilon_c`, rate | ε̄ (9 values) | ε_u |
|---|---|---|---|
| `get_upper_lower_binary_spreading_fine_b1.jl` | 0.143664, 0.000192 | 0.142896–0.144432 | 0.5954–0.6018 |
| `get_upper_lower_binary_spreading_fine_b6.jl` | 0.110784, 0.000216 | 0.10992–0.111648 | 0.458–0.4652 |

- Every ε_u is a multiple of 0.00002, so ε_l = ε_u/20 is never a rounding tie that Julia and Python name differently. No existing chunk file is overwritten.
- Cost: about 2× an earlier spreading copy each. Data: ~15 MB each, to the repo's `data/spreading`.
- Commands: `stavskya_mc/block_disorder/ssh_transfer.txt`, last section.

**Analysis:** `analyze_spreading4.ipynb` (block_len 1) and `analyze_spreading5.ipynb` (block_len 6).
- They are copies of `analyze_spreading2/3` with three knobs changed (`AVG_EPS_C`, `AVG_EPS_RATE`, `EPS_STEPS`) and one new last cell, the joint test. That cell takes the drift of the log-log slope of P_s, N and R between [10³,10⁴] and [10⁴,10⁵], and the ε̄ where each crosses zero.
- **Exponents to quote:** the existing cells give the effective exponents at a chosen ε̄, each with its window and regime:
  - δ_eff, θ_eff and 1/z_eff, early (t ≈ 50–400) and late (10⁴–10⁵);
  - y_N and y_R where 1/P_s is straightest;
  - ν_eff from the crossing-time cell, stated as effective (the coarse-grid estimate was ν_eff ≈ 2.1–2.5, H §7.21).

---

## Spin chain (when the v2 data arrive; no new code planned)

- **Effective exponents:** use the same estimators and the same time windows as Stavskaya, then compare with block_len 1 and 6 at equal t:
  - δ_eff per decade, with early times from the L = 256/512 Lyapunov runs, which record every step (the v2 S_diff copy records every 200 steps);
  - z from finite-size scaling, as in the paper;
  - ν from the collapse.
- **Block length on the J signs is not a disorder-strength knob.** Each sign pair is the same Heisenberg step up to a symmetry that leaves the target fixed (H §7.20), so longer blocks mean less scrambling and the solitons come back. The closest Stavskaya counterpart is block_len 1.
- **Stronger disorder in the chain:** if ever wanted, a time-random push strength a(t) (Plans for later).

---

## Plans for later

- **Disorder-strength ladder:** vary `block_len` and the contrast ε_u/ε_l (a weaker rung, e.g. p = 0.8, only if the paper needs one).
- **Spin chain disorder-strength knob:** a binary time-random push strength a(t), held for block_len steps, with the J signs still redrawn every step. Not J-sign blocks: they reduce the kicks and bring back the solitons (H §7.20 item 2).
- **Temporal Griffiths: deprioritized.**
  - Temporal Griffiths phases appear for any relevant temporal disorder (Vazquez et al., PRL 106, 235702 (2011)), so they don't discriminate.
  - At ε_c the test reduces to N4.
- **Rewrite the paper's claim** (H §5 item 7):
  - keep the Hamiltonian control transition and the discontinuous Lyapunov exponent;
  - drop "new universality class";
  - cite BVH;
  - show the crossover analysis: the J1 table, ν_eff(t) and the 1/z ladder. What each piece supplies is listed under "Bringing the two models together".

---

## Housekeeping

- **Untracked files to decide on:**
  - `heisen_spin_chain/tests/cluster_launch_test.{sh,jl}`: the launch diagnostic, no longer needed;
  - `stavskya_mc/block_disorder/analysis/_claude_scratch/`, three files:
    - `summary.json`: aggregated decay means, read by the review checks;
    - `review_checks_2026_10_05.py`: the H §7.19 checks, plus sections [9] (H §7.20) and [10]–[11] (H §7.21), about 15 s;
    - `coupling_pilot_2026_10_05.jl`: the coupling pilot, about 40 s.

    Both scripts write nothing.
  - `heisen_spin_chain/_claude_scratch/spin_checks_2026_10_05.py`: the H §7.20 spin-chain checks (a-response from the old means, sample spread, J-sign symmetry), under 1 s; writes nothing.
- **New files of 2026-10-05 (untracked):** the two fine generators, their submit scripts, `analyze_spreading4/5.ipynb`; plus two lines in `tests/check_local_test_output.py` and a section in `block_disorder/ssh_transfer.txt`.
- **Your local edit:** `time_log_tools.load_time_log_run` now prints and skips unreadable sample files (try/except).
- **Known issues left alone:** `find_t_star_dist` (bookkeeping only); `s_diff_analysis_python` cells 29 and 14; log(S_diff) intercepts relative to `t_min`; DC items in H §4. Details in H §7.15.
- **Tests:**
  ```bash
  cd ~/research/senior_thesis
  bash stavskya_mc/block_disorder/tests/run_all_tests.sh
  bash heisen_spin_chain/tests/run_spin_tests.sh
  ```
