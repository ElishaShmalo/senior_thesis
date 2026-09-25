# Referee conflict: active tasks and plans

*This is the working file: current status, open tasks and future plans only. Everything already done is in **`PROJECT_HISTORY.md`**: the original review, the code review, every fix, and all test results. References like "H §7.9" point to sections there. Last updated 2026-09-25.*

---

## Where things stand

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

## Active tasks (rough order)

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

2. **Stavskaya log-time production runs** (`stavskya_mc/block_disorder/submit/*.sh`).
   - Start with `block_len = 1`.
   - The critical control values in the generators were found for `block_len = 1` only. Rescan them with the `*_rho_per_ep_block` generators before using `block_len > 1`.

3. **Decide which late times to trust.**
   - Keep `time_prefact = 100` (H §7.10).
   - Run the `*_time_log_fss` generators at $L$ and $L/2$ (or $L/4$).
   - Use only the times where the sizes agree within errors. `analyze_time_log_fss.ipynb` has the light-cone/agreement check built in.

4. **Run the BVH tests on the new data.** The notebooks already contain all of these: `analyze_time_log.ipynb`, plus the BVH cells appended to the two `rho_per_time` notebooks and `analyze_sdiff_per_time3`.
   - $1/A$ against $\ln t$ is a straight line at infinite-noise criticality; the local slope $d(1/A)/d\ln t$ is flat.
   - Running $\delta_{\rm eff}$ drifts toward 0 (it is constant at a power-law point); $1/\delta_{\rm eff}$ against $\ln t$ is linear.
   - Crossing times off criticality: $\ln t_x\propto r^{-1/2}$ (BVH) vs $\nu_t\ln(1/r)$ (power law).
   - Width of $P(\ln A)$ over disorder realizations: it grows linearly in $\ln t$ under BVH and saturates at a finite-disorder fixed point. This can already be tried on the existing per-sample Stavskaya CSVs.

5. **Re-run the edited old Stavskaya notebooks** to refresh their saved outputs. These are the upper/lower and sliding-p `rho_per_time` notebooks, the `_z` notebooks and `rho_per_ep` (list in H §7.5, §7.11).

---

## Plans for later

- **Disorder-strength scans.**
  - Stavskaya: vary `block_len` and the contrast $\varepsilon_u/\varepsilon_l$.
  - If the "exponents" move with disorder strength, they are crossover values. If strong disorder gives clean log scaling sooner, that is BVH directly.
  - Spin chain: this needs a new knob (not implemented yet). Options are J-signs held fixed over blocks $\Delta t>1$, or a binary-random $a(t)$.
- **Spreading runs from a single seed** ($P_s(t)$, $N_s(t)$, $R(t)$): the cleanest test of $z=1$ with log corrections, free of finite-size effects. Not implemented yet.
- **Temporal Griffiths:** lifetime $\tau(L)$ on the active/chaotic side. BVH predict $\tau\sim L^{1/\kappa}$ with $\kappa$ varying continuously.
- **Rewrite the paper's claim** (H §5 item 7). Keep the Hamiltonian control transition and the discontinuous Lyapunov exponent, drop "new universality class", cite BVH and show the crossover analysis.
- **Optional, "Option A":** allowlist `julialang-s3.julialang.org`, `pkg.julialang.org` and `*.pkg.julialang.org` so I can run the Julia tests myself before handing code to you (H §7.7).

---

## Known issues deliberately left alone

- **`find_t_star_dist` (capture-time line matching).** It is kept for bookkeeping only. The Lyapunov-exponent fitting is the method of record, so the suspected missing parentheses there don't matter.
- **Pre-existing notebook failures.**
  - `s_diff_analysis_python` cell 29: KeyError, because a = 0.69 is not in `a_vals`.
  - `s_diff_analysis_python` cell 14: needs Python ≥ 3.12, which your Mac has.
- **log(S_diff) fit intercepts.** They are relative to `t_min`, hence the hand-tuned `+0.2` shift in the zoomed figures. Say if you want absolute intercepts.
- **DataCollapse and the other items marked DC** in H §4.

---

## How to run the tests

```bash
cd ~/research/senior_thesis
bash stavskya_mc/block_disorder/tests/run_all_tests.sh   # Stavskaya: kernel tests, 6 local generator runs, file-name check, pytest
bash heisen_spin_chain/tests/run_spin_tests.sh           # spin chain: refactor tests, 2 local runs, file/script check
```
Each writes `last_test_run.log` next to itself, ending in `OVERALL: PASS/FAIL`. Tell me you ran one and I'll read the log from your folder. Local test output goes to git-ignored folders (`_local_test_output/`).
