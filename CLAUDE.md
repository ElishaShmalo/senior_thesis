# Project context for Claude Code

Senior-thesis project (Elisha Shmalo): classical Heisenberg spin chain with a control push and temporal randomness, plus a Stavskaya cellular automaton as a toy model. The paper (arXiv:2606.09297) claimed a new "temporally random DP" universality class; a referee cited Barghathi–Vojta–Hoyos (BVH, arXiv:1603.08075), whose infinite-noise critical point contradicts it. The work now is to show whether the measured exponents are a crossover toward BVH.

## Read these first
- `REFEREE_CONFLICT_REVIEW.md`: **active** status, findings and proposed next steps (short; kept current).
- `PROJECT_HISTORY.md`: full record of everything done ("H §…" references point here). §7.14–7.17 cover the Stavskaya BVH-reproduction runs and their first results; §7.19 is the 2026-10-05 assessment (decay/spreading duality, ε–form degeneracy of single-observable tests, joint zero-curvature test, ε-response exponent). §7.20 brings in the spin chain: its J-sign randomness as target-fixing symmetry kicks, the observable map, the joint test in short windows, and the spin-chain a-response from the old data. §7.21 is the Stavskaya final takeaway: what the last runs show directly versus by interpolation, the corrected response exponent (ν_eff ≈ 2.1–2.5 in both disordered rungs), and the figure plan behind the runs N1-S, N1-C and P1. §7.22–7.23: that plan was implemented and then cut back to one run, the fine spreading grids as knob copies (`*_spreading_fine_b{1,6}.jl`, analysed in `analyze_spreading4/5.ipynb`).

## Layout
- `heisen_spin_chain/`: spin chain (Julia).
  - Generators: `lyapunov_exponents/get_good_data_severalL*.jl`, `get_sdiff_data_severalL*.jl`; output goes to `data/*_v2/`.
  - Tests: `heisen_spin_chain/tests/run_spin_tests.sh`.
- `stavskya_mc/`: Stavskaya model.
  - Kernel: `utils/dynamics.jl` (`time_random_p_record_rho`, `time_random_spreading`, `spreading_chunk`).
  - New pipeline: `stavskya_mc/block_disorder/`:
    - `generators/`: knob-style generators, including the BVH copies `*_bvh_b1`, `*_bvh_b6`, `*_clean`, `*_fss_bvh_b1` and `get_upper_lower_binary_spreading_*`;
    - `submit/`: Slurm scripts, run from `stavskya_mc/` as `sbatch block_disorder/submit/<script>.sh`;
    - `analysis/`: notebooks plus `time_log_tools.py` and `spreading_tools.py`;
    - `tests/run_all_tests.sh`.
- `python_code/`: spin-chain analysis notebooks.

## Data locations
- Cluster (Amarel, `es1074@amarel-new.hpc.rutgers.edu`):
  - Stavskaya work: `~/senior_thesis3/senior_thesis`;
  - spin chain: `~/senior_thesis` and `~/senior_thesis2/senior_thesis`.
  - Cluster Julia: `~/julia-1.11.6`.
- Local:
  - GB-sized decay data (`time_log/`) are on `/Volumes/ExternalData/stavskya_mc/data/time_log/`;
  - small data (spreading, block scans) are in the repo's git-ignored `stavskya_mc/data/`.
  - Transfer commands: `stavskya_mc/block_disorder/ssh_transfer.txt`.

## Conventions and preferences
- Don't commit or push unless asked. Use `git --no-optional-locks` for read-only git commands. A plain `git status` once left a stale `.git/index.lock`.
- New variants: copy an existing generator or submit script and change only the knobs; keep the existing style. Avoid building new machinery when knobs suffice.
- Don't run production-size simulations locally; those go to the cluster. Small pilots and tests are fine.
- Don't change analysis or generate new data on your own initiative. Proposals go into `REFEREE_CONFLICT_REVIEW.md`, and the user decides.
- Old content moves from `REFEREE_CONFLICT_REVIEW.md` into `PROJECT_HISTORY.md`; the review file stays short.
- When interpreting the crossover data, always state the regime a prediction holds in (active, critical or inactive side; "BVH only once ln t ≫ c").

## Housekeeping left from the Cowork session
- `stavskya_mc/block_disorder/analysis/_claude_scratch/summary.json`: aggregated decay means, mean ln A and its spread, per parameter set. Read by `review_checks_2026_10_05.py` (same folder, H §7.19); deleting it skips that script's decay rows.
- `heisen_spin_chain/_claude_scratch/spin_checks_2026_10_05.py`: the H §7.20 spin-chain checks; reads the old local means in `data/s_diff_per_time/`, writes nothing.
