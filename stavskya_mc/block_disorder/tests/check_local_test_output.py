"""Check the files written by the generators in local test mode (STAV_LOCAL_TEST=1).

The point is to exercise the Julia -> Python hand-off for real: the Julia generators
write the files (names from naming.jl), and the Python loaders in
../analysis/time_log_tools.py must find every one of them, using the parameter values
the analysis notebooks rebuild. It checks:
  * every expected file exists at the path Python builds (so naming.jl and
    time_log_tools.py agree, including float formatting such as 0.4787 -> "0p4787");
  * there are no unexpected files (a naming mismatch would show up here);
  * the time column equals make_log_times(round(L*time_prefact), 20);
  * rho is in [0, 1] with no NaN, and once rho == 1 (absorbed) it stays 1;
  * the rho-per-epsilon files have 4 samples each.

The expected parameter sets below mirror the `if LOCAL_TEST ... end` blocks of the six
generators. If you change those blocks, change this file too.

Run:  python3 stavskya_mc/block_disorder/tests/check_local_test_output.py
(run_all_tests.sh runs it after the local generator runs.)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
import time_log_tools as tl  # noqa: E402

OUT = HERE.parent / "_local_test_output"
N_SAMPLES, PPD, BLOCKS = 4, 20, (1, 3)


def make_log_times(t_max, ppd=20):
    """Python mirror of make_log_times in stavskya_mc/utils/general.jl."""
    n = int(np.ceil(ppd * np.log10(t_max))) + 1
    ts = np.round(10 ** np.linspace(0, np.log10(t_max), max(n, 2))).astype(int)
    return np.array(list(dict.fromkeys([0, *ts, t_max])))


def upper_lower_sets(steps):
    # same arithmetic as the Julia generators (and analyze_time_log.ipynb)
    c, rate, p, div = 0.27033, 0.00005, 0.8, 20
    f = p + (1 - p) / div
    out = []
    for i in steps:
        u = round(c / f + i * (rate / f), 6)
        out.append((u, round(u / div, 6), p))
    return out


def sliding_sets(p_list):
    return [(0.43, round(0.43 / 20, 6), p) for p in p_list]


TIME_LOG = [  # (model_dir, L list, time_prefact, parameter sets)
    ("time_rand_window_binary", [64], 4.0, upper_lower_sets([-1, 0, 1])),           # get_upper_lower_binary_time_log.jl
    ("time_rand_window_binary", [32, 64], 4.0, upper_lower_sets([0])),              # ..._time_log_fss.jl
    ("time_rand_slidding_p", [64], 4.0, sliding_sets([round(0.479 + i * 0.0003, 6) for i in (-1, 0, 1)])),  # get_slidding_p_time_log.jl
    ("time_rand_slidding_p", [32, 64], 4.0, sliding_sets([0.47882])),               # ..._time_log_fss.jl
]
PER_EP = [  # (model_dir, L list, z, parameter sets)
    ("time_rand_window_binary", [16, 32], 1.45,
     [(u, round(u / 20, 6), 0.8) for u in (0.32, 0.321, 0.322)]),                   # get_upper_lower_binary_rho_per_ep_block.jl
    ("time_rand_slidding_p", [16, 32], 1.45, sliding_sets([0.465, 0.467, 0.469])),  # get_slidding_p_rho_per_ep_block.jl
]

failures, expected_files = [], set()


def check(cond, msg):
    if not cond:
        failures.append(msg)


# ---- log-time runs -----------------------------------------------------------------
for model, Ls, tp, sets in TIME_LOG:
    for L in Ls:
        grid = make_log_times(round(L * tp), PPD)
        for u, l, p in sets:
            for b in BLOCKS:
                for k in range(1, N_SAMPLES + 1):
                    expected_files.add(tl.time_log_sample_path(OUT / "time_log", model, L, u, l, p, b, tp, PPD, k).resolve())
                try:
                    run = tl.load_time_log_run(OUT / "time_log", model, L, u, l, p, b, tp, PPD, N_SAMPLES,
                                               min_samples=N_SAMPLES)
                except Exception as e:  # noqa: BLE001
                    check(False, f"{model} L={L} u={u} l={l} p={p} b={b}: {e}")
                    continue
                tag = f"{model} L={L} p={p} u={u} b={b}"
                check(np.array_equal(run.times, grid), f"{tag}: time grid != make_log_times({round(L*tp)})")
                check(np.all((run.rho >= 0) & (run.rho <= 1)), f"{tag}: rho outside [0,1]")
                for row in run.rho == 1:          # absorbed samples must stay absorbed
                    if row.any():
                        check(row[np.argmax(row):].all(), f"{tag}: rho left 1 after absorption")

# ---- rho-per-epsilon block runs -------------------------------------------------------
for model, Ls, z, sets in PER_EP:
    for L in Ls:
        for u, l, p in sets:
            for b in BLOCKS:
                f = tl.block_rho_per_ep_path(OUT / "block_rho_per_ep", model, L, u, l, p, b, z, N_SAMPLES)
                expected_files.add(f.resolve())
                if not f.exists():
                    check(False, f"missing {f}")
                    continue
                df = pd.read_csv(f)
                check(list(df.columns) == ["sample", "rho"], f"{f.name}: columns {list(df.columns)}")
                check(len(df) == N_SAMPLES and df["rho"].between(0, 1).all(), f"{f.name}: bad content")

# ---- nothing unexpected on disk --------------------------------------------------------
on_disk = {p.resolve() for p in OUT.rglob("*.csv")} if OUT.exists() else set()
check(OUT.exists(), f"{OUT} does not exist - run the generators with STAV_LOCAL_TEST=1 first")
for extra in sorted(on_disk - expected_files)[:10]:
    check(False, f"unexpected file (naming mismatch?): {extra.relative_to(OUT)}")

print(f"expected {len(expected_files)} files, found {len(on_disk & expected_files)}; "
      f"{len(on_disk - expected_files)} unexpected")
if failures:
    print(f"FAILED ({len(failures)} problems):")
    for m in failures[:30]:
        print("  -", m)
    sys.exit(1)
print("all local-test outputs OK")
