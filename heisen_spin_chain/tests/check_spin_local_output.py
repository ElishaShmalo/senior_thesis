"""Check the output of the SPIN_LOCAL_TEST=1 runs of get_good_data_severalL.jl and
get_sdiff_data_severalL0.jl, using the same path templates the updated notebooks use.

Checks:
  * spin_dists_per_time_v2 files (read by analysis_lyapunov_fixed, s_diff_analysis_python,
    s_diff_analysis_logsdiff_plot, capture_time_distrabution) exist under the notebook naming
    N{N}_a{a}_IC1_L{L}_z{z}_timestep{k}_sample{i}.csv, with columns t, lambda, delta_s and
    t = k, 2k, ... <= round(L^1.7);
  * s_diff_per_time_v2 files (read by analyze_sdiff_per_time3) exist, with columns t, s_diff
    and t = 0, step, 2*step, ..., n (the true times);
  * no unexpected files;
  * the other generator scripts (get_good_data_severalL2-6, get_sdiff_data_severalL1-3) differ
    from the two that were run only in their settings lines, so testing two covers all ten.
The expected parameters mirror the `if LOCAL_TEST ... end` blocks of those scripts.
"""
import difflib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GEN = HERE.parent / "lyapunov_exponents"
OUT = HERE / "_local_test_output" / "data"
N_VAL, A_VALS, UNIT_CELLS, N_IC = 4, [0.7, 0.76], [2, 4], 2
problems, expected = [], set()

# ---- get_good_data_severalL.jl -> spin_dists_per_time_v2 (notebook path template) ----------
z_val_name, record_every = "1p7", 1
for nuc in UNIT_CELLS:
    L = nuc * N_VAL
    n = int(round(L ** 1.7))
    for a in A_VALS:
        a_name = str(a).replace(".", "p")
        for ic in range(1, N_IC + 1):
            f = (OUT / f"spin_dists_per_time_v2/N{N_VAL}/a{a_name}/IC1/L{L}/"
                 f"N{N_VAL}_a{a_name}_IC1_L{L}_z{z_val_name}_timestep{record_every}_sample{ic}.csv")
            expected.add(f.resolve())
            if not f.exists():
                problems.append(f"missing {f}"); continue
            df = pd.read_csv(f)
            if list(df.columns) != ["t", "lambda", "delta_s"]:
                problems.append(f"{f.name}: columns {list(df.columns)}")
            if not np.array_equal(df["t"].to_numpy(), np.arange(record_every, n + 1, record_every)):
                problems.append(f"{f.name}: t column is not {record_every}, {2*record_every}, ..., <= {n}")
            if not (np.isfinite(df[["lambda", "delta_s"]].to_numpy()).all() and (df["delta_s"] >= 0).all()):
                problems.append(f"{f.name}: non-finite lambda/delta_s or negative delta_s")

# ---- get_sdiff_data_severalL0.jl -> s_diff_per_time_v2 (analyze_sdiff_per_time3 template) ---
time_prefact, step = 2, 3
for nuc in UNIT_CELLS:
    L = nuc * N_VAL
    n = L * time_prefact
    for a in A_VALS:
        a_name = f"{a}".replace(".", "p")
        for ic in range(1, N_IC + 1):
            f = (OUT / f"s_diff_per_time_v2/N{N_VAL}/a{a_name}/IC1/L{L}/"
                 f"N{N_VAL}_a{a_name}_IC1_L{L}_timepref{time_prefact}_timestep{step}_sample{ic}.csv")
            expected.add(f.resolve())
            if not f.exists():
                problems.append(f"missing {f}"); continue
            df = pd.read_csv(f)
            if list(df.columns) != ["t", "s_diff"]:
                problems.append(f"{f.name}: columns {list(df.columns)}")
            if not np.array_equal(df["t"].to_numpy(), np.arange(0, n + 1, step)):
                problems.append(f"{f.name}: t column is not 0, {step}, ..., <= {n}")

on_disk = {p.resolve() for p in OUT.rglob("*.csv")} if OUT.exists() else set()
for extra in sorted(on_disk - expected)[:10]:
    problems.append(f"unexpected file (naming mismatch?): {extra}")

# ---- the untested generator scripts differ only in settings lines ---------------------------
allowed = ("num_unit_cells_vals", "num_initial_conds", "init_cond_name_offset", "a_vals")
def changed_lines(a, b):
    return [l for l in difflib.unified_diff(a.read_text().splitlines(), b.read_text().splitlines(), lineterm="", n=0)
            if l[:1] in "+-" and not l.startswith(("+++", "---"))]
for ref, others in [("get_good_data_severalL.jl", [f"get_good_data_severalL{i}.jl" for i in range(2, 7)]),
                    ("get_sdiff_data_severalL0.jl", [f"get_sdiff_data_severalL{i}.jl" for i in range(1, 4)])]:
    for o in others:
        for l in changed_lines(GEN / ref, GEN / o):
            body = l[1:].strip()
            if body and not body.startswith("#") and not any(k in body for k in allowed) \
                    and not body.startswith(("0.", "0.7", "]")):
                problems.append(f"{o} differs from {ref} in a non-settings line: {l}")

print(f"expected {len(expected)} files, found {len(on_disk & expected)}, unexpected {len(on_disk - expected)}")
if problems:
    print(f"FAILED ({len(problems)} problems):")
    for p in problems[:30]:
        print("  -", p)
    sys.exit(1)
print("spin local-test outputs and generator scripts OK")
