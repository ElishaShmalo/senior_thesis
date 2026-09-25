"""Apply the 2026-09 spin-chain notebook changes (review 4.1.2-4.1.4 follow-up) and the BVH cells
(review 4.3.4). Usage (run once, from anywhere): python3 review_2026_09/apply_review_spin_notebooks.py <senior_thesis root>
Each replacement asserts that the old text occurs exactly as expected."""
import json, sys
from pathlib import Path
ROOT = Path(sys.argv[1])

def load(rel): return json.loads((ROOT / rel).read_text())
def save(rel, nb): (ROOT / rel).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
def src(nb, i): return "".join(nb["cells"][i]["source"])
def set_src(nb, i, s):
    lines = s.split("\n")
    nb["cells"][i]["source"] = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
def rep(nb, i, old, new, count=1):
    s = src(nb, i); n = s.count(old)
    assert n == count, f"cell {i}: expected {count} x {old!r}, found {n}"
    set_src(nb, i, s.replace(old, new))
def code_cell(s):
    c = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": []}
    lines = s.split("\n"); c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]; return c
def md_cell(s):
    c = {"cell_type": "markdown", "metadata": {}, "source": []}
    lines = s.split("\n"); c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]; return c

PARAMS_ADD = '''

# ---- 2026-09: regenerated data (fixed random initial state), see REFEREE_CONFLICT_REVIEW.md 7.9 ----
DATA_DIR     = "spin_dists_per_time_v2"   # written by heisen_spin_chain/lyapunov_exponents/get_good_data_severalL*.jl
RECORD_EVERY = 1                           # that script's record_every knob (_timestep<k> in the file names)

# Rows are at the times collected_times[L] (= RECORD_EVERY, 2*RECORD_EVERY, ...), filled by the loader.
def t_index(L, t):
    """Row index of the recorded time closest to t (for RECORD_EVERY = 1: row t - 1, as before)."""
    return int(np.argmin(np.abs(np.asarray(collected_times[L]) - t)))

def t_window(L, t_min, t_max):
    """Boolean mask of the rows with t_min <= t <= t_max (replaces slicing [t_min - 1 : t_max])."""
    ts = np.asarray(collected_times[L])
    return (ts >= t_min) & (ts <= t_max)'''

def sdiff_loader_edits(nb, i, usecols_old, usecols_new, extra_arrays=False):
    """Common edits of the raw-CSV loader cell in the S_diff notebooks."""
    rep(nb, i, 'f"{parent_data_path}/spin_dists_per_time/N{N_val}/a{a_name}/IC1/L{L}/"',
               'f"{parent_data_path}/{DATA_DIR}/N{N_val}/a{a_name}/IC1/L{L}/"', count=2)
    rep(nb, i, 'f"N{N_val}_a{a_name}_IC1_L{L}_z{z_val_name}_sample{ic}.csv"',
               'f"N{N_val}_a{a_name}_IC1_L{L}_z{z_val_name}_timestep{RECORD_EVERY}_sample{ic}.csv"')
    rep(nb, i, 'f"N{N_val}_a{a_name}_IC1_L{L}_z1p7_sample{ic}.csv"',
               'f"N{N_val}_a{a_name}_IC1_L{L}_z1p7_timestep{RECORD_EVERY}_sample{ic}.csv"')
    rep(nb, i, usecols_old, usecols_new, count=2)
    rep(nb, i, 'fname = f"N{N_val}_ar{a_range}_IC{num_initial_conds}_L{L}_z{z_fit_name}"',
               'fname = f"N{N_val}_ar{a_range}_IC{num_initial_conds}_L{L}_z{z_fit_name}_timestep{RECORD_EVERY}"')

# =========================== s_diff_analysis_python.ipynb ===========================
f = "python_code/s_diff_analysis_python.ipynb"; nb = load(f)
rep(nb, 4, 'parent_data_path = "/Volumes/ExternalData"  # <- adjust as needed',
           'parent_data_path = "/Volumes/ExternalData"  # <- adjust as needed' + PARAMS_ADD)
sdiff_loader_edits(nb, 6, 'usecols=["delta_s"]', 'usecols=["t", "delta_s"]')
rep(nb, 6, "collected_S_diff_SEMs = {}\n\nfor num_unit_cells",
           "collected_S_diff_SEMs = {}\ncollected_times       = {}   # collected_times[L] = recorded times (same for every a)\n\nfor num_unit_cells")
rep(nb, 6, "        all_ic_diffs = np.zeros((num_initial_conds, n))\n", "        all_ic_diffs = []\n")
rep(nb, 6, '            all_ic_diffs[ic - 1] = df["delta_s"].values[:n]\n',
           '            keep = df["t"].values <= n                  # rows are at t = RECORD_EVERY, 2*RECORD_EVERY, ...\n'
           '            all_ic_diffs.append(df["delta_s"].values[keep])\n'
           '            collected_times[L] = df["t"].values[keep]\n')
rep(nb, 6, "        collected_S_diffs[L][a_val]     = all_ic_diffs.mean(axis=0)\n",
           "        all_ic_diffs = np.array(all_ic_diffs)\n        collected_S_diffs[L][a_val]     = all_ic_diffs.mean(axis=0)\n")
rep(nb, 6, '    np.savez(out_dir + fname + "_sem.npz"',
           '    np.savez(out_dir + fname + "_times.npz", t=collected_times[L])\n    np.savez(out_dir + fname + "_sem.npz"')
rep(nb, 8, "collected_S_diff_SEMs = {}\n\nfor", "collected_S_diff_SEMs = {}\ncollected_times       = {}\n\nfor")
rep(nb, 8, 'f"N{N_val}_ar{a_range}_IC{num_initial_conds}_L{L}_z{z_fit_name}"\n',
           'f"N{N_val}_ar{a_range}_IC{num_initial_conds}_L{L}_z{z_fit_name}_timestep{RECORD_EVERY}"\n')
rep(nb, 8, '    sem_data = np.load(base + "_sem.npz")\n',
           '    sem_data = np.load(base + "_sem.npz")\n    collected_times[L] = np.load(base + "_times.npz")["t"]\n')
# downstream indexing
rep(nb, 10, "    y    = collected_S_diffs[L_plot][a_val][:n_plot]\n    yerr = collected_S_diff_SEMs[L_plot][a_val][:n_plot]\n    ax.errorbar(np.arange(n_plot), y,",
            "    m    = collected_times[L_plot] <= n_plot\n    y    = collected_S_diffs[L_plot][a_val][m]\n    yerr = collected_S_diff_SEMs[L_plot][a_val][m]\n    ax.errorbar(collected_times[L_plot][m], y,")
rep(nb, 12, "    t_idx = int(round(L ** z_val)) - 1   # 0-indexed\n", "    t_idx = t_index(L, round(L ** z_val))   # row of t = L^z\n")
rep(nb, 13, "    t_idx = int(round(L ** z_val)) - 1\n", "    t_idx = t_index(L, round(L ** z_val))\n")
rep(nb, 14, "    t_idx = int(round(L ** z_val)) - 1\n", "    t_idx = t_index(L, round(L ** z_val))\n")
rep(nb, 16, "    t_idx = int(round(L ** z_val)) - 1   # 0-indexed\n", "    t_idx = t_index(L, round(L ** z_val))   # row of t = L^z\n")
rep(nb, 16, "    t_idx = int(round(L ** z_val)) - 1\n", "    t_idx = t_index(L, round(L ** z_val))\n")
FIT_OLD = ("        log_s = np.log(collected_S_diffs[L][a_val][t_min - 1 : t_max])\n"
           "        xs    = np.arange(1, len(log_s) + 1, dtype=float)\n")
FIT_NEW = ("        w     = t_window(L, t_min, t_max)\n"
           "        log_s = np.log(collected_S_diffs[L][a_val][w])\n"
           "        xs    = (collected_times[L][w] - t_min + 1).astype(float)   # 1, 2, ... for RECORD_EVERY = 1 (as before)\n")
rep(nb, 24, FIT_OLD, FIT_NEW)
rep(nb, 26, "        log_s = np.log(collected_S_diffs[L][a_val][t_min - 1 : t_max])\n        xs    = np.arange(1, len(log_s) + 1)\n",
            "        w     = t_window(L, t_min, t_max)\n        log_s = np.log(collected_S_diffs[L][a_val][w])\n        xs    = collected_times[L][w] - t_min + 1\n")
rep(nb, 26, "        ax.plot(log_s,  color=c,", "        ax.plot(xs, log_s,  color=c,")
rep(nb, 29, "    y    = np.log(collected_S_diffs[L_plot][a][:350])\n    yerr = collected_log_S_diff_errs[L_plot][a][:350]\n    ax.errorbar(np.arange(350), y,",
            "    m    = collected_times[L_plot] <= 350\n    y    = np.log(collected_S_diffs[L_plot][a][m])\n    yerr = collected_log_S_diff_errs[L_plot][a][m]\n    ax.errorbar(collected_times[L_plot][m], y,")
rep(nb, 29, "all_log_s_diff_offsets[L][a] + all_log_s_diff_slopes[L][a] * ts",
            "all_log_s_diff_offsets[L_plot][a] + all_log_s_diff_slopes[L_plot][a] * ts")
save(f, nb); print("edited", f)

# =========================== s_diff_analysis_logsdiff_plot.ipynb ===========================
f = "python_code/s_diff_analysis_logsdiff_plot.ipynb"; nb = load(f)
rep(nb, 4, 'parent_data_path = "/Volumes/ExternalData"  # <- adjust as needed',
           'parent_data_path = "/Volumes/ExternalData"  # <- adjust as needed' + PARAMS_ADD)
sdiff_loader_edits(nb, 6, 'usecols=["delta_s"]', 'usecols=["t", "delta_s"]')
rep(nb, 6, "collected_S_diff_SEMs = {}\n\nfor num_unit_cells",
           "collected_S_diff_SEMs = {}\ncollected_times       = {}   # collected_times[L] = recorded times (same for every a)\n\nfor num_unit_cells")
rep(nb, 6, "        all_ic_diffs = np.zeros((num_initial_conds, n))\n", "        all_ic_diffs = []\n")
rep(nb, 6, '            all_ic_diffs[ic - 1] = df["delta_s"].values[:n]\n',
           '            keep = df["t"].values <= n                  # rows are at t = RECORD_EVERY, 2*RECORD_EVERY, ...\n'
           '            all_ic_diffs.append(df["delta_s"].values[keep])\n'
           '            collected_times[L] = df["t"].values[keep]\n')
rep(nb, 6, "        collected_S_diffs[L][a_val]     = all_ic_diffs.mean(axis=0)\n",
           "        all_ic_diffs = np.array(all_ic_diffs)\n        collected_S_diffs[L][a_val]     = all_ic_diffs.mean(axis=0)\n")
rep(nb, 6, '    np.savez(out_dir + fname + "_sem.npz"',
           '    np.savez(out_dir + fname + "_times.npz", t=collected_times[L])\n    np.savez(out_dir + fname + "_sem.npz"')
rep(nb, 8, "collected_S_diff_SEMs = {}\n\nfor", "collected_S_diff_SEMs = {}\ncollected_times       = {}\n\nfor")
rep(nb, 8, 'f"N{N_val}_ar{a_range}_IC{num_initial_conds}_L{L}_z{z_fit_name}"\n',
           'f"N{N_val}_ar{a_range}_IC{num_initial_conds}_L{L}_z{z_fit_name}_timestep{RECORD_EVERY}"\n')
rep(nb, 8, '    sem_data = np.load(base + "_sem.npz")\n',
           '    sem_data = np.load(base + "_sem.npz")\n    collected_times[L] = np.load(base + "_times.npz")["t"]\n')
rep(nb, 11, FIT_OLD, FIT_NEW)
rep(nb, 12, "    y    = np.log(collected_S_diffs[L_plot][a][:350])\n    yerr = collected_log_S_diff_errs[L_plot][a][:350]\n",
            "    m    = collected_times[L_plot] <= 350\n    y    = np.log(collected_S_diffs[L_plot][a][m])\n    yerr = collected_log_S_diff_errs[L_plot][a][m]\n")
rep(nb, 12, "ax.errorbar(np.arange(350), y,", "ax.errorbar(collected_times[L_plot][m], y,", count=2)
rep(nb, 12, "all_log_s_diff_offsets[L][a] + all_log_s_diff_slopes[L][a] * ts",
            "all_log_s_diff_offsets[L_plot][a] + all_log_s_diff_slopes[L_plot][a] * ts")
save(f, nb); print("edited", f)

# =========================== capture_time_distrabution.ipynb ===========================
f = "python_code/capture_time_distrabution.ipynb"; nb = load(f)
rep(nb, 4, 'parent_data_path = "/Volumes/ExternalData"  # <- adjust as needed',
           'parent_data_path = "/Volumes/ExternalData"  # <- adjust as needed' + PARAMS_ADD +
           '\n\n# time_steps (above) still thins the loaded rows further: every time_steps-th recorded row is kept.')
sdiff_loader_edits(nb, 5, 'usecols=["delta_s", "lambda"]', 'usecols=["t", "delta_s", "lambda"]')
rep(nb, 5, "collected_lambda_errs = {}\n\nfor num_unit_cells",
           "collected_lambda_errs = {}\ncollected_times = {}   # collected_times[L] = recorded (and thinned) times\n\nfor num_unit_cells")
rep(nb, 5, '            diffs_vals = df["delta_s"].values[0:n:time_steps]\n            lambdas_vals = df["lambda"].values[0:n:time_steps]\n',
           '            keep = df["t"].values <= n                  # rows are at t = RECORD_EVERY, 2*RECORD_EVERY, ...\n'
           '            diffs_vals = df["delta_s"].values[keep][::time_steps]\n'
           '            lambdas_vals = df["lambda"].values[keep][::time_steps]\n'
           '            collected_times[L] = df["t"].values[keep][::time_steps]\n')
rep(nb, 7, "    ax.errorbar([i * time_steps for i in range(len(y))], y,", "    ax.errorbar(collected_times[L_plot][:len(y)], y,")
rep(nb, 11, "ts = np.array([i*time_steps for i in range(len(ys))])", "ts = np.asarray(collected_times[L])[:len(ys)]")
rep(nb, 12, "        ts = np.array([i*time_steps for i in range(len(ys))])\n", "        ts = np.asarray(collected_times[L])[:len(ys)]\n", count=2)
rep(nb, 12, "        log_sample_data = np.log(np.array(sample_data))\n        ts = [i*time_steps for i in range(len(log_sample_data)) if i*time_steps < L**z_val]\n",
            "        log_sample_data = np.log(np.array(sample_data))\n"
            "        times_all = np.asarray(collected_times[L], dtype=float)[:len(log_sample_data)]\n"
            "        keep_t = times_all < L**z_val\n"
            "        ts = list(times_all[keep_t])\n"
            "        log_sample_data = log_sample_data[keep_t]\n")
rep(nb, 12, "            ax.plot(ts, np.log(collected_S_diffs[L][a_val][sample]), c=color, alpha=0.5)",
            "            ax.plot(ts, log_sample_data, c=color, alpha=0.5)")
rep(nb, 18, "        ts = np.array([i*time_steps for i in range(len(ys))])\n", "        ts = np.asarray(collected_times[L])[:len(ys)]\n", count=2)
rep(nb, 18, "        t_stars.append((np.argmax((np.abs((avged_sample_data[skip_init:] - np.log(a_val))/np.log(a_val)) <= threshold)) + skip_init)*time_steps)",
            "        t_stars.append(collected_times[L][np.argmax((np.abs((avged_sample_data[skip_init:] - np.log(a_val))/np.log(a_val)) <= threshold)) + skip_init])")
rep(nb, 23, FIT_OLD.replace("collected_S_diffs[", "collected_S_diffs_mean["), FIT_NEW.replace("collected_S_diffs[", "collected_S_diffs_mean["))
rep(nb, 25, "        log_s = np.log(collected_S_diffs_mean[L][a_val][t_min - 1 : t_max])\n        xs    = np.arange(1, len(log_s) + 1)\n",
            "        w     = t_window(L, t_min, t_max)\n        log_s = np.log(collected_S_diffs_mean[L][a_val][w])\n        xs    = collected_times[L][w] - t_min + 1\n")
rep(nb, 25, "        ax.plot(log_s,  color=c,", "        ax.plot(xs, log_s,  color=c,")
save(f, nb); print("edited", f)

# =========================== analysis_lyapunov_fixed.ipynb ===========================
f = "python_code/analysis_lyapunov_fixed.ipynb"; nb = load(f)
rep(nb, 4, 'local_parent_data_path = "../data"  # <- adjust as needed',
           'local_parent_data_path = "../data"  # <- adjust as needed\n\n'
           '# ---- 2026-09: regenerated data (fixed random initial state), see REFEREE_CONFLICT_REVIEW.md 7.9 ----\n'
           'DATA_DIR     = "spin_dists_per_time_v2"   # written by get_good_data_severalL*.jl\n'
           'RECORD_EVERY = 1                           # its record_every knob (_timestep<k>); lambda rows are block means\n'
           'SUMMARY_DIR  = "spin_chain_lambdas_v2"     # summary CSVs of this notebook (kept apart from the old ones)')
rep(nb, 7, 'data_folder = f"{parent_data_path}/spin_dists_per_time_new/N{N_val}/a{a_name}/IC1/L{L}/"',
           'data_folder = f"{parent_data_path}/{DATA_DIR}/N{N_val}/a{a_name}/IC1/L{L}/"')
rep(nb, 7, 'f"{parent_data_path}/spin_dists_per_time_new/N{N_val}/a{a_name}/IC1/L{L}/"',
           'f"{parent_data_path}/{DATA_DIR}/N{N_val}/a{a_name}/IC1/L{L}/"')
rep(nb, 7, 'f"N{N_val}_a{a_name}_IC1_L{L}_z{z_val_name}_sample{init_cond}.csv"',
           'f"N{N_val}_a{a_name}_IC1_L{L}_z{z_val_name}_timestep{RECORD_EVERY}_sample{init_cond}.csv"')
rep(nb, 7, 'df_sample = pd.read_csv(sample_filepath, usecols=["lambda"])\n                lambda_vec = df_sample["lambda"].values[num_skip:n]',
           'df_sample = pd.read_csv(sample_filepath, usecols=["t", "lambda"])\n'
           '                t_s = df_sample["t"].values          # rows at t = RECORD_EVERY, 2*RECORD_EVERY, ...\n'
           '                lambda_vec = df_sample["lambda"].values[(t_s > num_skip) & (t_s <= n)]   # = values[num_skip:n] for RECORD_EVERY = 1')
rep(nb, 7, 'f"{local_parent_data_path}/spin_chain_lambdas_new/N{N_val}', 'f"{local_parent_data_path}/{SUMMARY_DIR}/N{N_val}')
rep(nb, 10, 'f"{local_parent_data_path}/spin_chain_lambdas_new/N{N_val}', 'f"{local_parent_data_path}/{SUMMARY_DIR}/N{N_val}')
save(f, nb); print("edited", f)

# =========================== analyze_sdiff_per_time3.ipynb ===========================
f = "python_code/analyze_sdiff_per_time3.ipynb"; nb = load(f)
rep(nb, 2, '"/Volumes/ExternalData/s_diff_per_time/"', '"/Volumes/ExternalData/s_diff_per_time_v2/"', count=3)
rep(nb, 2, '"../data/s_diff_per_time/"', '"../data/s_diff_per_time_v2/"')
rep(nb, 4, "                df = pd.read_csv(sample_filepath_name)\n\n                current_sdiff.append(df.s_diff)\n",
           "                df = pd.read_csv(sample_filepath_name)\n"
           "                # 2026-09 files (get_sdiff_data_severalL*.jl): t = 0, time_step, ..., T_f are the true times.\n"
           "                # Keep the first T_f // time_step rows (t = 0 ... T_f - time_step) so every later cell\n"
           "                # sees the same array length and row i <-> t = i * time_step, as before.\n"
           "                n_keep = T_f // time_step\n"
           "                if not np.allclose(df.t.values[:n_keep], np.arange(n_keep) * time_step):\n"
           "                    raise ValueError(f\"unexpected t column in {sample_filepath_name}\")\n"
           "                current_sdiff.append(df.s_diff.values[:n_keep])\n")
save(f, nb); print("edited", f)

# =========================== 4.3.4: BVH fit cells ===========================
BVH_MD = """## BVH infinite-noise test: $1/A = a + B\\ln t$ (review 4.3.4, added 2026-09)
The infinite-noise form of Barghathi, Vojta and Hoyos is linear in $\\ln t$, so it needs no
extrapolation, and $a/B=\\ln t_0$ absorbs the microscopic time scale. The table compares it
with a power law fitted over the same window. The $\\chi^2_\\nu$ values ignore the
correlations between times, so compare the two columns with each other only. The plot
shows the local slope $d(1/A)/d\\ln t$ over a factor 2 in $t$: constant (= B) at the BVH
critical point, rising on the inactive side and falling toward 0 on the active side."""
BVH_CORE = '''rows = []
for key in bvh_keys:
    t = np.asarray(bvh_t[key], float); A = np.asarray(bvh_A[key], float); sA = np.asarray(bvh_sem[key], float)
    m = (t >= fit_tmin) & (t <= fit_tmax) & (A > 0) & (sA > 0)
    x = np.log(t[m])
    y, sy = 1 / A[m], sA[m] / A[m]**2                               # BVH: 1/A, error sigma_A / A^2
    (B, a), cov = np.polyfit(x, y, 1, w=1/sy, cov="unscaled")
    chi2_bvh = np.sum(((y - (a + B*x)) / sy)**2) / (m.sum() - 2)
    (slope, c), _ = np.polyfit(x, np.log(A[m]), 1, w=A[m]/sA[m], cov="unscaled")   # power law, same window
    chi2_pow = np.sum(((np.log(A[m]) - (c + slope*x)) / (sA[m]/A[m]))**2) / (m.sum() - 2)
    rows.append({"key": key, "B": B, "B_err": np.sqrt(cov[0, 0]), "ln t0 = a/B": a / B, "chi2r BVH": chi2_bvh,
                 "delta (power law)": -slope, "chi2r power": chi2_pow, "points": int(m.sum())})
print("chi2r ignores correlations between times: compare the two columns, do not read them absolutely")
display(pd.DataFrame(rows).round(4))

fig, ax = plt.subplots(figsize=(8, 5))                    # local slope over a factor 2 in t
for key in bvh_keys:
    t = np.asarray(bvh_t[key], float); A = np.asarray(bvh_A[key], float)
    ok = (t > 0) & (A > 0); t, A = t[ok], A[ok]; T = t[t / 2 >= t[0]]
    ax.plot(np.log(T), (1/A[np.searchsorted(t, T)] - np.interp(np.log(T/2), np.log(t), 1/A)) / np.log(2), label=f"{key[-1]}")
ax.set_xlabel(r"$\\ln t$"); ax.set_ylabel(r"$d(1/A)/d\\ln t$"); ax.legend(fontsize=8); plt.show()'''
BVH_STAV = '''# A = 1 - rho. Uses time_vals, mean_rhos, sem_rhos from the loading cell.
fit_tmin, fit_tmax = 2e4, 2e6          # a window inside the data (ideally t < L, review 4.2.3)
bvh_keys = [(L_val, c) for L_val in L_vals for c in control_vals]
bvh_t    = {k: time_vals[k] for k in bvh_keys}
bvh_A    = {k: 1 - np.asarray(mean_rhos[k]) for k in bvh_keys}
bvh_sem  = {k: sem_rhos[k] for k in bvh_keys}
''' + BVH_CORE
BVH_SPIN = '''# A = S_diff. Uses collected_sdiffs / collected_sdiff_stds from the loading cells;
# row i is t = i * time_step.
fit_tmin, fit_tmax = 50, 5000          # a window inside the data (and ideally before finite-size effects)
bvh_keys = [(L_val, a) for L_val in L_vals for a in a_vals]
bvh_t    = {k: np.arange(len(collected_sdiffs[k[0]][k[1]])) * time_step for k in bvh_keys}
bvh_A    = {k: np.asarray(collected_sdiffs[k[0]][k[1]]) for k in bvh_keys}
bvh_sem  = {k: np.asarray(collected_sdiff_stds[k[0]][k[1]]) / np.sqrt(num_initial_conds) for k in bvh_keys}
''' + BVH_CORE
for f, body in [("stavskya_mc/analyze_random_upper_lower_binary_rho_per_time.ipynb", BVH_STAV),
                ("stavskya_mc/analyze_random_slidding_p_rho_per_time2.ipynb", BVH_STAV),
                ("python_code/analyze_sdiff_per_time3.ipynb", BVH_SPIN)]:
    nb = load(f)
    assert not any("BVH infinite-noise test" in "".join(c["source"]) for c in nb["cells"]), f"{f}: BVH cell already present"
    nb["cells"] += [md_cell(BVH_MD), code_cell(body)]
    save(f, nb); print("appended BVH cells to", f)
