"""Apply the 2026-09 review-4.3 fixes to the existing analysis notebooks.
Usage: python3 apply_nb_fixes.py <senior_thesis root>
Every replacement asserts that the old text occurs the expected number of times, so running
it on an unexpected version of a notebook fails loudly instead of editing the wrong thing."""
import json, sys
from pathlib import Path
ROOT = Path(sys.argv[1])

def load(rel):
    return json.loads((ROOT / rel).read_text())

def save(rel, nb):
    (ROOT / rel).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")

def src(nb, i):
    return "".join(nb["cells"][i]["source"])

def set_src(nb, i, s):
    lines = s.split("\n")
    nb["cells"][i]["source"] = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

def rep(nb, i, old, new, count=1):
    s = src(nb, i)
    n = s.count(old)
    assert n == count, f"cell {i}: expected {count} x {old!r}, found {n}"
    set_src(nb, i, s.replace(old, new))

log = []

# ---------------- stavskya_mc/analyze_random_slidding_p_rho_per_time2.ipynb --------------
f = "stavskya_mc/analyze_random_slidding_p_rho_per_time2.ipynb"; nb = load(f)
# 4.3.1: set(sorted(...)) -> sorted(set(...))  (a set is unordered; zip(control_vals, ..., p_vals) mispaired)
rep(nb, 2, "p_vals = set(sorted([round(p_c + i * p_rate, 6) for i in range(-2, 3)]))",
           "p_vals = sorted(set([round(p_c + i * p_rate, 6) for i in range(-2, 3)]))")
rep(nb, 2, "# p_vals = set(sorted(", "# p_vals = sorted(set(", count=2)
# 4.3.4: `positive` for crit_key was the mask left over from the loop's last iteration
rep(nb, 13, "    crit_key = (L_val, control_c)\n",
            "    crit_key = (L_val, control_c)\n    positive = (time_vals[crit_key] > 0) & (1 - mean_rhos[crit_key] > 0)   # mask for crit_key itself\n")
# 4.3.7 labels
rep(nb, 12, r'ax.set_ylabel(r"$1/\log(1 - \rho(t))$")', r'ax.set_ylabel(r"$1/(1 - \rho(t))$")')
rep(nb, 14, r'plt.xlabel(r"$1/t$")', r'plt.xlabel(r"$\ln(1/t)$")')
rep(nb, 10, r'plt.xlabel(r"$t^{1/\nu_{\|}}(\varepsilon - \varepsilon_c)$")', r'plt.xlabel(r"$t^{1/\nu_{\|}}(p - p_c)$")')
rep(nb, 15, r'plt.xlabel(r"$\ln(t)^{1/\nu_{\perp}}(\varepsilon - \varepsilon_c)$")', r'plt.xlabel(r"$\ln(t)^{1/\nu_{\perp}}(p - p_c)$")')
save(f, nb); log.append(f)

# ---------------- stavskya_mc/analyze_random_upper_lower_binary_rho_per_time.ipynb -------
f = "stavskya_mc/analyze_random_upper_lower_binary_rho_per_time.ipynb"; nb = load(f)
rep(nb, 8, "    crit_key = (L_val, control_c)\n",
           "    crit_key = (L_val, control_c)\n    positive = (time_vals[crit_key] > 0) & (1 - mean_rhos[crit_key] > 0)   # mask for crit_key itself\n")
rep(nb, 7, r'ax.set_ylabel(r"$1/\log(1 - \rho(t))$")', r'ax.set_ylabel(r"$1/(1 - \rho(t))$")')
rep(nb, 9, 'label = fr"p = {control_val}",', 'label = fr"{control_label} = {control_val}",')
rep(nb, 9, r'plt.xlabel(r"$1/t$")', r'plt.xlabel(r"$\ln(1/t)$")')
rep(nb, 10, 'label = fr"$p = {control_val}$",', 'label = fr"{control_label} = {control_val}",')
rep(nb, 10, r'plt.xlabel(r"$\ln(t)^{1/\nu_{\perp}}(\varepsilon - \varepsilon_c)$")',
            r'plt.xlabel(r"$\ln(t)^{1/\nu_{\perp}}(\bar\varepsilon - \bar\varepsilon_c)$")')
save(f, nb); log.append(f)

# ---------------- stavskya_mc/analyze_random_upper_lower_binary_rho_per_ep.ipynb ---------
f = "stavskya_mc/analyze_random_upper_lower_binary_rho_per_ep.ipynb"; nb = load(f)
# 4.3.2: round(x) without ndigits -> 0 for every entry
rep(nb, 3, "epsilon_vals.append(round(p_val * upper_ep + (1-p_val)*lower_ep))",
           "epsilon_vals.append(round(p_val * upper_ep + (1-p_val)*lower_ep, ndigits=6))")
save(f, nb); log.append(f)

# ---------------- 4.3.7: log(0) once every sample is absorbed (_z notebooks) --------------
MASK_OLD = "order = times > 0"
MASK_NEW = "order = (times > 0) & (1 - mean_rhos[key] > 0)   # drop absorbed times: log(0)"
f = "stavskya_mc/analyze_random_upper_lower_binary_rho_per_time_z.ipynb"; nb = load(f)
for i in (6, 9):
    rep(nb, i, MASK_OLD, MASK_NEW)
save(f, nb); log.append(f)
f = "stavskya_mc/analyze_random_slidding_p_rho_per_time_z2.ipynb"; nb = load(f)
for i in (5, 6, 9):
    rep(nb, i, MASK_OLD, MASK_NEW)
save(f, nb); log.append(f)

# ---------------- python_code/analyze_sdiff_per_time3.ipynb (4.3.5) ----------------------
f = "python_code/analyze_sdiff_per_time3.ipynb"; nb = load(f)
assert src(nb, 10).startswith("b = 2\nalpha = 1.0"), "cell 10 is not the b=2 alpha cell"
assert src(nb, 11).startswith("b = 10\ndelta_guess = 0.1075"), "cell 11 is not the b=10 delta cell"
assert "spinchain_determining_ac_delta_nu_par.png" in src(nb, 16)
new10 = r'''# Running log-exponent  alpha_eff(T) = ln[S(T/b)/S(T)] / ln[ln T / ln(T/b)]
# (constant = alpha for S ~ (ln t)^-alpha; BVH predict alpha -> 1).
# Fixed 2026-09 (review 4.3.5): the old version compared S(T/b - 1) with S(T - 1) (off by one),
# hit data[-1] (the LAST element) at T = 0, and plotted against a `time_vals` left over
# from another cell. Here collected_sdiffs[L][a][k] is S_diff at t = k * time_step.
b = 2
alpha = 1.0
cmaps = {2000: plt.colormaps.get_cmap("Oranges").resampled(len(a_vals) + 5), 512: plt.colormaps.get_cmap("Oranges").resampled(len(a_vals) + 5), 1024: plt.colormaps.get_cmap("Reds").resampled(len(a_vals) + 5)}

plt.figure(figsize=(9,6))
for j, L in enumerate(L_vals):
    for i, a_val in enumerate(a_vals):
        c = cmaps[L](i+3)
        S = np.asarray(collected_sdiffs[L][a_val])
        k_small = np.arange(1, len(S))                  # index of T/b
        k_big = k_small * b                             # index of T
        keep = (k_big < len(S)) & (np.log(k_small * time_step) >= 1.0)   # ln(T/b) >= 1: denominator well away from 0
        k_small, k_big = k_small[keep], k_big[keep]
        T = k_big * time_step
        alpha_eff = np.log(S[k_small] / S[k_big]) / np.log(np.log(T) / np.log(T / b))
        plt.plot(
            np.log(1 / T),
            alpha_eff,
            label = fr"$a = {a_val}$",
            c=c
            )

plt.axhline(alpha, c = 'k', label=fr"$\alpha = {alpha}$", linestyle ="--")
plt.xlabel(r"$\ln(1/t)$")
plt.ylabel(r"$\alpha_{\rm eff}=\frac{\ln[S(t/b)/S(t)]}{\ln[\ln t/\ln(t/b)]}$")
plt.legend(fontsize=10)
plt.xlim((-9.5, -1))
plt.tight_layout()
file_path = f"figs/time_random/alpha{str(alpha).replace('.', 'p')}/spinchain_alpha_eff.png"
make_path_exist(file_path)
# plt.savefig(file_path)
plt.show()'''
set_src(nb, 10, new10)
nb["cells"][10]["outputs"] = []
nb["cells"][10]["execution_count"] = None
del nb["cells"][11]            # mislabelled duplicate of what cell 16 (now 15) does correctly
save(f, nb); log.append(f)

print("edited:", *log, sep="\n  ")
