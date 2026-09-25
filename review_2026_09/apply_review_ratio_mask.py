"""2026-09 (review 7.8.2): in the running-exponent ratio cells, skip times at which the mean
activity Q = 1 - rho is 0 (every sample absorbed), instead of taking log(x/0) or log(0/0).
Usage (run once): python3 review_2026_09/apply_review_ratio_mask.py <senior_thesis root>"""
import json, sys
from pathlib import Path
ROOT = Path(sys.argv[1])
def load(rel): return json.loads((ROOT / rel).read_text())
def save(rel, nb): (ROOT / rel).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
def rep(nb, i, old, new):
    s = "".join(nb["cells"][i]["source"]); assert s.count(old) == 1, (i, old[:60], s.count(old))
    s = s.replace(old, new); lines = s.split("\n")
    nb["cells"][i]["source"] = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
ALPHA_OLD = ("        data_to_plot = np.log(data_to_plot_numerator1/data_to_plot_numerator2) / "
             "np.log(data_to_plot_denomenator1/data_to_plot_denomenator2)\n")
ALPHA_NEW = ("        ok = (data_to_plot_numerator1 > 0) & (data_to_plot_numerator2 > 0)   # 2026-09: only times with Q(t/b), Q(t) > 0 (no log of 0)\n"
             "        l_time_vals = np.array(l_time_vals)[ok]\n"
             "        data_to_plot = np.log(data_to_plot_numerator1[ok]/data_to_plot_numerator2[ok]) / "
             "np.log(data_to_plot_denomenator1[ok]/data_to_plot_denomenator2[ok])\n")
DELTA_OLD = "        data_to_plot = log_b(data_to_plot_numerator[:len(data_to_plot_denomenator)]/data_to_plot_denomenator, 10)\n"
DELTA_NEW = ("        ok = (data_to_plot_numerator > 0) & (data_to_plot_denomenator > 0)   # 2026-09: only times with Q(t/b), Q(t) > 0 (no log of 0)\n"
             "        l_time_vals = np.array(l_time_vals)[ok]\n"
             "        data_to_plot = log_b(data_to_plot_numerator[ok]/data_to_plot_denomenator[ok], 10)\n")
f = "stavskya_mc/analyze_random_upper_lower_binary_rho_per_time.ipynb"; nb = load(f)
rep(nb, 9, ALPHA_OLD, ALPHA_NEW); save(f, nb); print("edited", f, "cell 9")
f = "stavskya_mc/analyze_random_slidding_p_rho_per_time2.ipynb"; nb = load(f)
rep(nb, 9, DELTA_OLD, DELTA_NEW); rep(nb, 14, ALPHA_OLD, ALPHA_NEW); save(f, nb); print("edited", f, "cells 9, 14")
