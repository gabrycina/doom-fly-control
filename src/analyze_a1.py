"""Per-scenario means with bootstrap 95% CIs; paired (same engine seed) differences vs the real-wiring arm."""
import json, sys, collections
import numpy as np
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results/a1"
ARMS = ["real", "shuf0", "noconn", "real_rewired_after_training", "real_rewired_s1", "real_rewired_s2", "random"]
SCEN = ["basic", "defend_the_center", "health_gathering", "deadly_corridor", "defend_the_line"]
data = collections.defaultdict(dict)
for arm in ARMS:
    p = R / f"{arm}.jsonl"
    if not p.exists(): continue
    for l in open(p):
        r = json.loads(l); data[(arm, r["scenario"])][r["episode"]] = r
rng = np.random.default_rng(0)
def boot(x, B=10000):
    x = np.asarray(x, float); idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
out = {}
for s in SCEN:
    for arm in ARMS:
        d = data.get((arm, s), {})
        if not d: continue
        eps = sorted(d); x = [d[e]["ret"] for e in eps]; k = [d[e]["kills"] for e in eps]
        rec = dict(n=len(x), mean=float(np.mean(x)), std=float(np.std(x)), ci95=boot(x), kills_mean=float(np.mean(k)),
                   steps_mean=float(np.mean([d[e]["steps"] for e in eps])))
        if arm != "real" and data.get(("real", s)):
            common = sorted(set(eps) & set(data[("real", s)]))
            diff = [d[e]["ret"] - data[("real", s)][e]["ret"] for e in common]
            rec["paired_diff_vs_real"] = dict(n=len(common), mean=float(np.mean(diff)), ci95=boot(diff))
            # permutation-free check: Welch t
            from scipy import stats
            rec["welch_p_vs_real"] = float(stats.ttest_ind(x, [data[("real", s)][e]["ret"] for e in data[("real", s)]], equal_var=False).pvalue)
            rec["mannwhitney_p_vs_real"] = float(stats.mannwhitneyu(x, [data[("real", s)][e]["ret"] for e in data[("real", s)]]).pvalue)
        out[f"{s}/{arm}"] = rec
        pd_ = rec.get("paired_diff_vs_real")
        print(f"{s:18s} {arm:28s} n={rec['n']:3d} mean={rec['mean']:8.2f} CI=[{rec['ci95'][0]:8.2f},{rec['ci95'][1]:8.2f}]"
              + (f"  diff={pd_['mean']:+7.2f} [{pd_['ci95'][0]:+.2f},{pd_['ci95'][1]:+.2f}] p_welch={rec['welch_p_vs_real']:.3f}" if pd_ else ""))
json.dump(out, open(R / "summary.json", "w"), indent=1)
