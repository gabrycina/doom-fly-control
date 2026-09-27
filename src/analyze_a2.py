"""A2 summary: nftechie/doomfly 166,700-neuron LIF fly in combat_survival, 60 s cap per round."""
import json, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
def boot(x, B=10000):
    x = np.asarray(x, float); m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
d = collections.defaultdict(list)
for f in ["real", "shuf0", "shuf1", "shuf2", "blind", "noconn", "spray", "spray_matched"]:
    for l in open(ROOT / f"results/a2/{f}.jsonl"):
        r = json.loads(l); d[r["arm"]].append(r)
d["shuffled (3 seeds pooled)"] = d["shuf0"] + d["shuf1"] + d["shuf2"]
out = {}
for arm, rs in d.items():
    s = [r["survival_s"] for r in rs]; k = [r["kills"] for r in rs]
    out[arm] = dict(n=len(rs), survival_mean=float(np.mean(s)), survival_ci=boot(s), kills_mean=float(np.mean(k)), kills_ci=boot(k),
                    attack_frac=float(np.sum([r["attack_tics"] for r in rs]) / np.sum([r["tics"] for r in rs])),
                    died_frac=float(np.mean([r["died"] for r in rs])))
    print(f"{arm:28s} n={len(rs):2d} survival {np.mean(s):5.1f}s {out[arm]['survival_ci']}  kills {np.mean(k):5.2f} {out[arm]['kills_ci']} attack {out[arm]['attack_frac']:.2f}")
json.dump(out, open(ROOT / "results/a2/summary.json", "w"), indent=1)
