"""Part B summary: MN9 rate per trial (real vs shuffles), network-wide activity, bootstrap CIs."""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
ids = json.loads((ROOT / "src/partB_ids.json").read_text())
rng = np.random.default_rng(0)
def boot(x, B=10000):
    x = np.asarray(x, float); m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
out = {}
for p in sorted(glob.glob(str(ROOT / "partB/spikes/*.parquet"))):
    name = Path(p).stem; df = pd.read_parquet(p); T = 30
    rec = {"trials": int(T), "total_spikes_per_trial": float(len(df) / T),
           "active_neurons": int(df.flywire_id.nunique()),
           "active_non_stimulated": int(df.loc[~df.flywire_id.isin(ids["sugar"] + ids["bitter"])].flywire_id.nunique())}
    for side, i in zip(["MN9_L", "MN9_R"], ids["mn9"]):
        per = np.zeros(T); c = df.loc[df.flywire_id == i].groupby("trial").size()
        per[c.index.astype(int)] = c.values
        rec[side] = {"mean_hz": float(per.mean()), "ci95": boot(per), "trials_with_any_spike": int((per > 0).sum())}
    out[name] = rec
    print(name, json.dumps(rec))
(ROOT / "partB/summary.json").write_text(json.dumps(out, indent=1))
