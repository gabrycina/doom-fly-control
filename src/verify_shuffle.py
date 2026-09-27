"""Check what the shuffled control actually is: degrees, signs, overlap with the real wiring."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repos/doomfly-rl"))
from doomfly.model.flynet import load_connectome, shuffle_connectome
from safetensors import safe_open
c = load_connectome("hf/connectome_malecns49k.npz")
n = int(c["n"]); print({k: v.shape for k, v in c.items()})
res = {"n": n, "edges": int(len(c["src"])), "synapses": int(c["syn_count"].sum())}
real = set((c["dst"].astype(np.int64) * n + c["src"]).tolist())
for seed in [0, 1, 2]:
    s = shuffle_connectome(c, seed)
    indeg_ok = np.array_equal(np.bincount(s["dst"], minlength=n), np.bincount(c["dst"], minlength=n))
    outdeg_ok = np.array_equal(np.bincount(s["src"], minlength=n), np.bincount(c["src"], minlength=n))
    # sign travels with the presynaptic neuron?
    sgn_real = {}
    for src_, sg in zip(c["src"][:2_000_000], c["sign"][:2_000_000]): sgn_real.setdefault(int(src_), set()).add(int(sg))
    mixed = sum(len(v) > 1 for v in sgn_real.values())
    keys = s["dst"].astype(np.int64) * n + s["src"]
    overlap = np.isin(keys, np.fromiter(real, np.int64)).mean()
    res[f"seed{seed}"] = dict(edges=int(len(keys)), dropped=int(s["shuffled"][1]), indeg_exact=bool(indeg_ok),
                              outdeg_exact=bool(outdeg_ok), frac_edges_also_in_real=float(overlap))
    print(seed, res[f"seed{seed}"], "neurons with mixed-sign outputs in real (first 2M edges):", mixed, flush=True)
# edge counts stored in checkpoints
for arm in ["malecns49k_v2_final", "malecns49k_shuffled_s0"]:
    with safe_open(f"hf/{arm}/model.safetensors", "pt") as f:
        res[arm + "_log_gain_len"] = f.get_slice("connectome.log_gain").get_shape()[0]
        col = f.get_tensor("connectome.col"); crow = f.get_tensor("connectome.crow")
        dst = np.repeat(np.arange(n), np.diff(crow.numpy()))
        keys = dst.astype(np.int64) * n + col.numpy()
        res[arm + "_frac_edges_in_real"] = float(np.isin(keys, np.fromiter(real, np.int64)).mean())
print(json.dumps(res, indent=1))
json.dump(res, open("results/shuffle_check.json", "w"), indent=1)
