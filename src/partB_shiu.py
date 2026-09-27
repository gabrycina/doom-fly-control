"""Part B: Shiu et al. 2024 LIF brain model (unmodified model.py), real FlyWire 630 wiring vs
degree-preserving shuffles. Stimulate right-hemisphere sugar GRNs, read out MN9 (proboscis motor neuron).

Only the connectivity parquet is swapped. Shuffle: permute (presynaptic neuron, its sign) across all
edge rows, keep (postsynaptic neuron, synapse count) in place -> exact in-degree, exact out-degree,
exact per-neuron sign, exact per-neuron input synapse mass. Duplicate (pre, post) pairs are re-permuted
among the colliding rows only; leftovers are dropped and counted.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repos/Drosophila_brain_model"
sys.path.insert(0, str(REPO))

def shuffle(df, seed, rounds=60):
    g = np.random.default_rng(seed)
    pre_idx = df.Presynaptic_Index.to_numpy().copy(); pre_id = df.Presynaptic_ID.to_numpy().copy()
    exc = df.Excitatory.to_numpy().copy()
    post = df.Postsynaptic_Index.to_numpy().astype(np.int64)
    n = int(max(pre_idx.max(), post.max())) + 1
    p = g.permutation(len(pre_idx)); pre_idx, pre_id, exc = pre_idx[p], pre_id[p], exc[p]
    def dups(k):
        o = np.argsort(k)                       # quicksort; only need equal keys adjacent
        ks = k[o]; d = np.zeros(len(k), bool); d[o[1:][ks[1:] == ks[:-1]]] = True
        return d
    for _ in range(rounds):
        idx = np.flatnonzero(dups(post * n + pre_idx))
        print("  dup rows", len(idx), flush=True)
        if len(idx) == 0: break
        q = g.permutation(len(idx))
        pre_idx[idx], pre_id[idx], exc[idx] = pre_idx[idx][q], pre_id[idx][q], exc[idx][q]
    keep = np.flatnonzero(~dups(post * n + pre_idx))
    out = pd.DataFrame({
        "Presynaptic_ID": pre_id[keep], "Postsynaptic_ID": df.Postsynaptic_ID.to_numpy()[keep],
        "Presynaptic_Index": pre_idx[keep], "Postsynaptic_Index": df.Postsynaptic_Index.to_numpy()[keep],
        "Connectivity": df.Connectivity.to_numpy()[keep], "Excitatory": exc[keep]})
    out["Excitatory x Connectivity"] = out.Excitatory * out.Connectivity
    return out, len(df) - len(keep)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="*", default=["real", "shuf0", "shuf1", "shuf2"])
    ap.add_argument("--stim", default="sugar")
    ap.add_argument("--rate", type=float, default=100)
    ap.add_argument("--n_run", type=int, default=30)
    ap.add_argument("--n_proc", type=int, default=-1)
    ap.add_argument("--t_run_ms", type=float, default=1000)
    a = ap.parse_args()
    from model import run_exp, default_params
    from brian2 import Hz, ms
    params = dict(default_params); params["r_poi"] = a.rate * Hz; params["n_run"] = a.n_run
    params["t_run"] = a.t_run_ms * ms
    comp = REPO / "2023_03_23_completeness_630_final.csv"
    real = REPO / "2023_03_23_connectivity_630_final.parquet"
    work = ROOT / "partB"; (work / "spikes").mkdir(parents=True, exist_ok=True)
    stims = json.loads((ROOT / "src/partB_ids.json").read_text())
    meta = {}
    for c in a.conds:
        con = real
        if c.startswith("shuf"):
            con = work / f"connectivity_630_{c}.parquet"
            if not con.exists():
                df = pd.read_parquet(real)
                s, dropped = shuffle(df, int(c[4:]))
                s.to_parquet(con)
                meta[c] = {"rows_real": len(df), "rows_shuffled": len(s), "dropped_duplicates": int(dropped),
                           "frac_pairs_also_in_real": float(np.isin(s.Postsynaptic_Index.to_numpy().astype(np.int64) * 200000 + s.Presynaptic_Index.to_numpy(),
                                                                   df.Postsynaptic_Index.to_numpy().astype(np.int64) * 200000 + df.Presynaptic_Index.to_numpy()).mean())}
                print(c, meta[c], flush=True)
        run_exp(exp_name=f"{a.stim}_{int(a.rate)}Hz_{c}", neu_exc=stims[a.stim], path_res=str(work / "spikes"),
                path_comp=str(comp), path_con=str(con), params=params, n_proc=a.n_proc)
    if meta:
        (work / f"shuffle_meta_{'_'.join(meta)}.json").write_text(json.dumps(meta, indent=1))
