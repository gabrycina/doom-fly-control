"""Seed-matched evaluation of the doomfly-rl checkpoints (real / shuffled / no-connectome).

For every (scenario, episode index e): engine seed = 1000 + e, action-sampling generator seed = e.
Identical for every arm, so arms face the same episode starts and the same sampling RNG stream.
Writes one JSON line per episode.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, vizdoom as vzd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repos/doomfly-rl"))
from doomfly.doom.actions import SCENARIOS
from doomfly.doom.env import DoomEnv
from doomfly.surgery import load_flynet
from doomfly.train import legal_mask_table

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)            # name
ap.add_argument("--ckpt", required=True)
ap.add_argument("--connectome", default="hf/connectome_malecns49k.npz")
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
ap.add_argument("--greedy", action="store_true")
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", required=True)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--rewire-seed", type=int, default=-1, help="post-training lesion: permute presynaptic partners (+sign) of the trained graph")
a = ap.parse_args()
if a.threads: torch.set_num_threads(a.threads)
dev = torch.device(a.device)
RANDOM = a.ckpt == "random"
model, cfg = load_flynet("hf/malecns49k_noconn" if RANDOM else a.ckpt, a.connectome, "cpu")
model.eval()
model = model.cpu()
if model.connectome is not None:
    # Same W @ h as the repo's CSR spmm (verified max abs diff ~3e-6), written as index_add so it runs on MPS.
    C = model.connectome
    n = C.n
    dst = torch.repeat_interleave(torch.arange(n), (C.crow[1:] - C.crow[:-1]).long())
    src = C.col.long().clone(); sign = C.sign.clone()
    if a.rewire_seed >= 0:  # post-training lesion: presynaptic partner + its sign move, slot keeps prior*gain
        g = torch.Generator().manual_seed(a.rewire_seed)
        p = torch.randperm(len(src), generator=g)
        src, sign = src[p], sign[p]
        print("rewired", len(src), "edges; frac unchanged src", float((src == C.col.long()).float().mean()), flush=True)
    w = (sign * C.prior * torch.exp(C.log_gain)).detach()
    class IndexAddConnectome(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("dst", dst); self.register_buffer("src", src); self.register_buffer("w", w)
        def forward(self, x):
            x = x.float()
            return torch.zeros_like(x).index_add_(0, self.dst, self.w[:, None] * x[self.src])
    model.connectome = IndexAddConnectome()
model = model.to(dev)
legal = legal_mask_table(dev)
model.eval()
out = open(a.out, "a")
for name in a.scenarios:
    sc = SCENARIOS[name]
    env = DoomEnv(name, render=False, seed=1000)
    g2l = {int(g): i for i, g in enumerate(env.sc.actions)}
    for e in range(a.start, a.start + a.episodes):
        t0 = time.time()
        obs, _ = env.reset(seed=1000 + e)
        gen = torch.Generator().manual_seed(e)
        R, n, done = 0.0, 0, False
        acts = []
        with torch.no_grad():
            while not done and n < 2100:
                fr = torch.from_numpy(obs[None]).to(dev)
                logits = model(fr, torch.tensor([sc.id], device=dev), legal_mask=legal[sc.id][None])["policy"].float()[0].cpu()
                if RANDOM:  # uniform over the scenario's legal actions (reference floor)
                    legal_idx = torch.nonzero(legal[sc.id].cpu()).flatten()
                    act = int(legal_idx[torch.randint(len(legal_idx), (1,), generator=gen)])
                elif a.greedy:
                    act = int(logits.argmax())
                else:
                    act = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=gen))
                acts.append(act)
                obs, r, done, _, _ = env.step(g2l[act])
                R += r; n += 1
        kills = env.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        rec = dict(arm=a.arm, scenario=name, episode=e, ret=R, steps=n, kills=kills,
                   tics=env.game.get_episode_time(), greedy=a.greedy, wall=round(time.time() - t0, 2))
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(json.dumps(rec), flush=True)
    env.close()
