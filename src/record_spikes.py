"""Re-run one nftechie/doomfly round (same code path as run_a2.py) and save, per game tic, the frame,
kill count, applied controls, and the indices+counts of every neuron that spiked (first --spike-secs of game time)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1] / "repos/doomfly"
sys.path.insert(0, str(REPO))
from doom.native import NativeBrain
from doom.engine import NeuralControls
from doom.game import Game, retinal_samples
ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="real"); ap.add_argument("--episode", type=int, default=0)
ap.add_argument("--spike-secs", type=float, default=25); ap.add_argument("--max-seconds", type=float, default=60)
ap.add_argument("--out", required=True)
a = ap.parse_args()
GRAPH = REPO / "outputs/doom/malecns_v1/graph.npz"
M = json.loads((GRAPH.parent / "manifest.json").read_text())
b = NativeBrain(GRAPH)
if a.arm.startswith("shuf"):
    g = np.random.default_rng(int(a.arm[4:])); post = b.post.copy(); g.shuffle(post); b.post[:] = post
d = NeuralControls(M["readouts"], mode="bci"); e = a.episode
game = Game(seed=5000 + e, scenario="combat_survival")
tick = 0; frames = []; kills = []; ctrl = []; sp_idx = []; sp_cnt = []; sp_ptr = [0]
maxt = int(a.max_seconds * 35); stick = int(a.spike_secs * 35)
while not game.game.is_episode_finished() and tick < maxt:
    frame = game.pixels()
    frames.append(frame[::2, ::2].copy()); kills.append(game.observation()["kills"])
    light = retinal_samples(frame, b.uv)
    tick += 1; steps = int(round(tick * 10000 / 35)) - b.cursor
    c, _ = b.step(light, steps * .1)
    act = d.decode(c, steps * .1 / 1000)
    game.act(act); ctrl.append((act["turn"], act["forward"], float(act["attack"])))
    if tick <= stick:
        nz = np.flatnonzero(c); sp_idx.append(nz.astype(np.int32)); sp_cnt.append(c[nz].astype(np.uint8)); sp_ptr.append(sp_ptr[-1] + len(nz))
obs = game.observation()
np.savez_compressed(a.out, frames=np.stack(frames), kills=np.array(kills), ctrl=np.array(ctrl, np.float32),
                    sp_idx=np.concatenate(sp_idx), sp_cnt=np.concatenate(sp_cnt), sp_ptr=np.array(sp_ptr, np.int64))
print(json.dumps(dict(arm=a.arm, episode=e, survival_s=tick / 35, kills=obs["kills"], died=bool(obs["finished"]), tics=tick)))
