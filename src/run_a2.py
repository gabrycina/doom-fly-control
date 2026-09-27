"""Part A2: lesion the nftechie/doomfly fixed-weight 166,700-neuron LIF fly (MaleCNS v1.0) in its
combat_survival arena, using the repo's own NativeBrain kernel, retina, and fixed 'bci' decoder unchanged.

Arms (fresh neural state each round; round = spawn until death, capped at --max-seconds game time):
  real        : graph.npz as prepared by the repo
  shufK       : degree-preserving rewire, seed K: every edge keeps its presynaptic neuron, its weight
                (synapse count x presynaptic sign x 0.275 mV); postsynaptic targets are permuted across all
                25.6M edges -> exact in- and out-degree, exact per-neuron sign and output synapse mass.
                Retina, lamina, readout neuron identities unchanged.
  blind       : real wiring, all photoreceptor input zero (the repo's own 'blank_vision' condition)
  noconn      : all edge weights zero (the repo's 'all_edges_disconnected'): the decoder receives no DN activity
"""
import argparse, json, sys, time, hashlib
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1] / "repos/doomfly"
sys.path.insert(0, str(REPO))
from doom.native import NativeBrain
from doom.engine import NeuralControls
from doom.game import Game, retinal_samples

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True); ap.add_argument("--episodes", type=int, default=8)
ap.add_argument("--start", type=int, default=0); ap.add_argument("--max-seconds", type=float, default=90)
ap.add_argument("--out", required=True)
ap.add_argument("--turn", type=float, default=0.0); ap.add_argument("--forward", type=float, default=0.0); ap.add_argument("--record", default=None, help="save frames of these episodes (npz dir)")
a = ap.parse_args()
GRAPH = REPO / "outputs/doom/malecns_v1/graph.npz"
M = json.loads((GRAPH.parent / "manifest.json").read_text())

def shuffled_post(brain, seed):
    g = np.random.default_rng(seed)
    post = brain.post.copy(); g.shuffle(post)
    return post

out = open(a.out, "a")
SPRAY = a.arm.startswith("spray")   # no brain: fixed autopilot, trigger held every tic
for e in range(a.start, a.start + a.episodes):
    b = None if SPRAY else NativeBrain(GRAPH)
    if a.arm.startswith("shuf"):
        b.post[:] = shuffled_post(b, int(a.arm[4:]))
    if a.arm == "noconn":
        b.weight.fill(0)
    d = NeuralControls(M["readouts"], mode="bci"); turns = []; fwds = []
    game = Game(seed=5000 + e, scenario="combat_survival")
    t0 = time.time(); tick = 0; attack = 0; moving = 0; frames = []; kills_t = []
    maxt = int(a.max_seconds * 35)
    while not game.game.is_episode_finished() and tick < maxt:
        frame = game.pixels()
        if a.record is not None and e in (0,):
            frames.append(frame[::2, ::2].copy()); kills_t.append(game.observation()["kills"])
        tick += 1
        if SPRAY:
            act = {"turn": a.turn, "forward": a.forward, "attack": True}
        else:
            light = retinal_samples(frame, b.uv)
            if a.arm == "blind": light.fill(0)
            steps = int(round(tick * 10000 / 35)) - b.cursor
            c, _ = b.step(light, steps * .1)
            act = d.decode(c, steps * .1 / 1000)
        turns.append(act["turn"]); fwds.append(act["forward"])
        game.act(act); attack += int(act["attack"]); moving += int(abs(act["turn"]) > 1e-9 or abs(act["forward"]) > 1e-9)
    obs = game.observation()
    rec = dict(arm=a.arm, episode=e, seed=5000 + e, died=bool(obs["finished"]), survival_s=tick / 35, kills=obs["kills"],
               kills_per_min=obs["kills"] / (tick / 35 / 60), attack_tics=attack, moving_tics=moving, tics=tick,
               spikes=0 if SPRAY else int(b.total_spikes), mean_turn=float(np.mean(turns)), mean_abs_turn=float(np.mean(np.abs(turns))),
               mean_forward=float(np.mean(fwds)), spray_turn=a.turn if SPRAY else None, spray_forward=a.forward if SPRAY else None, wall=round(time.time() - t0, 1))
    out.write(json.dumps(rec) + "\n"); out.flush(); print(json.dumps(rec), flush=True)
    if frames:
        Path(a.record).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(a.record) / f"{a.arm}_ep{e}.npz", frames=np.stack(frames), kills=np.array(kills_t))
    if a.record is not None and not SPRAY:
        Path(a.record).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(a.record) / f"{a.arm}_trace_ep{e}.npz", turn=np.array(turns), forward=np.array(fwds))
    game.close(); del b
