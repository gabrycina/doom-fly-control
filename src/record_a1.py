"""Record footage of one pre-registered episode (index --episode, default 0 = first eval episode, no selection)
per arm, with the same engine seed and sampling seed as eval_arms.py. Saves frames (npz) + per-frame kill count."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, vizdoom as vzd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repos/doomfly-rl"))
sys.argv, argv = sys.argv[:1], sys.argv[1:]
ap = argparse.ArgumentParser()
ap.add_argument("--arm"); ap.add_argument("--ckpt"); ap.add_argument("--scenario", default="defend_the_center")
ap.add_argument("--episode", type=int, default=0); ap.add_argument("--rewire-seed", type=int, default=-1)
ap.add_argument("--device", default="mps"); ap.add_argument("--out", required=True)
a = ap.parse_args(argv)
# reuse eval_arms model construction by exec'ing its prefix
src = open(Path(__file__).parent / "eval_arms.py").read()
prefix = src[:src.index("out = open(a.out")]
prefix = prefix[prefix.index("RANDOM ="):]
a.connectome = "hf/connectome_malecns49k.npz"; a.threads = 2; a.greedy = False
dev = torch.device(a.device)
from doomfly.doom.actions import SCENARIOS
from doomfly.doom.env import DoomEnv
from doomfly.surgery import load_flynet
from doomfly.train import legal_mask_table
exec(prefix)
sc = SCENARIOS[a.scenario]
env = DoomEnv(a.scenario, render=True, seed=1000, record_tics=False)
g2l = {int(g): i for i, g in enumerate(env.sc.actions)}
e = a.episode
# identical to DoomEnv.reset(seed) but records a .lmp demo so we can replay every tic afterwards
from doomfly.doom.env import _resize_gray
lmp = str(Path(a.out).with_suffix(".lmp").resolve())
env.game.set_seed(1000 + e); env.game.new_episode(lmp); env.frames.clear()
f0 = _resize_gray(env.game.get_state().screen_buffer)
for _ in range(4): env.frames.append(f0)
obs = np.stack(env.frames); gen = torch.Generator().manual_seed(e)
R, n, done = 0.0, 0, False
kills_at = [0]
with torch.no_grad():
    while not done and n < 2100:
        fr = torch.from_numpy(obs[None]).to(dev)
        logits = model(fr, torch.tensor([sc.id], device=dev), legal_mask=legal[sc.id][None])["policy"].float()[0].cpu()
        act = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=gen))
        obs, r, done, _, _ = env.step(g2l[act]); R += r; n += 1
env.close()
# replay the demo tic by tic: exact same episode, every frame
g = vzd.DoomGame(); import os
g.load_config(os.path.join(vzd.scenarios_path, sc.cfg)); g.set_window_visible(False)
g.set_screen_resolution(vzd.ScreenResolution.RES_320X240); g.set_screen_format(vzd.ScreenFormat.RGB24)
g.set_available_buttons([getattr(vzd.Button, b) for b in sc.buttons]); g.set_render_hud(True)
if sc.doom_skill is not None: g.set_doom_skill(sc.doom_skill)
g.init(); g.replay_episode(lmp)
frames, kills = [], []
while not g.is_episode_finished():
    st = g.get_state(); frames.append(st.screen_buffer.copy()); kills.append(g.get_game_variable(vzd.GameVariable.KILLCOUNT))
    g.advance_action()
replay_ret = g.get_total_reward(); g.close()
frames = np.stack(frames); kills_at = np.array(kills)
np.savez_compressed(a.out, frames=frames, kills=kills_at, ret=R, steps=n, replay_ret=replay_ret)
print(json.dumps(dict(arm=a.arm, episode=e, ret=R, steps=n, frames=len(frames), final_kills=float(kills_at[-1]), replay_ret=replay_ret)))
