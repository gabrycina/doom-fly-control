# Is the fly brain actually playing DOOM?

In September 2026 the complete wiring map of a male fruit fly's nervous system was published
(MaleCNS v1.0, Janelia / Google: 166,700 neurons, 124M synapses). Within days, videos of
"the fly brain playing DOOM" went viral. This repo runs the control: break the brain and see if it
still plays.

https://github.com/gabrycina/doom-fly-control/raw/main/media/doom_fly_control.mp4

## What we found

**The trained DOOM fly** ([doomfly-rl](https://github.com/nonatofabio/doomfly-rl)) is a standard AI
"eye" (looks at the screen), then a network wired like the fly brain, then a standard AI "hand"
(presses buttons), trained end to end. We played the author's three published versions, trained the
same way with the **real wiring**, the **wiring reconnected at random**, and **no wiring at all**:
100 seed-matched games each, on all five ViZDoom scenarios. They score the same (e.g. 19.1 / 19.6 /
19.3 kills per game on defend-the-center). The trained network does use its wiring (scramble it
*after* training and it drops to near random), but which wiring it was trained through doesn't matter.

**The untrained DOOM fly** ([nftechie/doomfly](https://github.com/nftechie/doomfly)) simulates all
166,700 neurons with no learning. Scrambling its 25.6M connections kills it fast (7 s vs 53 s alive).
But its output is nearly constant: turn slightly right, walk forward, keep shooting. A robot that
does exactly that, with no brain, survives just as long (52 s vs 53 s, 12 vs 12 kills, 20 rounds each).

**The fly brain isn't what's playing DOOM.** (The wiring map is still real science: in the
no-AI whole-brain model of Shiu et al. 2024, sugar drives the feeding motor neuron MN9 at 66 Hz with
the real wiring and 0 Hz with it scrambled; see `partB/`.)

Credit: **nonatofabio (doomfly-rl) ran a smaller 10-game version of this control first.** This repo
extends it to 100 seed-matched games per arm with confidence intervals, verifies the shuffle, adds the
rewire-after-training probe, the spiking-fly autopilot comparison and the taste experiment.

Full numbers, honesty ledger, leak checks and caveats: [`REPORT.md`](REPORT.md).

## Layout
```
src/       experiment, analysis, chart, video and audio scripts
results/   per-episode results (jsonl) and summaries for the DOOM experiments, shuffle verification
partB/     Shiu-model taste experiment: summary and spike trains
media/     the video and poster
```

## Reproduce
Clone the upstream projects into `repos/` (their code is not redistributed here):
- https://github.com/nonatofabio/doomfly-rl (checkpoints: Hugging Face `fabiononato/doomfly-rl`, revision `2e58329`)
- https://github.com/nftechie/doomfly (downloads MaleCNS v1.0 with hash checks)
- https://github.com/philshiu/Drosophila_brain_model (commit `91bdd1e`, FlyWire v630)

Then see the Reproduce section of `REPORT.md` (`src/run_a1.sh`, `src/run_a2.sh`, `src/partB_shiu.py`,
`src/analyze_*.py`, `src/render_v2.py`, `src/audio_v2.py`). Everything ran on an Apple M3 Max laptop.

## Media
Music and sound effects are synthesised (`src/audio_v2.py`); game sounds are from Freedoom
(BSD-3-Clause). The fly drawing is our own illustration. Neuron activity shown and heard in the video
is the real simulated spike output.

## Licence
Code: MIT. Results, video and text: CC BY 4.0. Upstream projects and data keep their own licences.
