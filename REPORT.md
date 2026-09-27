# Report: lesioning the DOOM fly

## Result in one line
Trained DOOM fly (doomfly-rl): real wiring, degree-preserving shuffled wiring and no connectome score
the same on all 5 ViZDoom scenarios (100 seed-matched games each). Untrained 166,700-neuron spiking fly
(nftechie/doomfly): shuffled wiring loses, but a brainless constant 3-number autopilot plays as well as
the real one. Shiu et al. brain model (no ML): the same kind of shuffle abolishes the sugar -> proboscis
motor response (MN9 66 Hz -> 0 Hz).

## What the video shows (42.5 s, 1920x1080, 30 fps, 12.1 MB, silent, captions burned in)
0.0-14.1  "shuffle the fly brain's wiring. it still plays doom." 3 doomfly-rl checkpoints play
          defend_the_center side by side (real / shuffled / no connectome), live kills + 100-game mean
          with 95% CI (19.1 / 19.6 / 19.3). Eval game #1, same seeds, fixed before viewing.
14.4-19.9 5-scenario bar chart, 100 games/arm, 95% CI; grey refs "real, rewired after training", "random".
20.2-26.2 untrained 166,700-neuron LIF fly vs brainless autopilot (turn 0.81, forward 19.2, fire held).
26.5-32.0 spiking-fly chart: real (n=20), shuffled (3 seeds, n=12), eyes off (n=8), no connections (n=8),
          autopilot (n=20).
32.3-38.8 Part B: Shiu model, MN9 raster, 30 trials, real vs 3 shuffles (66.1 Hz vs 0/0/0).
39.1-42.5 end card: "the connectome is a real circuit. it just isn't what's playing doom."

## Honesty ledger
Real data: MaleCNS v1.0 (SHA-256 matched; reproduces 166,700 neurons / 25,582,938 edges / 124,177,617
synapses); doomfly-rl MaleCNS-49k subgraph (49,393 neurons, 9,050,172 edges; HF rev 2e58329); FlyWire
v630 (Shiu repo commit 91bdd1e). All game scores from ViZDoom episodes run here.
Trained by others: the 3 doomfly-rl checkpoints (author's recipe, 60k distillation steps from PPO
teachers). Nothing retrained. Shuffle verified (results/shuffle_check.json): edge list == their
shuffle_connectome(seed=0); 1.3% edge overlap with real; degrees exact except 47 duplicate drops.
Modeled/engineered (unchanged from repos): doomfly-rl = conv stem -> rate network on fly wiring with
learned per-edge gains -> MLP. nftechie = LIF with Shiu params, inferred retina, engineered motor readout
(DNp20 L/R turn, DNpe017 forward/fire) chosen by calibrating on the real wiring. Shiu = published code,
only the connectivity file swapped.
Own choices: degree- and sign-preserving shuffles in all three systems (nftechie: postsynaptic targets
permuted across 25.6M edges, preserving output mass not input mass); spiking rounds capped at 60 s with
fresh state; "rewired after training" is an extra probe; matched autopilot constants are the real fly's
own mean output (fitted) -- guarded by an untuned 12-point grid whose best point (turn 1, fwd 20) also
matches.
Leak checks: seed matching (engine 1000+k, sampling k); CPU==MPS; footage replays eval game #1 exactly
(repo's own recorder changes trajectories, not used); pixels only; same train/test protocol across arms.

## Numbers
A1 doomfly-rl, 100 episodes/arm/scenario, bootstrap 95% CI (10k):
| scenario | real | shuffled s0 | no connectome | rewired after training | random |
| defend_the_center | 18.08 [17.70,18.44] | 18.56 [18.17,18.93] | 18.29 [17.86,18.73] | 2.31 | 0.64 |
| defend_the_line | 21.58 [20.61,22.55] | 21.76 [20.79,22.74] | 21.49 [20.60,22.37] | 10.39 | 8.26 |
| health_gathering | 1579 [1447,1708] | 1574 [1449,1697] | 1512 [1383,1637] | 783 | 384 |
| basic | 80.57 [78.5,82.5] | 80.72 [78.8,82.6] | 80.72 [78.8,82.6] | -148 | -131 |
| deadly_corridor | 2280.35 | 2280.29 | 2266.6 [2239,2281] | 570 | 170 |
Only nominally non-zero gap favours the shuffle (dtc +0.48 [+0.04,+0.93]). The trained net does use its
own wiring (rewired after training -> near random), so "the connectome does nothing" is too strong; the
claim is "which wiring it is trained through doesn't matter".

A2 nftechie/doomfly, combat_survival, 60 s cap:
real 53.3 s [49.7,56.8], 12.10 kills (n=20) | shuffled pooled 6.7 s, 0.67 kills (n=12) | eyes off 10.5 s,
2.5 kills (n=8) | no connections 5.5 s (n=8) | matched autopilot 51.9 s, 11.75 kills (n=20) | best untuned
autopilot 53.4 s, 13.25 kills (n=16). Real fly output is near-constant: turn 0.80±0.56 deg/tic, forward
19.2±1.2 (clamp 20), fire held 94%.

Part B Shiu, 21 sugar GRNs at 100 Hz, 30 x 1 s trials: MN9 L 66.1 Hz real vs 0 in 3 shuffles (also 0 at
200 Hz); bitter -> 0. Caveat: shuffled brains are quieter overall (~50-100 vs ~400 active neurons).

## Reply-guy defence
1. "doomfly-rl already ran this control." True (10 episodes/scenario, no video): credit them. New here:
   100 seed-matched episodes with CIs, verified shuffle, rewire probe, the spiking-fly autopilot
   comparison, and Part B.
2. "Unfair null, a random graph is dead." In doomfly-rl the shuffle isn't dead. In the spiking fly the
   defensible claim is the autopilot comparison, not the shuffle.
3. "That's a 49k subset, not 125M synapses." Correct and labelled; the full graph appears only in the
   spiking fly.

## Hook drafts
1. the doom-playing fly brain went viral so i ran the control. shuffle all its wiring and it plays exactly as well. delete the connectome entirely, same score. the fly part isn't doing the playing. credit to the doomfly-rl author who ran a 10 game version first, this is 100 games per arm plus the spiking one
2. i tried to find what the fly connectome contributes to the doom demos. trained version: nothing, a shuffled or deleted wiring scores the same. untrained spiking version: it basically circles and holds fire, and a 3 number autopilot with no brain does just as well. then i ran the fly taste model with the wiring shuffled and feeding just disappears. the wiring is real, it just isn't playing doom
3. same fly wiring, two experiments. doom: shuffle it, nothing changes. sugar to proboscis: shuffle it, the motor neuron goes from 66 hz to zero. connectomes are for asking questions like the second one

## Reproduce
See src/ (run_a1.sh, eval_arms.py, analyze_a1.py, record_a1.py, run_a2.sh, run_a2_spray.sh, run_a2.py,
analyze_a2.py, partB_shiu.py, analyze_b.py, charts.py, render_video.py). Envs: .venv, .venv-doomfly,
.venv-brian. Repos under repos/ (code reviewed before running; no sudo, nothing outside this dir).

## Cost
Mac only, ~4.5 h wall. GPU box never used (start attempts hit NotEnoughResources / concurrent op).
~2.2 GB downloads, 4.5 GB on disk. No money spent, no accounts created.

## Weakest part
One trained shuffle seed (only the author's published one); spiking-fly shuffles n=4/seed; fitted
autopilot constants; Part B shuffle also quiets the network (needs an activity-matched null).
