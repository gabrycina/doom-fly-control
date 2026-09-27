#!/bin/bash
# Seed-matched eval of the three published doomfly-rl checkpoints + a post-training rewire lesion.
cd "$(dirname "$0")/.." && source .venv/bin/activate
EP=${EP:-100}; START=${START:-0}
run() { nohup python src/eval_arms.py --arm $1 --ckpt $2 --rewire-seed $3 --episodes $EP --start $START --device mps --threads 2 --out results/a1/$1.jsonl > results/a1/$1.$START.log 2>&1 & }
mkdir -p results/a1
run real hf/malecns49k_v2_final -1
run shuf0 hf/malecns49k_shuffled_s0 -1
run noconn hf/malecns49k_noconn -1
run real_rewired_after_training hf/malecns49k_v2_final 0
