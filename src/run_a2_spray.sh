#!/bin/bash
cd "$(dirname "$0")/.." && source .venv-doomfly/bin/activate
for t in 0 1 2 4; do for f in 0 10 20; do
  python src/run_a2.py --arm spray_t${t}_f${f} --turn $t --forward $f --episodes 16 --max-seconds 60 --out results/a2/spray.jsonl
done; done
