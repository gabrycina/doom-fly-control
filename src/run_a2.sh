#!/bin/bash
cd "$(dirname "$0")/.." && source .venv-doomfly/bin/activate
export OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
mkdir -p results/a2 footage/a2
r() { nohup nice -n 10 python src/run_a2.py --arm $1 --start $2 --episodes $3 --max-seconds 60 $4 --out results/a2/$1.jsonl > results/a2/$1.$2.log 2>&1 & }
r real 0 8 "--record footage/a2"
r real 8 8
r shuf0 0 4 "--record footage/a2"
r shuf1 0 4
r shuf2 0 4
r blind 0 8
r noconn 0 8
