#!/bin/bash
cd "$(dirname "$0")/.." && source .venv-brian/bin/activate
while pgrep -f "partB_shiu.py --conds shuf0" >/dev/null; do sleep 10; done
python src/partB_shiu.py --conds real --stim bitter --rate 100 --n_run 30 --n_proc 6
python src/partB_shiu.py --conds shuf0 shuf1 shuf2 real --stim sugar --rate 200 --n_run 30 --n_proc 6
