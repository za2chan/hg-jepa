#!/bin/bash
set -e
cd "$(dirname "$0")"
# reproducibility: 3 seeds x {nepa, ar} x {gate on/off} (+ dcor for nepa)
for s in 0 1 2; do
  for m in nepa ar; do
    python3 train.py mode=$m gate=1 dcor=0 seed=$s
    python3 train.py mode=$m gate=0 dcor=0 seed=$s
  done
  python3 train.py mode=nepa gate=1 dcor=1 seed=$s
  python3 train.py mode=nepa gate=0 dcor=1 seed=$s
done
# tuning sweeps (seed 0)
python3 train.py mode=nepa gate=1 dcor=0 seed=0 tau=4
python3 train.py mode=nepa gate=1 dcor=0 seed=0 tau=64
python3 train.py mode=nepa gate=1 dcor=0 seed=0 dslow=8
python3 train.py mode=nepa gate=1 dcor=0 seed=0 dslow=32
python3 train.py mode=nepa gate=1 dcor=1 seed=0 lam=1
python3 train.py mode=nepa gate=1 dcor=1 seed=0 lam=16
echo ALL_DONE
