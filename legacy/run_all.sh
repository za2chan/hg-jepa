#!/bin/bash
# Synthetic sweep -> runs/ (consumed by figures.py). Hard two-timescale data.
# Probe eval is leak-free (disjoint windows + contiguous split).
set -e
cd "$(dirname "$0")"
for s in 0 1 2; do
  python3 train.py mode=nepa gate=1 dcor=1 seed=$s        # ours
  python3 train.py mode=nepa gate=1 dcor=0 seed=$s        # gate only
  python3 train.py mode=nepa gate=0 dcor=0 seed=$s        # no gate
  python3 train.py mode=nepa gate=0 dcor=1 seed=$s        # dcor only
  python3 train.py mode=ar   gate=1 dcor=0 seed=$s        # AR + gate (H3)
  python3 train.py mode=ar   gate=0 dcor=0 seed=$s        # AR no gate
done
# tuning sweeps (seed 0)
for tau in 4 64;   do python3 train.py mode=nepa gate=1 dcor=0 seed=0 tau=$tau; done
for ds  in 8 32;   do python3 train.py mode=nepa gate=1 dcor=0 seed=0 dslow=$ds; done
for lam in 1 16;   do python3 train.py mode=nepa gate=1 dcor=1 seed=0 lam=$lam; done
echo SYNTH_DONE
