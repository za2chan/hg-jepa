#!/bin/bash
# Full leak-free re-run: synthetic + HAPT + PTB-XL (3 seeds) + XJTU.
set -e
cd "$(dirname "$0")"

echo "### synthetic ###"
bash run_all.sh

echo "### HAPT (subject-split) ###"
for s in 0 1 2; do
  for a in "gate=1 dcor=0" "gate=0 dcor=0" "gate=1 dcor=1" "gate=0 dcor=1"; do
    python3 hapt_train.py mode=nepa $a seed=$s
  done
  python3 hapt_train.py mode=ar gate=1 dcor=0 seed=$s
  python3 hapt_train.py mode=ar gate=0 dcor=0 seed=$s
done

echo "### PTB-XL (patient-split) ###"
for s in 0 1 2; do
  for a in "gate=1 dcor=0" "gate=0 dcor=0" "gate=1 dcor=1" "gate=0 dcor=1"; do
    python3 ptbxl_train.py mode=nepa $a seed=$s
  done
  python3 ptbxl_train.py mode=ar gate=1 dcor=0 seed=$s
  python3 ptbxl_train.py mode=ar gate=0 dcor=0 seed=$s
done

echo "### XJTU raw-AM (low-pass baseline) ###"
python3 raw_am_train.py mode=nepa gate=1 dcor=1 seed=0
python3 raw_am_train.py mode=nepa gate=0 dcor=0 seed=0

echo "### XJTU snapshot (held-out bearings, negative control) ###"
python3 real_train.py mode=nepa gate=1 dcor=1 seed=0
python3 real_train.py mode=nepa gate=0 dcor=0 seed=0

echo ALL_EXPERIMENTS_DONE
