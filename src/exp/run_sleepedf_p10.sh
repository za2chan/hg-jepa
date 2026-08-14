#!/bin/bash
# Sleep-EDF patch-10 re-run: configs A and C x two stems, 3 seeds.
#
# Why this exists. The patch-50 setting put the transient proxy's autocorrelation
# time at 1.0 patch -- the floor at which patch averaging erases the factor the
# exclusion term is supposed to measure -- and forced tau to be hand-set to 24
# because the D1 rule returned 3, below dmin. At patch 10 the proxy T_ac is 3.0
# patches and the rule returns tau=9, so config A is the first real dataset whose
# threshold is NOT hand-picked. Config C keeps a hand-set tau=16 as the control:
# it gives z_mix far more gradient (33% of horizons below tau, against A's 5.7%),
# so the pair separates "the rule's value" from "a value that trains well".
#
# Ordering matters for the checkpoint cache: the ungated control ignores tau, so
# running A before C means C reuses A's ungated encoders instead of retraining
# them (6 of the 24 trainings).
set -u
cd "$(dirname "$0")"
export HGLP_NSEED=3
export HGLP_NPZ=../../data/sleepedf_p10.npz
# DMAX=64 is what THESE results were produced with, kept so they stay reproducible.
# For NEW Sleep-EDF work use 128. The other three datasets all put dmax at half the
# window (128/256 synth, 128/256 HAPT, 48/100 PTB-XL); 64 is a quarter, and nothing
# was ever written down to justify the exception -- it is inherited, not chosen.
# run_sleepedf_p10_dmax.sh re-ran tau=16 at dmax=128: every difference sat inside the
# seed spread and the optimal lambda stayed 0 on both stems. So this is a convention
# fix, not a results fix, and switching the default here would orphan config A.
export HGLP_W=8 HGLP_DMIN=8 HGLP_DMAX=64 HGLP_STEPS=5000

run () {                       # run <tag> <tau> <stem>
  local tag=$1 tau=$2 stem=$3
  local out=/tmp/rot_p10_${tag}_${stem%%+*}.log
  echo "=== $(date +%H:%M) start ${tag} tau=${tau} ${stem} ==="
  HGLP_TAU=$tau HGLP_TAG=_p10${tag} \
    python3 rotation.py sleepedf "$stem" > "$out" 2>&1
  echo "    exit=$? -> $out"
}

run A 9  nce+ema
run A 9  l1+ema
run C 16 nce+ema
run C 16 l1+ema

echo "=== all done $(date +%H:%M) ==="
ls -la ../../runs_v2/rotation_sleepedf_*_p10*.json 2>/dev/null
