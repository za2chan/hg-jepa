#!/usr/bin/env bash
# Re-measure everything with the xcov weight set PER STEM (L1 lam=1, NCE lam=16)
# instead of the shared lam=4.
#
# Why: the +-3% sweep showed lam=4 destroys L1's inclusion (0.964 at lam=1 ->
# 0.524 at lam=4) while NCE still improves up to lam=16. Every L1 number reported
# so far -- the 2x2, the difficulty curve, the loss to post-hoc ICA -- was measured
# at a lam that was actively hurting it.
#
# Checks three things at once:
#   1. does the L1 collapse across difficulty disappear at lam=1
#   2. does L1 beat post-hoc ICA/SFA once lam is right
#   3. does per-stem lam transfer to REAL data or is it a synthetic artifact
#
# ponytail: plain bash job slots, no xargs. xargs does its own quote processing
# and strips the quotes before bash sees them, which silently broke two earlier
# batches (jobs ran with the wrong args, all writing to one log).
set -u
cd "$(dirname "$0")"
export HGLP_LAM='{"l1": 1.0, "nce": 16.0}'
export HGLP_LAM_TAG=_perstemlam
export HGLP_NSEED=5
LOG=/tmp/perstem; mkdir -p "$LOG"
MAXPAR=5

launch() {                       # launch <gap> <logname> <script> [args...]
  local gap=$1 name=$2; shift 2
  while (( $(jobs -rp | wc -l) >= MAXPAR )); do wait -n; done
  echo "start $name : HGLP_GAP=$gap $*"
  HGLP_GAP=$gap python3 "$@" > "$LOG/$name.log" 2>&1 &
}

launch 0.05 tw005    twosided.py synth
launch 0.03 tw003    twosided.py synth
launch 0.02 tw002    twosided.py synth
launch 0.01 tw001    twosided.py synth
launch 0.03 ln003    twosided.py synth ln2x2
launch 0.03 rot_l1   rotation.py synth l1+ema
launch 0.03 rot_nce  rotation.py synth nce+ema
launch 0.05 tw_ptbxl twosided.py ptbxl
launch 0.05 tw_hapt  twosided.py hapt
wait

echo "ALL DONE $(TZ=America/Toronto date '+%F %H:%M %Z')" | tee "$LOG/DONE"
