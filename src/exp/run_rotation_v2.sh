#!/usr/bin/env bash
# Post-hoc rotation baseline, re-measured. Four defects fixed:
#   1. the "ungated" control had per-block LayerNorm ON, which privileges the
#      coordinate split on its own -- it was never mechanism-free
#   2. z_full (the exclusion denominator) was borrowed from a DIFFERENT training
#      run because this script never scored it; now scored in-run, per row-source
#   3. 3 seeds -> 5, and per-seed values stored so a PAIRED test is possible
#   4. the gate was compared at one lambda; the post-hoc methods have no comparable
#      knob, so the gate is now measured across lam = 0,1,4,16,64 and reported as a
#      curve against their single points
#
# 3 datasets x 2 stems x 5 seeds x 6 trainings (1 ungated + 5 gated) = 180 runs.
# ponytail: plain bash job slots -- xargs strips quotes and broke two prior batches.
set -u
cd "$(dirname "$0")"
export HGLP_NSEED=5
export HGLP_LAMS=0,1,4,16,64
LOG=/tmp/rotv2; mkdir -p "$LOG"
MAXPAR=5

launch() {                       # launch <gap> <dataset> <stem>
  local gap=$1 ds=$2 stem=$3
  while (( $(jobs -rp | wc -l) >= MAXPAR )); do wait -n; done
  echo "start $ds/$stem gap=$gap  ($(TZ=America/Toronto date '+%H:%M'))"
  HGLP_GAP=$gap python3 rotation.py "$ds" "$stem" > "$LOG/${ds}_${stem}.log" 2>&1 &
}

for stem in l1+ema nce+ema; do
  launch 0.03 synth "$stem"
  launch 0.05 ptbxl "$stem"      # gap is unused on real data
  launch 0.05 hapt  "$stem"
done
wait

echo "ALL DONE $(TZ=America/Toronto date '+%F %H:%M %Z')" | tee "$LOG/DONE"
