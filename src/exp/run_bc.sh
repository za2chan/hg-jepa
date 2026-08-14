#!/usr/bin/env bash
# B: synthetic Part 2 with L1 at lam=1.
#    lam=4 collapses L1's inclusion at +-3% (0.964 -> 0.524), so the cost measured
#    there is a hyperparameter artefact, not the mechanism's price. NCE is fine at
#    lam=4 and is left alone, so this run carries {"l1": 1.0} only.
#
# C: main 2x2 re-run so the table carries its OWN random-split control.
#    block_factor's rand16/rand48 were independent draws at each width, not a
#    split, so they could never form a SEP; the numbers quoted so far were borrowed
#    from the post-hoc-rotation files, a different training run. On HAPT the random
#    split scores ~0.34-0.54, high enough that a table without it reads far too
#    favourably, so this control has to come from the same run.
set -u
cd "$(dirname "$0")"
export HGLP_NSEED=5
LOG=/tmp/bc; mkdir -p "$LOG"
MAXPAR=4

launch() {
  local name=$1; shift
  while (( $(jobs -rp | wc -l) >= MAXPAR )); do wait -n; done
  echo "start $name  ($(TZ=America/Toronto date '+%H:%M'))"
  "$@" > "$LOG/$name.log" 2>&1 &
}

# C -- main 2x2, now with randsplit_slow / randsplit_fast rows
HGLP_GAP=0.03 launch m22_synth python3 twosided.py synth main22
launch m22_ptbxl python3 twosided.py ptbxl main22
launch m22_hapt  python3 twosided.py hapt  main22

# B -- synthetic Part 2, L1 at lam=1
HGLP_GAP=0.03 HGLP_LAM='{"l1": 1.0}' HGLP_TAG=_l1lam1 \
  launch p2_synth_lam1 python3 label_x_perturb.py synth 2500 5 3
wait

echo "ALL DONE $(TZ=America/Toronto date '+%F %H:%M %Z')" | tee "$LOG/DONE"
