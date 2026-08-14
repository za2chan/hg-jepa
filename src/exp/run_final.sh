#!/usr/bin/env bash
# The two batches left before writing.
#
# A. Main 2x2 at 5 seeds, one file per dataset.
#    The four cells currently live in three different files per dataset, and the
#    real-data ones are 3-seed while synthetic is 5-seed -- n must not vary inside
#    one table. `main22` runs exactly the reported axis (per-block LN travels with
#    the xcov arm) in a single run so every cell shares seeds 0-4.
#
# B. Part 2 with the SSL baselines on all three datasets.
#    Metric: (score in the clean domain) - (score in the perturbed domain) as the
#    probe label budget shrinks. That is a WITHIN-arm difference, so TS2Vec's 320
#    and PatchTST's C*128 dimensions do not have to match our 64.
#    HAPT caveat to restate when reporting: last-position scoring leaves ~955 train
#    rows against PatchTST's 1536 dims, so its post-hoc subspaces are underdetermined.
set -u
cd "$(dirname "$0")"
export HGLP_NSEED=5
LOG=/tmp/final; mkdir -p "$LOG"
MAXPAR=5

launch() {
  local name=$1; shift
  while (( $(jobs -rp | wc -l) >= MAXPAR )); do wait -n; done
  echo "start $name  ($(TZ=America/Toronto date '+%H:%M'))"
  "$@" > "$LOG/$name.log" 2>&1 &
}

# A -- main 2x2
HGLP_GAP=0.03 launch m22_synth python3 twosided.py synth main22
launch m22_ptbxl python3 twosided.py ptbxl main22
launch m22_hapt  python3 twosided.py hapt  main22

# B -- Part 2 with baselines (5 seeds, 3 label draws per budget)
HGLP_GAP=0.03 launch p2_synth python3 label_x_perturb.py synth 2500 5 3
launch p2_ptbxl python3 label_x_perturb.py ptbxl 2500 5 3
launch p2_hapt  python3 label_x_perturb.py hapt  2500 5 3
wait

echo "ALL DONE $(TZ=America/Toronto date '+%F %H:%M %Z')" | tee "$LOG/DONE"
