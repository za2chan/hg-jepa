#!/usr/bin/env bash
# Remaining work after rotation. Domain shift now runs the BLOCK split (label mix
# matched: TV(A_test,B) 0.395 -> 0.073) as primary, and the original TIME split
# after it so the two designs can be compared under identical code rather than
# silently swapped.
set -u
cd "$(dirname "$0")"
log() { echo "[$(date +%H:%M:%S)] $*"; }

for mode in block time; do
  for stem in nce+ema l1+ema; do
    log "domain_shift $stem $mode"
    python3 domain_shift.py 3 3 2500 "$stem" "$mode" > "/tmp/ds_${stem}_${mode}.log" 2>&1 &
  done
  wait
  log "domain_shift $mode done"
done

log "difficulty train"
python3 difficulty.py train > /tmp/difficulty_train.log 2>&1
log "fastaxis"
python3 fastaxis.py > /tmp/fastaxis.log 2>&1

for ds in hapt synth ptbxl; do
  log "blocknorm ablation $ds"
  python3 twosided.py "$ds" bn > "/tmp/ts_bn_${ds}.log" 2>&1
done
log "ALL DONE"
