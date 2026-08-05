#!/usr/bin/env bash
# Everything the 2026-08-05 question round needs, chained so the GPU stays busy.
# Waits for the block x factor sweep, then re-runs the post-hoc rotation two-sided,
# the domain shift with an ungated control, and the difficulty calibration.
set -u
cd "$(dirname "$0")"
log() { echo "[$(date +%H:%M:%S)] $*"; }

while pgrep -f "twosided.py" > /dev/null; do sleep 30; done
log "twosided done"

for ds in synth ptbxl hapt; do
  ( for stem in nce+ema l1+ema; do
      log "rotation $ds $stem"
      python3 rotation.py "$ds" "$stem" > "/tmp/rot_${ds}_${stem}.log" 2>&1
    done ) &
done
wait
log "rotation done"

for stem in nce+ema l1+ema; do
  log "domain_shift $stem"
  python3 domain_shift.py 3 3 2500 "$stem" > "/tmp/ds_${stem}.log" 2>&1
  cp ../../runs_v2/domain_shift_hapt.json "../../runs_v2/domain_shift_hapt_${stem}.json"
done
log "domain_shift done"

python3 difficulty.py train > /tmp/difficulty_train.log 2>&1
log "difficulty done"
python3 fastaxis.py > /tmp/fastaxis.log 2>&1
log "ALL DONE"
