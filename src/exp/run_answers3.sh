#!/usr/bin/env bash
# Re-run only the BLOCK configs with the final metric set (delta_matched), then
# finish the remaining measurements. The TIME configs keep their existing results:
# they exist to answer "does fixing the label mix change the conclusion?", and the
# matched metric turned out to agree with the raw one, so re-running them is waste.
set -u
cd "$(dirname "$0")"
log() { echo "[$(date +%H:%M:%S)] $*"; }
while kill -0 3828450 2>/dev/null; do sleep 30; done
log "time split done"
for stem in nce+ema l1+ema; do
  log "domain_shift $stem block (final metrics)"
  python3 domain_shift.py 3 3 2500 "$stem" block > "/tmp/ds_${stem}_block.log" 2>&1 &
done
wait
log "domain_shift done"
log "difficulty train"; python3 difficulty.py train > /tmp/difficulty_train.log 2>&1
log "fastaxis";         python3 fastaxis.py      > /tmp/fastaxis.log 2>&1
for ds in hapt synth ptbxl; do
  log "blocknorm ablation $ds"
  python3 twosided.py "$ds" bn > "/tmp/ts_bn_${ds}.log" 2>&1
done
log "ALL DONE"
