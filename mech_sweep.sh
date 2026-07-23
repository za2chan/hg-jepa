#!/bin/bash
set -e; cd "$(dirname "$0")"
# (a) target masking: does tmask help gate-alone or gate+dcor?
for s in 0 1 2; do
  python3 mech.py dz=64 dslow=16 gate=1 dcor=0 tmask=0 seed=$s
  python3 mech.py dz=64 dslow=16 gate=1 dcor=0 tmask=1 seed=$s
  python3 mech.py dz=64 dslow=16 gate=1 dcor=1 tmask=0 seed=$s
  python3 mech.py dz=64 dslow=16 gate=1 dcor=1 tmask=1 seed=$s
done
# (b) width sweep: does gate+dcor separation survive wider embeddings?
for s in 0 1; do
  for dz in 128 256; do
    python3 mech.py dz=$dz dslow=$((dz/4)) gate=1 dcor=1 tmask=0 seed=$s   # proportional block
    python3 mech.py dz=$dz dslow=16       gate=1 dcor=1 tmask=0 seed=$s   # fixed block
  done
done
echo MECH_DONE
