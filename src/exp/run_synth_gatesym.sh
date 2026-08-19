#!/bin/bash
# Symmetric gate on the synthetic testbed, both stems, 5 seeds x 5 lambdas.
#
# The idea. train_real.py:156 gates z_mix only and lets z_slow through untouched.
# That asymmetry is exactly why the Exclusion clause says "the objective cannot force
# fast information out of z_slow": with z_slow always on, transient content in it is
# free profit at short Delta. The symmetric form weights z_slow by 1-g, so it is muted
# below tau -- transient content then pays nowhere, because at long Delta the
# transient factor is already decorrelated.
#
# Why synthetic first. This is the one testbed where the mechanism demonstrably works
# (we beat SFA by +0.395 and +0.451 on the product). Before asking whether the change
# FIXES the real-data exclusion losses, the first question is whether it BREAKS what
# already works. If synthetic survives, HAPT and PTB-XL are the informative next runs,
# since section 7.1 showed every real-data loss lives in the exclusion term.
#
# Seeds and lambdas match rotation_synth_*_gap0.03.json exactly so the comparison is
# paired seed by seed. Synthetic trains in-process with no checkpoint cache, so all
# 5 seeds x (5 lambdas + 1 ungated) x 2 stems = 60 trainings are fresh.
#
# What would count as success, written before the run:
#   * exclusion RISES at lam=0 relative to the asymmetric lam=0 run. That is the whole
#     point -- structure doing the job the penalty was added for.
#   * the optimal lambda MOVES DOWN. If the gate handles exclusion, the penalty should
#     be needed less. Asymmetric optima are lam=1 (Reg) and lam=16 (NCE).
#   * inclusion does NOT collapse. z_slow now trains on a weighted subset of horizons,
#     so this is the thing most likely to break. RankMe is the collapse guard.
set -u
cd "$(dirname "$0")"
export HGLP_NSEED=5 HGLP_GAP=0.03 HGLP_GATE_SYM=1

for stem in nce+ema l1+ema; do
  out=/tmp/rot_synth_sym_${stem%%+*}.log
  echo "=== $(date +%H:%M) start synth gate_sym ${stem} ==="
  HGLP_TAG=_sym python3 rotation.py synth "$stem" > "$out" 2>&1
  echo "    exit=$? -> $out"
done

echo "=== done $(date +%H:%M) ==="
ls -la ../../runs_v2/rotation_synth_*_sym.json 2>/dev/null
