#!/bin/bash
# Sleep-EDF patch-10, config D: dmax raised 64 -> 128, tau kept at 16.
#
# Why. Every Sleep-EDF run so far capped the horizon at 64 patches (6.4 s), which is
# a quarter of the 256-patch window. HAPT and PTB-XL both use half (128 of 239, 48 of
# 91). Nothing was ever written down to justify the quarter, and CLAUDE.md already
# lists the horizon set as underived, so 64 is an inherited value rather than a
# chosen one. This closes that hole.
#
# What to expect, computed before running. Deltas are drawn log-uniformly, so raising
# the ceiling moves the upper tail (95th percentile 56 -> 99 patches) but barely moves
# the bulk (median 20 -> 24). Against tau=16 the gate mean therefore goes 0.358 ->
# 0.307 -- z_mix receives slightly LESS gradient than config C, not more. So this is
# close to a repeat of C, whose result was lam=0 optimal in 4/4 and at or below the
# random split. The one thing that genuinely changes is how far z_slow is trained to
# reach: 5.6 s -> 9.9 s at the 95th percentile.
#
# The honest framing: this tests the ONE setting axis section 4.7 never varied. The
# prediction is that it reproduces C. If it does, the axis is closed; if it does not,
# that is the more interesting outcome.
#
# Config C runs first in the existing script and shares nothing here (dmax is part of
# the checkpoint key), so all 30 gated trainings are fresh. The 6 ungated controls
# ignore tau but NOT dmax, so they are fresh too.
set -u
cd "$(dirname "$0")"
export HGLP_NSEED=3
export HGLP_NPZ=../../data/sleepedf_p10.npz
export HGLP_W=8 HGLP_DMIN=8 HGLP_DMAX=128 HGLP_STEPS=5000

run () {                       # run <stem>
  local stem=$1
  local out=/tmp/rot_p10_D_${stem%%+*}.log
  echo "=== $(date +%H:%M) start D tau=16 dmax=128 ${stem} ==="
  HGLP_TAU=16 HGLP_TAG=_p10D \
    python3 rotation.py sleepedf "$stem" > "$out" 2>&1
  echo "    exit=$? -> $out"
}

run nce+ema
run l1+ema

echo "=== done $(date +%H:%M) ==="
ls -la ../../runs_v2/rotation_sleepedf_*_p10D.json 2>/dev/null
