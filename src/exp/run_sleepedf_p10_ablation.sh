#!/bin/bash
# 2x2 ablation on Sleep-EDF at patch 10, configs A and C.
#
# This is the counterpart to run_sleepedf_p10.sh (post-hoc rotation). It fills the
# row that lines up directly with the paper's Table 2, whose claim is that the gate
# and the cross-covariance penalty are complementary and that their combination is
# first in all six settings. The rotation runs already showed lambda=0 winning on
# both stems and both tau settings, so this is the direct test of that claim in the
# paper's own cell layout: g0_x0_noBlockLN | g1_x0_noBlockLN / g0_x1_LN | g1_x1_LN.
#
# main22 is the mode that uses the corrected mechanism-free control (no per-block
# LayerNorm); the older `twosided_sleepedf.json` used the pre-2026-08-07 cells and
# is not comparable with the paper's table.
#
# Config A is run first so config C reuses A's cached ungated encoders (the gate is
# off there, which makes tau inert and the checkpoint shared).
set -u
cd "$(dirname "$0")"
export HGLP_NSEED=3
export HGLP_NPZ=../../data/sleepedf_p10.npz
export HGLP_W=8 HGLP_DMIN=8 HGLP_DMAX=64 HGLP_STEPS=5000

run () {                       # run <tag> <tau>
  local tag=$1 tau=$2
  local out=/tmp/abl_p10_${tag}.log
  echo "=== $(date +%H:%M) start ablation ${tag} tau=${tau} ==="
  HGLP_TAU=$tau HGLP_LAM_TAG=_p10${tag} \
    python3 twosided.py sleepedf main22 > "$out" 2>&1
  echo "    exit=$? -> $out"
}

run A 9
run C 16

echo "=== ablation done $(date +%H:%M) ==="
ls -la ../../runs_v2/twosided_main22_sleepedf_p10*.json 2>/dev/null
