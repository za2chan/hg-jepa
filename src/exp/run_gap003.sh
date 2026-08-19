#!/usr/bin/env bash
# Everything that was measured at the +-1% synthetic setting, re-measured at +-3%.
#
# Why: at +-1% the slow factor is at chance (inclusion 0.352 vs chance 0.333), so
# SEP's inclusion term is uninterpretable there; at +-5% a training-free FFT
# classifier already reaches 0.784, so separation there proves little. difficulty
# .json puts the FFT classifier at 0.566 for +-3%, i.e. between the two failure
# modes. That is the setting the paper should report.
#
# 5 seeds everywhere (NCE's across-seed sd is 2.4x L1's at 3 seeds -- too wide).
# Runs 5 at a time; each line is one job, so a crash loses one job, not the batch.
set -u
cd "$(dirname "$0")"
export HGLP_GAP=0.03 HGLP_NSEED=5
LOG=/tmp/gap003; mkdir -p $LOG

jobs_list=(
  "python3 twosided.py synth                > $LOG/twosided.log 2>&1"
  "python3 twosided.py synth ln2x2          > $LOG/ln2x2.log 2>&1"
  "python3 rotation.py synth l1+ema         > $LOG/rot_l1.log 2>&1"
  "python3 rotation.py synth nce+ema        > $LOG/rot_nce.log 2>&1"
  "python3 sweeps.py lam synth 5            > $LOG/sw_lam.log 2>&1"
  "python3 sweeps.py tau synth 5            > $LOG/sw_tau.log 2>&1"
  "python3 sweeps.py dslow synth 5          > $LOG/sw_dslow.log 2>&1"
  "python3 label_x_perturb.py synth 2500 5 3 > $LOG/labeleff.log 2>&1"
  "HGLP_VARIANT='{\"target_mode\":\"cumulative\"}' HGLP_VARIANT_TAG=cumtarget python3 twosided.py synth variant > $LOG/v_cum.log 2>&1"
  "HGLP_VARIANT='{\"target_space\":\"raw\"}'       HGLP_VARIANT_TAG=rawtarget python3 twosided.py synth variant > $LOG/v_raw.log 2>&1"
  "HGLP_VARIANT='{\"gate_hard\":true}'            HGLP_VARIANT_TAG=gatehard  python3 twosided.py synth variant > $LOG/v_hard.log 2>&1"
)

printf '%s\n' "${jobs_list[@]}" | xargs -P 5 -I{} bash -c '{}'
echo "ALL DONE $(TZ=America/Toronto date '+%F %H:%M %Z')" | tee $LOG/DONE
