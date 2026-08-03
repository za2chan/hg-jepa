#!/bin/bash
# Day-1 overnight matrix (CHECKLIST_v3 Day 1). Sequential on the H200;
# ~36 s per run x ~45 runs ~= 30 min. Skips cells whose JSON already exists.
# Old cpc_* runs are the nce+ema control cells — never re-run here (hard rule 4).
set -e

run() {  # run <tagfile> <args...>
  local tag=$1; shift
  if [ -f "runs/${tag}.json" ]; then echo "skip ${tag}"; return; fi
  python3 train.py "$@"
}

for s in 0 1 2; do
  # full Reg re-run under D3 (seeds resample data) + AR raw-target controls
  run "reg-ema_g0_x0_s${s}_tau16_ds16_lam4_vf1" loss=reg target=ema gate=0 xcov=0 seed=$s
  run "reg-ema_g1_x0_s${s}_tau16_ds16_lam4_vf1" loss=reg target=ema gate=1 xcov=0 seed=$s
  run "reg-ema_g1_x1_s${s}_tau16_ds16_lam4_vf1" loss=reg target=ema gate=1 xcov=1 seed=$s
  run "ar_g0_x0_s${s}_tau16_ds16_lam4_vf1"      loss=ar gate=0 xcov=0 seed=$s
  run "ar_g1_x0_s${s}_tau16_ds16_lam4_vf1"      loss=ar gate=1 xcov=0 seed=$s

  # new NCE(online) cells (D2: both-sided grads, no EMA, no stop-grad)
  run "nce-online_g1_x1_s${s}_tau16_ds16_lam4_vf1" loss=nce target=online gate=1 xcov=1 seed=$s
  run "nce-online_g1_x0_s${s}_tau16_ds16_lam4_vf1" loss=nce target=online gate=1 xcov=0 seed=$s
  run "nce-online_g0_x1_s${s}_tau16_ds16_lam4_vf1" loss=nce target=online gate=0 xcov=1 seed=$s
  run "nce-online_g0_x0_s${s}_tau16_ds16_lam4_vf1" loss=nce target=online gate=0 xcov=0 seed=$s

  # vfloor=0 ablation, both stems (D7 rule applies on collection)
  run "reg-ema_g1_x1_s${s}_tau16_ds16_lam4_vf0"    loss=reg target=ema    gate=1 xcov=1 vfloor=0 seed=$s
  run "nce-online_g1_x1_s${s}_tau16_ds16_lam4_vf0" loss=nce target=online gate=1 xcov=1 vfloor=0 seed=$s

  # P3-1 variance-only regime, both stems
  run "reg-ema_g1_x1_s${s}_tau16_ds16_lam4_vf1_vm"    loss=reg target=ema    gate=1 xcov=1 regime_mode=variance seed=$s
  run "nce-online_g1_x1_s${s}_tau16_ds16_lam4_vf1_vm" loss=nce target=online gate=1 xcov=1 regime_mode=variance seed=$s

  # NCE d_slow sweep (returned to sprint; free filler)
  run "nce-online_g1_x1_s${s}_tau16_ds8_lam4_vf1"  loss=nce target=online gate=1 xcov=1 dslow=8  seed=$s
  run "nce-online_g1_x1_s${s}_tau16_ds32_lam4_vf1" loss=nce target=online gate=1 xcov=1 dslow=32 seed=$s
  # HAPT tau-sensitivity arm (claim: broad plateau, auto inside it — not optimality)
  # grid brackets Delta_max=32p (0.81x auto) and the ~1s gait lifetime (0.25x);
  # 1.0x = "gate effectively never closes" control
  for m in 0.25 0.4 0.55 0.7 0.9 1.0; do
    if [ ! -f "runs_hapt/hapt_nepa_g1_d1_s${s}_tauauto${m}x.json" ]; then
      python3 hapt_train.py mode=nepa gate=1 dcor=1 tau=auto taumult=$m seed=$s
    else echo "skip hapt tauauto${m}x s${s}"; fi
  done
done
echo "matrix done"
