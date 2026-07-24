# HG-JEPA — Session Handoff

Progress-report research project. This file lets a fresh session resume without
re-deriving context. Read this + `report.tex` + skim the code, then continue.

## What this is

**Horizon-Gated JEPA (HG-JEPA):** label-free separation of slow vs fast
generative factors in time series. Backbone = horizon-conditioned JEPA (causal
encoder + multi-horizon predictor + EMA target; CPC/I-JEPA/HEPA lineage). Our
addition = a **gate** that removes the fast latent block from the predictor at
long horizons. Because only slow info helps predict the far future, slow factors
concentrate in the ungated block. NOT NEPA (NEPA is autoregressive next-latent;
ours is multi-horizon direct — the gate needs the horizon knob).

Deliverable: a 2-page LaTeX progress report (`report.tex`) for an advisor, due
originally 2026-07-24. Compiles on Overleaf (`report.tex` + `fig_main.pdf`); no
LaTeX in this env. Report language English, format LaTeX.

## Environment

- Dir: `/mnt/workspace/jaechan_lee/hgjepa` (git repo).
- Remote: `https://github.com/za2chan/hg-jepa.git` (branch `main`). Push needs a
  PAT the user provides interactively — do NOT commit tokens. Both tokens shared
  earlier in chat should be revoked by the user.
- GPU: H200, PyTorch present. `python3 <script>.py`. No `gh`, no LaTeX, no sudo.
- Data (gitignored, ~5.5G, already prepped in `data/`):
  - XJTU-SY at `/mnt/workspace/data/MFM_data/awesome_industrial_dataset/XJTU-SY Bearing Datasets/...` (parquet, 2ch vibration).
  - HAPT + PTB-XL downloaded under `data/`. Prep scripts write `.npz` caches.

## Current state (commit 5c7f233)

Experiments are leak-free: probes use GROUP splits (HAPT=subject, PTB-XL=patient,
XJTU=bearing) + disjoint windows; absolute scores + RankMe collapse monitor.

**Established results (all hold, 3 seeds):**
- Synthetic: gate+dcor separates — fast leak u 0.79→0.19, phase 0.73→0.27,
  regime kept 0.80. AR (raw-target) gate does nothing (H3). dcor-alone
  degenerates. dcor is data-dependent.
- HAPT (subject-split): gate cuts fast leak 0.56→0.17, activity F1 0.71 (full
  0.80) — holds on unseen subjects (domain robustness).
- PTB-XL (patient-split): weaker but consistent (leak 0.61→0.41 with dcor).
- XJTU: negative — no timescale gap (carrier & envelope T_ac ~1-2 patches);
  low-pass baseline can't recover fault envelope (R2 -0.19) = motivation confirm.
- **unmixing.py (#1):** no label-free SFA/PCA/ICA on the ungated embedding
  matches the gate (they trade regime for leak). Refutes "unmixing suffices."
- **mech.py (#5):** target masking doesn't help gate-alone (gate has no removal
  pressure — concede this). Width sweep: separation tracks the ABSOLUTE slow-
  block size, not total width — fixed 16-dim block separates even at dz=256
  (leak 0.08). Mechanism = gate ASSIGNS the slow factor to the block; narrow
  bottleneck + dcor EXCLUDE the fast factor; scales with width if block stays
  narrow.

## The reviewer critique driving all next work

A rigorous review flagged 10 issues (see below). Highest-value ones:
- #1 rotation symmetry / unmixing baseline — DONE (refuted).
- #5 no removal pressure + capacity + scaling — DONE (reframed honestly).
- #2 missing slowness/identifiability literature, zero theorems — TODO (writing).
- #3 model selection uses labels (Locatello 2019) — TODO (unsup criterion).
- #4 linear probe ≠ info; ratio hides collapse — PARTLY DONE (abs+RankMe);
  still need MLP probe + MINE + DCI/MIG.
- #7 PTB-XL is static/dynamic (=DSAE setting) — TODO (concede in writing).
- #8 contrastive rejection self-contradictory; F2 missing latent-contrastive
  cell — TODO (gated CPC-InfoNCE).
- #9 anomaly attribution confounds; bad benchmarks (SMAP/PSM) — TODO (redesign).
- #6 XJTU poster-child fails; naive low-pass strawman vs Hilbert envelope;
  within-window slowness boomerang — TODO.
- #10 reproducibility gaps, "guarantee" overclaim — PARTLY DONE.

## Next steps (ordered; see report for detail)

1. ~~Report writing~~ DONE (2026-07-24): unmixing table = Tab 1; H2 renamed
   "assignment asymmetry"; F4 = width sweep ("gate assigns, bottleneck
   excludes, scales if block stays narrow"); SMAP/PSM dropped. NOT compiled —
   check 2-page overflow on Overleaf, trim Planned work if needed.
2. ~~Latent-contrastive cell (#8)~~ DONE: train.py mode=cpc (InfoNCE vs EMA,
   in-batch negs). **CPC gate+dcor separates BEST: regime 0.81, leak 0.05/0.00**
   (z_fast keeps 0.85/0.95, no collapse). Gate-alone leaks (0.46/0.81) = F3
   consistent. F2 reframed: latent-target family, not JEPA-regression-specific.
3. ~~Competitor (#2 partial)~~ DONE: slowvae.py (arch-matched SlowVAE, Laplace
   transition prior). Loses the slow factor (regime 0.63 slow / 0.54 FULL) at
   leak 0.27/0.29 — recon is local, regime needs long integration. Still no
   theorems of our own (#2 writing/lit part remains).
4. **Metrics (#4):** DCI/MIG DONE (metrics.py, emb_*.npz): MIG ~0 for all
   (block codes); DCI ranks SlowVAE (0.66) > ours (0.39) despite SlowVAE not
   decoding regime — reported honestly as axis-metric failure (Locatello).
   Remaining: MLP probe + MINE for "displaced".
5. ~~Anomaly attribution~~ DONE, **NEGATIVE** (anomaly.py, model_*.pt saved by
   train.py): point detection AUROC 0.93-0.97 (short residual), contextual at
   CHANCE (~0.50) in both designs (ctx freq shift 1.08 on-manifold and 1.15
   off-manifold; long residual full-dim AND slow-dim only). Diagnosis: encoder
   projects unseen dynamics onto normal slow codes — no novelty response.
   Reported honestly in report ("Anomaly attribution: a negative pilot").
   Next fix would be novelty-sensitive machinery (density model on z_slow or
   training-time drift exposure), not more gating.
   MLP/MINE (#4 remainder) also DONE (nonlinear.py): linear probes overstate
   exclusion — MLP leak from z_slow u 0.45/phase 0.78 (vs 0.19/0.27 linear;
   ungated 0.84/0.96); MINE I(z_slow;u) 0.95 vs 1.21 nats ungated. Report
   reframed: "linear-subspace separation + partial information reduction".
6. ~~XJTU redesign~~ DONE: hilbert_baseline.py + life probe in raw_am_train.py.
   FOUND CIRCULARITY: env label IS |hilbert| (xjtu_raw_prep) → envelope-proxy
   comparison retracted in report. Non-circular target = life (RUL), held-out
   bearings: Hilbert feats R2 -1.08, lowpass -0.11, learned z_slow +0.18
   (z_full 0.18, z_fast 0.09). Weak for everyone; z_slow carries what exists.
7. ~~Literature + honesty~~ DONE in report: Positioning paragraph (SFA /
   predictive-info + past-future IB / TCL + SlowVAE + Locatello; "no theorem
   yet" stated as the main gap), expanded Limitations (2-scale assumption,
   τ=c·T_ac ACF caveat, patch/sampling-rate bound).

## Repo map

- `datagen.py` synthetic (hard regime: ±5% freq, equal amplitude). `train.py`
  synthetic HG-JEPA + ablations → `runs/`. `figures.py` → `fig_main.pdf` (reads
  `runs/`, `runs_hapt/`). `run_all.sh` synthetic sweep; `run_experiments.sh`
  full re-run.
- `hapt_prep/train.py`, `ptbxl_prep/train.py` real data (group-split, RankMe).
- `xjtu_prep.py`+`real_train.py` (snapshot, bearing held-out, negative);
  `xjtu_raw_prep.py`+`raw_am_train.py` (raw AM + low-pass baseline).
- `unmixing.py` (#1), `mech.py`+`mech_sweep.sh` (#5).
- `screen.py`/`screen_model.py`: pre-training suitability screens — a robust
  cheap universal screen was NOT found (slow factor hides in any descriptor;
  circular). Model-based screen catches within-window-varying factors only.
- Probe protocol everywhere: fit on train group, test on held-out group,
  absolute scores. Never random-split overlapping windows (that was the leak).

## Gotchas

- Report figure uses ABSOLUTE scores (not ratios) after reviewer #4; z_slow can
  beat z_full on the slow task (selective-use payoff, not a bug).
- τ sweep is weak under leak-free (report honestly). Dim/width sweep is the
  strong mechanism evidence.
- dcor helps synthetic + PTB-XL, HURTS HAPT — data-dependent, say so.
