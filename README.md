# Horizon-Gated JEPA (HG-JEPA)

Label-free separation of **slow** and **fast** generative factors in time
series. A horizon-conditioned latent-predictive learner (causal encoder +
multi-horizon predictor + EMA target, in the CPC / I-JEPA / HEPA lineage) with
one addition: a **gate** that removes the fast latent block from the predictor
at long horizons. Only slow information helps predict the far future, so the
gate *assigns* slow factors to the ungated block (provably — Prop. 1 in the
report); a narrow bottleneck + decorrelation penalty handle *exclusion* of
fast factors, which the objective alone cannot enforce (Prop. 2).

```
L = Σ_Δ || P(z_slow, g(Δ)·z_fast, Δ) − z̄_{t+Δ} ||²  + λ·dcor,
g(Δ) = σ((τ−Δ)/w),   τ = c·T_ac  (from the signal's own autocorrelation time)
```

## Key results (3 seeds, absolute scores, leak-free group-split probes)

| Claim | Evidence | Numbers |
|---|---|---|
| Gate+dcor separates (synthetic) | z_slow keeps regime, drops fast | regime 0.80 (full 0.69); u/phase leak 0.19/0.27 vs 0.79/0.73 ungated |
| Works across latent objectives | CPC-InfoNCE cell separates best | regime 0.81, leak 0.05/0.00; raw-AR target: no separation |
| Post-hoc unmixing insufficient | SFA/ICA/PCA on ungated embedding | each trades regime for leak; only training-time gating gets both |
| Exclusion rides on block size | width sweep, fixed 16-dim block | leak 0.06–0.08 at d_z=128/256; proportional block: leak ~0.6 |
| Exclusion is linear-subspace | MLP probe + MINE | MLP leak 0.45 (vs 0.19 linear); MINE 0.95 vs 1.21 nats ungated |
| Real data | HAPT (subject-split), PTB-XL (patient-split) | HAPT leak 0.56→0.17 at F1 0.71; PTB-XL 0.61→0.41 |
| Predicted negative | XJTU-SY: no timescale gap | gate correctly does nothing; z_slow still best life probe (+0.18) |
| Anomaly attribution | typed injection pilot | point AUROC 0.93–0.97; contextual at chance — honest negative |
| Label-free tuning | retrospective grid validation | Spearman 0.68; catastrophe filter yes, fine tuner no |

## Experiment ↔ script ↔ output map

| Experiment | Script(s) | Results |
|---|---|---|
| Synthetic data + self-check | `datagen.py` | — |
| Synthetic HG-JEPA / ablations / AR / CPC | `train.py` (`mode=nepa|ar|cpc`, `gate=`, `dcor=`, `tau=`, `dslow=`, `lam=`) | `runs/<tag>.json`, `runs/emb_<tag>.npz`, `runs/model_<tag>.pt` |
| Unmixing control (SFA/ICA/PCA) | `unmixing.py` | `runs/unmixing_s*.json` |
| Width/masking mechanism sweep | `mech.py`, `mech_sweep.sh` | `runs/mech_*.json` |
| SlowVAE competitor | `slowvae.py` | `runs/slowvae_s*.json`, `runs/emb_slowvae_*.npz` |
| DCI / MIG metrics | `metrics.py <emb.npz...>` | stdout |
| MLP probe + MINE (nonlinear leak) | `nonlinear.py tag=<tag>` | `runs/nl_<tag>.json` |
| Anomaly injection + attribution | `anomaly.py tag=<tag> ctx=<mult>` | `runs/anom_*.json` |
| Label-free selection validation | `select.py` (needs grid runs) | stdout |
| Pre-training suitability screens | `screen.py`, `screen_model.py` | stdout |
| HAPT (subject-split) | `hapt_prep.py`, `hapt_train.py` | `runs_hapt/*.json` |
| PTB-XL (patient-split) | `ptbxl_prep.py`, `ptbxl_train.py` | `runs_ptbxl/*.json` |
| XJTU snapshot (bearing held-out) | `xjtu_prep.py`, `real_train.py` | `runs_real/*.json` |
| XJTU raw AM + life probe | `xjtu_raw_prep.py`, `raw_am_train.py` | `runs_am/*.json` |
| XJTU Hilbert/low-pass baselines | `hilbert_baseline.py` | `runs_am/hilbert_baseline.json` |
| Main figure | `figures.py` | `fig_main.pdf` |
| Full sweeps | `run_all.sh`, `run_experiments.sh` | `runs*/` |

Documents: `report.tex` (progress report, compile on Overleaf with
`fig_main.pdf`), `plan.tex` (advisor briefing: evidence status + ICLR plan),
`HANDOFF.md` (session resume notes).

## Reproduce

```bash
pip install -r requirements.txt
python datagen.py                              # data self-check
python train.py mode=nepa gate=1 dcor=1 seed=0 # ours
python train.py mode=cpc  gate=1 dcor=1 seed=0 # contrastive cell
python figures.py
```

Real datasets (gitignored, place under `data/`): **XJTU-SY** (parquet; path in
`xjtu_prep.py`), **HAPT** (UCI), **PTB-XL** (PhysioNet, via `wfdb`). Prep
scripts write `.npz` caches consumed by the trainers.

## Protocol (non-negotiable)

Probes fit on training groups, evaluated on held-out groups (HAPT=subject,
PTB-XL=patient, XJTU=bearing; synthetic=contiguous time split) over disjoint
windows. Absolute scores only (ratios once hid a collapse) + RankMe collapse
monitor. All variants share one backbone (~0.5M params) and budget; n=3 seeds.

## Caveats

Exclusion is soft and linear-subspace-level (see Prop. 2 / MLP results); dcor
helps synthetic+PTB-XL, hurts HAPT (data-dependent); the method needs a genuine
within-window timescale gap (diagnosable pre-training — XJTU correctly refuses);
fine hyperparameter selection still needs labels (unsup criterion = catastrophe
filter only). Training itself is fully label-free.
