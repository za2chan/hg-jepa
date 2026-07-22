# Horizon-Gated JEPA (HG-JEPA)

Label-free separation of **slow** and **fast** generative factors in time
series. A horizon-conditioned JEPA (causal encoder + multi-horizon predictor +
EMA target, in the CPC / I-JEPA / HEPA lineage) with one addition: a **gate**
that removes the fast latent block from the predictor at long horizons. Because
only slow information is useful for predicting the far future, slow factors
concentrate in the ungated block and fast factors are pushed to the gated
block — with no labels used in training.

This repo is a progress-report pilot: a controlled synthetic proof plus two
real-data reproductions (HAPT, PTB-XL) and one informative negative (XJTU-SY).

## Idea in one line

`L = Σ_Δ || P(z_slow, g(Δ)·z_fast, Δ) − z̄_{t+Δ} ||²  + λ·dcor`,
with a soft gate `g(Δ)=σ((τ−Δ)/w)` → for Δ≫τ the predictor sees only `z_slow`.
`τ = c·T_ac` is set from the signal's own autocorrelation time (no sampling
rate needed).

## Results (3 seeds)

| Data | slow-info in z_slow ↑ | fast-leak in z_slow ↓ | note |
|---|---|---|---|
| Synthetic (gate+dcor) | 0.98 | 0.22 / 0.15 | vs 0.88/0.77 ungated |
| HAPT (gate) | 0.91 | 0.31 | vs 0.76 ungated |
| PTB-XL (gate+dcor) | 0.97 | 0.40 | vs 0.66 ungated |
| XJTU-SY | — | — | no timescale gap → no separation (as predicted) |

`slow-info` / `fast-leak` = a block's probe score as a fraction of the full
embedding's. The gate on a raw-signal (next-token) head does nothing —
separation is specific to latent prediction.

## Layout

```
datagen.py         two-timescale synthetic generator (ground-truth factors)
train.py           synthetic HG-JEPA + ablations (gate/dcor/AR)
figures.py         main figure from runs_*/
screen.py          pre-training suitability screen (statistical checks)
screen_model.py    model-based screen (brief pretrain + long-horizon predictability)

xjtu_prep.py       XJTU-SY snapshot-sequence prep     -> data/xjtu.npz
xjtu_raw_prep.py   XJTU-SY raw amplitude-modulation prep -> data/xjtu_raw.npz
real_train.py      XJTU snapshot-level run
raw_am_train.py    XJTU raw-AM run (+ low-pass baseline)

hapt_prep.py / hapt_train.py     HAPT inertial (activity vs gait)
ptbxl_prep.py / ptbxl_train.py   PTB-XL ECG (diagnosis vs beat)

runs_*/            per-seed result JSONs (kept in repo)
report.tex         two-page progress report
```

## Reproduce

```bash
pip install -r requirements.txt

# synthetic (self-contained; no download)
python datagen.py                       # self-check
python train.py mode=nepa gate=1 dcor=1 seed=0
python figures.py

# suitability screen on prepared datasets
python screen_model.py
```

### Datasets (download separately, place under `data/`)

- **XJTU-SY** bearing run-to-failure (parquet). Set the path in `xjtu_prep.py`.
- **HAPT** (UCI Smartphone HAR + Postural Transitions): raw inertial signals.
- **PTB-XL** (PhysioNet): 100 Hz ECG records read via `wfdb`.

`data/` is gitignored. Prep scripts write `.npz` caches consumed by the trainers.

## Status / caveats

Progress-report pilot. Evaluations are in-distribution (matching the synthetic
protocol); subject-split and external baselines (TS2Vec, C-DSVAE) are planned.
Training is fully label-free; labels are used only to *measure* separation —
standard for disentanglement. The method needs a genuine slow/fast timescale
gap in the window, which is diagnosable before training.
