# HGLP — refactor & experiment checklist (v2, code-verified)

Near-term target: **Aug 7, 23:59 — IEEE-format 8pp course paper (Responsible-AI
course)**; the sprint is governed by `CHECKLIST_v3.md` (v3 wins on conflict).
This file = big-picture reference + post-deadline backlog for the workshop
version. Repo = `za2chan/hg-jepa` @ `b4e6c70`.
Every P0 claim below was verified against the actual code on 2026-08-02.
Do phases in order: **P0 → P1 → P2 → P3 → P4**.

Legend: `[ ]` todo · **AC** = acceptance criteria · ⛔ **STOP** = stop-the-line:
halt, report, do not proceed to dependent items.

**Decision register (LOCKED by the user — do not revisit; details in v3 §A):**
D1 τ anchor = **min** T_ac across derived series · D2 NCE-online = **both-sided
gradients, no EMA, no stop-grad** · D3 **`seed=seed`** + full synthetic re-run
(folded into the matrix run) · D4 Part-2 stem by rule (dominant on both
slow-kept and leak; tie → Reg) · D5 cut order approved · D6 deadline above ·
D7 vfloor by rule (RankMe collapse at `vfloor=0` → keep as third term; else
remove). Hardware: dedicated **H200** — compute-bound cuts are void;
implementation time is the binding constraint.

---

## P0 — paper↔code integrity (mismatches, not features)

### [ ] P0-1. Wire τ to the data (`train.py`, `screen.py` → new `tac.py`)
> **2026-08-02 amendment:** the 6.25-patch bar below was a spec error (it is
> unobservable u's own lifetime; the energy series tracks u², which forgets
> ×2 faster). D1-as-amended (CHECKLIST_v3 §A / `docs/tau.md`) supersedes this
> section: u-scale conversion (energy ×2), c = ln(1/eps), eps = 0.05.
`tau` is a hardcoded CLI default of 16 patches; never derived from the signal.
The paper's headline "τ = c·T_ac, label-free" currently has no code behind it.

- Extract `acf_decay` / `local_descriptors` from `screen.py` into `tac.py`:
  `estimate_tac(x_or_windows, patch=P) -> {"raw": .., "energy": .., "envelope": ..,
  "zcr": .., "chosen": .., "rule": str}` — T_ac in **patch units**.
- **DECIDED (D1): `chosen` = the minimum T_ac across derived series** — the
  fastest-forgetting series estimates the fast lifetime. τ must anchor to the
  *fast factor's lifetime*. Code fact: on synthetic, ground truth is
  `U_TAU=50 steps / P=8 = 6.25 patches`, and `16 / 6.25 = 2.56 ≈ c=2.5` — the
  hardcoded 16 is retroactively consistent **iff** the estimator lands near
  6.25. Raw-sample ACF of `x` reflects carrier decorrelation, not `u`'s
  lifetime; descriptor/envelope series may reflect slow dynamics instead.
  Document the rule + rationale in `docs/tau.md`.
- Add `tau=auto` to all four trainers; compute from the *training split only*;
  `c` a CLI arg (default 2.5). Log `{tau_source, tac, c}` into run JSON.
  Keep manual `tau=` for the Appendix sweep.

**AC:** `train.py mode=reg gate=1 xcov=1 tau=auto seed=0` runs; on synthetic the
chosen T_ac ≈ 6.25 patches (ground-truth anchored — sharper than "within 2× of
16"). If it lands far off, that is a finding: record it, do **not** silently
retune `c`.
**Downstream:** if derived τ ≈ 16, the existing tau∈{4,64} sweep runs remain
valid as the c-sweep. If not, F4's tuning claim is unsupported → re-run sweep.

### [ ] P0-2. Surface the third loss term (`train.py`, `hapt_train.py`, ...)
`F.relu(1.0 - z.std(0)).mean()` — a VICReg-style variance floor — is applied in
**every** mode (coefficient 1.0, explicit in `hapt_train.py` line ~114) and is
absent from the paper's loss equation.

- Make it switchable: `vfloor=1|0`, named constant. Record in run tag/JSON.
- Ablation: `vfloor=0`, gate+xcov, 3 seeds — know what it does before claiming
  which mechanism prevents collapse.
- **DECIDED (D7, rule):** RankMe collapse at `vfloor=0` → promote the term into
  the paper's loss equation; no collapse → default `vfloor=0` and remove it.
  Apply mechanically once the ablation lands. Also run `git log -S "relu(1.0"`
  to record when/why the term entered.
- **Paper prose fix this forces:** F3's "dcor-alone degenerates by *emptying*
  the slow block" is inaccurate while the floor enforces std ≥ 1 — the block is
  not emptied but filled with regime-irrelevant variance. Reword.

**AC:** code loss == paper loss (or the paper gains the term); ablation JSONs
in `runs/`.

### [ ] P0-3. Define the two stems cleanly (`train.py`)
Code fact: `mode=cpc` = InfoNCE **+ EMA target + vfloor** — the Reg stem with a
different distance, not an independent stem.

- Rename modes `nepa→reg`, `cpc→nce` (old names as deprecated aliases one
  commit). Add `target=ema|online` orthogonal to `loss=reg|nce`.
- **DECIDED (D2):** `target=online` = **both-sided gradients (SimCLR-style),
  no EMA, no stop-grad** — the simplest form. If training is unstable: ⛔ stop
  and report; no autonomous fallback to stop-grad.
- Canonical stems: HGLP-Reg = `reg+ema` (EMA asymmetry handles collapse);
  HGLP-NCE = `nce+online` (negatives handle collapse).
- **Windfall (code-verified):** `runs/` already holds `cpc_g0_d0`, `cpc_g1_d0`,
  `cpc_g1_d1` × 3 seeds. After the redefinition these become exactly the
  `loss=nce target=ema` **control** cells P0-3 asks for — do not delete, do not
  re-run; relabel via the tag migrator (P1-1). Only `nce+online` cells and the
  missing `nce xcov-only` are new compute.

**AC:** Method loss-comparison table fills from run JSONs; per-stem `no
collapse` claims backed by RankMe **with vfloor=0 controls**.

### [ ] P0-4. Seeds must vary the data — decide & document (`train.py`)
Code fact: `make_dataset(1_000_000, seed=0)` with an existing loud comment
("data fixed across seeds by design; seeds vary only init + sampling order").
So option (b) of the original item is half-done; the missing piece is the
**paper protocol sentence** stating error bars reflect training stochasticity,
not data resampling.

- **DECIDED (D3): (a)** — `make_dataset(..., seed=seed)`; every synthetic
  error bar regenerates. Fold the re-run into the P4-1/overnight matrix so
  nothing trains twice; the protocol sentence now states data-resampling seeds.

**AC:** protocol paragraph and code agree; re-run scope written down.

### [ ] P0-5. Screen must reject coherent-periodic data (`screen.py`)
Code facts: no oscillation-persistence check exists; C2 *passes strongly* for
periodic signals (periodic is predictable). **And a subtler trap:** `screen()`
collapses windows to patch-means (`W.mean(-1)`) — an implicit 8-sample low-pass
that suppresses short-period components (the synthetic carrier, period ~10
steps, mostly cancels). A Fisher g-test on patch-means would **miss**
short-period periodic confounds entirely.

- `check_coherent_periodic(series)`: (a) ACF-envelope decay test; (b) Fisher's
  g-test on the periodogram of the **sample-level** signal (pre-collapse) *and*
  descriptor series.
- Wire as **Check 1**, before T_ac/gap; hard reject `coherent_periodic`.
- Renumber C1/C2/C3 → Check-2 components.

**AC (regression):** HAPT / synthetic / PTB-XL / XJTU verdicts keep their
SUITABLE↔UNSUITABLE side vs today's `screen_results.json` (PTB-XL's tier
refinement in P0-6 is the allowed exception). Injected-periodic control is
REJECTED with reason `coherent_periodic`.
⛔ **STOP if synthetic fails Check 1.** The carrier is `sin(φ)` with ±5% FM;
whether OU phase-diffusion decays the ACF envelope fast enough is a threshold
calibration question. If the screen rejects its own mechanism testbed, fix the
calibration before anything downstream — all screen verdicts are untrustworthy
until this passes.

### [ ] P0-6. Window-adequacy must use Δ_max, not L (`screen.py`)
Code fact: `c3 = tau_fast < tau_slow < L` (L=256) but the load-bearing quantity
is Δ_max=128: slow must vary on a scale comparable to the longest trained
horizon or the method degenerates to static separation.

- Compare against Δ_max; verdict tiers `SUITABLE (slow)` /
  `SUITABLE (static-only)` / `UNSUITABLE`. PTB-XL landing in `static-only` is
  the correct answer, not a failure.
- **Reuse `screen_model.py`** — it already implements far-future embedding
  predictability (SHORT=2, LONG=64) and is exactly the label-free machinery for
  the upper-bound diagnostic (does predictability keep rising toward Δ_max, or
  saturate early = dwell ≫ Δ_max). Extend its horizon list rather than building
  new code.

**AC:** screen distinguishes HAPT (slow) from PTB-XL (static-only).

---

## P1 — naming & framing refactor (single pass, right after P0)

### [ ] P1-1. Project rename HG-JEPA → HGLP
- README/HANDOFF/`report*.tex`/`plan.tex`: "Horizon-Gated Latent Prediction
  (HGLP)", stems **HGLP-Reg** / **HGLP-NCE**. Keep `JEPA`/`contrastive` as
  searchable words in subtitle/abstract only.
- Run-tag scheme `{stem}_{target}_g{}_x{}_vf{}_s{}_tau{}_ds{}_lam{}` +
  `tools/migrate_tags.py` so `figures*.py` globs keep working (they read
  `runs/*.json` — the automation exists; **extend it, don't fork it**, and add
  the missing tables so every printed number regenerates from JSONs; the paper
  has a prior history of figure↔text number drift).

### [ ] P1-2. `Prop. 1 / Prop. 2` → `Gating / Exclusion`
Docstrings, README, HANDOFF, `report*.tex`, and the method-figure labels
(`fig_method_pptx.py` regenerates the figure — edit source, not the PDF).

### [ ] P1-3. `L_dcor` → `L_xcov`
Name collides with Székely's distance correlation; implementation is squared
cross-covariance. CLI `dcor=`→`xcov=` (alias one commit).

### [ ] P1-4. Fix the HEPA attribution in all prose
Verified against the HEPA paper: interval-summary targets over `(t, t+Δ]`
(bidirectional + attention pooling), weight-shared jointly-trained target (no
EMA/stop-grad), SIGReg, L1. Ours: **point target at t+Δ**, EMA, L2. State the
contrast; never "as in HEPA". **Scope (user decision):** the four-way contrast
lives in Related Works as one–two sentences; Method keeps exactly **one**
sentence on target shape (point vs interval) — that one is a premise of the
gate's logic, not a comparison. Files: README, HANDOFF, `report*.tex`,
`plan.tex`.

---

## P2 — screen hardening + early risk kill

### [ ] P2-1. Screen as callable API + CLI
`screen(x) -> {verdict, reason, tac_per_series, checks{...}}`; usable from
trainers/notebooks, not only `__main__`.

### [ ] P2-2. Derived-series coverage
Add **envelope** (`|hilbert(x)|`) to {raw, energy}; keep the set at three
(+ ZCR as frequency proxy). Do not expand; document the necessary-condition
caveat (slow structure invisible in all three is missed).

### [ ] P2-3. Normalization audit → `docs/normalization.md`
Code facts to record: `train.py` feeds **raw, unstandardized** `x` (good — the
amplitude cue reaches the encoder), and applies `F.layer_norm` over the D_Z
**embedding dims** per position — this erases magnitude-coding of variance but
not direction-coding; it constrains rather than destroys. Check the real-data
preps (`hapt_prep.py`, `ptbxl_prep.py`, `xjtu*_prep.py`) for input-side
standardization and record each.
**AC:** the note exists **and** a `layernorm=0` flag is available; P3-1 runs
both settings. P3-1 does not start before this lands.

### [ ] P2-4. ⚠️ Classical risk baseline — run EARLY (moved up from P4)
FM demodulation (instantaneous frequency via `scipy.signal.hilbert`) → moving
average (~hundreds of steps) → 3-state HMM → regime accuracy on synthetic.
Code fact: `hilbert_baseline.py` exists but is the **XJTU envelope-transfer**
baseline, not this; reuse its Hilbert scaffolding.
**Why early:** if this 30-line pipeline matches the gate, the "no short-window
statistic reveals the regime" defense dies and F1's narrative must be rewritten
— that rewrite needs calendar time, so know the answer before P3/P4, not at
submission. Losing is survivable (reframe: "a classical tool that knows the
generative form wins; the gate needs no such knowledge and transfers to real
data"); not knowing is not.

---

## P3 — new experiment modules

### [ ] P3-1. Variance-only regime (`datagen.py`)
`regime_mode=freq|variance`: `variance` leaves carrier freq and mean untouched,
modulates **OU amplitude** only.
- ⚠️ **Code trap:** `datagen.py`'s `__main__` self-test asserts
  `max(var_by_regime)/min < 1.2` ("hard mode") — variance mode violates this
  **by design**. Gate the assertion on `regime_mode` or the self-test fails.
- Run gate+xcov for **both stems**, 3 seeds, layernorm on/off (P2-3).
- Prediction on record before running: Reg (conditional-mean estimator) may
  struggle; NCE may not.
**AC:** 2 stems × 3 seeds in `runs/`; one result sentence written into the
paper — separation confirmed or not + whether the stems differ. (Memory note
exists for this; the theory footnote's fate — appendix or cut — is decided by
the result.)

### [ ] P3-2. Coherent-periodic confound (`datagen.py`)
`periodic_amp`, `periodic_coh` injectable component. Uses: (a) P0-5 acceptance
control; (b) optional coherence sweep — as coherence time crosses Δ_max the
phase factor should migrate z_fast → z_slow.

### [ ] P3-3. Interval-summary target ablation (`train.py`)
`target_shape=point|interval` (`interval` = bidirectional encode `(t, t+Δ]` +
attention pooling = the HEPA target). Reg stem only.
**AC:** one Part-1 row; separation degrades measurably under `interval`, or the
"gate requires a point target" argument gets rewritten.

### [ ] P3-4. Shift robustness (`shift_eval.py`) — **highest priority in P3**
Train unchanged; eval-time perturb **fast only** (OU lifetime 50→{20,150}, OU
scale ×{0.5,2}, noise level); regime process fixed. Probes fit on clean,
evaluated on shifted, z_slow vs z_full.
**AC:** the degradation-gap figure. Single load-bearing result of Part 2 — the
abstract has a placeholder sentence waiting for it.

### [ ] P3-5. Label efficiency (`label_eff.py`)
{1%, 10%, 100%} × {z_slow, z_full}, synthetic + HAPT. Ridge/logistic λ tuned on
a validation split **for both** input sets.

### [ ] P3-6. HI quality (`hi_metrics.py`) — optional, thesis bridge
z_slow trajectory (1st PC) on run-to-failure: monotonicity / trendability /
prognosability (Coble & Hines) vs the z_full trajectory. Skip if clock is tight.

---

## P4 — matrix runners, then figures

### [ ] P4-1. Mechanism-matrix cells (`run_all.sh` → `run_matrix.sh`)

| experiment | Reg | NCE (`nce+online`) | NCE-ema control |
|---|---|---|---|
| main (gate+xcov) | have | **todo** | have (old `cpc_g1_d1` ×3) |
| gate-only | have | **todo — highest value** | have (`cpc_g1_d0` ×3) |
| xcov-only | have | **todo — highest value** | todo (`cpc_g0_d1` never run) |
| no-gate control | have | todo | have (`cpc_g0_d0` ×3) |
| AR raw + gate | have | n/a | n/a |
| interval target (P3-3) | todo | n/a | n/a |
| d_slow sweep (`mech.py`) | have | todo (first to cut) | — |
| variance-only (P3-1) | todo | todo | — |
| coherent-periodic (P3-2) | todo | n/a | — |
| vfloor=0 (P0-2) | todo | todo | — |

The two "highest value" cells decide whether "structure assigns, penalty
purifies" is a property of the mechanism or of the Reg loss — **run them before
choosing which stem carries Part 2.** **DECIDED (D4, rule):** the stem that
dominates on *both* slow-kept and leak carries Part 2; if the metrics disagree
on the winner → keep Reg (avoids the real-data re-run); apply mechanically.
If the stems disagree (e.g. negatives'
uniformity substitutes for xcov in NCE), that is a finding, not a bug; budget
narrative-rewrite time for F3.
**If Part 2's stem = NCE:** HAPT/PTB-XL real-data training must be re-run once
under NCE (all existing real-data results are Reg).

### [ ] P4-2. External SSL baseline
TS2Vec (public code), same probe protocol; fairness twin: run the existing
`unmixing.py` (SFA/ICA/PCA) on the TS2Vec embedding too, so "best 16-dim
subspace of TS2Vec" is compared, not just the full embedding.
(The classical baseline moved to P2-4.)

### [ ] P4-3. Appendix sweeps
- τ/c sweep: **code fact — `runs/` already has tau∈{4,64} (s0,s1) + tau16
  (s0–2).** Add the missing seed for the extremes; validity conditional on
  P0-1's derived τ ≈ 16 (else full re-run).
- soft vs hard gate (`w→0`) + width `w` sweep.
- penalty variants: xcov vs HSIC.
- τ-estimator variants: raw vs envelope T_ac, biexponential ACF fit, and the
  model-based saturation point (`screen_model.py`) — sampling-rate units note
  (physical time vs patches) absorbed here.
- **Model-based τ estimation (added 2026-08-02, workshop version):**
  assumption-free alternative — read τ off the model's own horizon-wise error
  curve at the point where the fast block stops contributing; compare against
  the eps-derived τ of D1-as-amended (`docs/tau.md`). Not for the sprint.
- **Re-run `select.py` on the refreshed matrix** — the unsupervised
  model-selection retrospective already exists as code (rankme per block,
  cross-corr, slowness gap, Spearman vs labeled); it directly feeds the
  Limitations "model selection" paragraph and nobody should rebuild it.

### [ ] P4-4. Figures
- Method figure: loss box → abstract `ℓ(·, z̄)` with Reg/NCE branches (edit
  `fig_method_pptx.py`, regenerate; also fixes P1-2 labels).
- Concept figure (component traces + persistence axis): add generator as
  `fig_concept.py` (exists outside the repo — port it in). Apply the agreed
  fixes: split our capsule at the coherent-periodic point (no ✕-over-covered
  look), add per-method "what fixes the factor's location" labels, ordinal-axis
  caption sentence.
- Screen flowchart: serial Check 1 → Check 2; reject boxes `coherent periodic`
  and `no persistent component (e.g. XJTU-SY)`.
- Part-1/2 result figures after P4-1/P3-4; **every number via `figures_paper.py`
  from run JSONs — no hand transcription.**

---

## Paper-side items (no code) — targets: `report*.tex`, `plan.tex`, README

- [ ] Method: 2–3 sentences on **why long-horizon prediction works at all** —
  the objective is a conditional expectation; the predictor cannot foresee a
  regime switch, it can only encode the current state to lower its error.
  z_slow is a *memo about now*, not a guess about the future. Reviewers make
  this misreading by default.
- [ ] Method: applicability table — (i) dwell ≈ Δ_max → slow (+static);
  (ii) dwell ≫ Δ_max but varies across sequences → static (PTB-XL);
  (iii) invariant across the corpus → out of scope (normal-only anomaly
  detection is reconstruction-AD's job, not ours).
- [ ] Part 2 opening: usage protocols — z_slow probe (robustness, label
  efficiency), z_slow trajectory (HI), per-block queries; fallback sentence:
  unsure which block → use full embedding or diagnose with per-block probe
  scores (same procedure as Fig 2B).
- [ ] F3 prose fix from P0-2 (vfloor vs "emptying").
- [ ] "Separation is one-directional by design" sentence (z_fast retains regime
  0.57 — expected, not a bug).
- [ ] Limitations order: linear-level exclusion → unsupervised model selection
  → xcov polarity → two-timescale assumption. Scaling last (width sweep
  defends it).
- [ ] *(Aug-7 course version only)* explicit **"Relevance to Responsible AI"**
  subsection (~0.25p) — spec and the four pillars are in `CHECKLIST_v3.md`
  Day 3.

---

## Cut order (if the clock runs out, in this order) — **DECIDED (D5): approved**
NCE d_slow sweep → coherence-time sweep (keep the reject demo) → HI metrics
(P3-6) → penalty variants (HSIC) → P4-2 TS2Vec *only if* P2-4 classical
baseline already answered.
H200 note: NCE d_slow and variance-only re-enter the sprint as overnight
filler (compute cost ≈ 0, see v3); the remaining cuts are implementation-bound.

## Execution order recap
```
P0 (P0-5 has a ⛔ STOP) → P1 → P2 (P2-4 risk baseline FIRST in P2)
→ P3 (P3-4 shift first among new modules, after P2-3 unblocks P3-1)
→ P4 (P4-1 highest-value NCE cells before Part-2 stem choice)
```
