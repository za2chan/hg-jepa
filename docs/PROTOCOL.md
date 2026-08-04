# HGLP v2 — Protocol (FINAL — frozen for implementation)

**Status: STEP 2 complete.** Every item below is decided. Implementation
(STEP 3) may begin against this document. Nothing here is "to be decided";
items marked SWEEP have a decided *default* plus a grid.

This document supersedes the STEP 1 draft. It was finalized over three review
rounds (`PROTOCOL_QA_ko.md`, `PROTOCOL_QA2_ko.md`, `PROTOCOL_QA3_ko.md`), which
overturned two of the draft's recommendations and corrected one factual claim.
The "Why" line under each item records what settled it.

Source of truth: this file. `PROTOCOL_ko.md` is a reading mirror.
Old pipeline is frozen in `legacy/`; old results in `runs/`, `runs_hapt/`, ….
v2 code → `src/`, v2 results → `runs_v2/`. **No result reuse across versions.**

Measured anchors used throughout (patch units unless noted; source
`docs/eda/README.md`, `runs/tac_real.json`):

| dataset | patch | T_ac(min) → u-scale [fast] | dwell / segment | note |
|---|---|---|---|---|
| Synthetic | 8 steps | 5.25 p | dwell 3000 steps (375 p) | per-sample ground truth |
| HAPT | 4 samples @50 Hz | ~13 p ≈ 1.06 s (gait) | basic-segment median 17.2 s | 27% VOID, 12 classes |
| PTB-XL | 10 samples @100 Hz | ~19 p ≈ 1.9 s | record = 10 s, one label | heartbeat is coherent-periodic |
| XJTU | 4 samples @25.6 kHz | ~7 samples (carrier) | degradation = minutes | predicted negative |

---

## Decisions at a glance

| Item | Decision |
|---|---|
| A1 window / stride | Synthetic L=256; **HAPT L=256 (20.5 s, was 128)**; PTB-XL 100; XJTU 512. Eval windows disjoint; train overlap swept |
| A2 label policy | **Per-position labels.** Windows may span transitions/VOID. Train cut without looking at labels; score only where a label exists |
| A3 normalization | **Dataset-global, per-channel**, fit on the training split. Synthetic raw |
| A4 channels | HAPT acc 3-axis / PTB-XL lead II / XJTU horizontal. **Channel-mixing is a deliberate design** |
| A5 synthetic data | Persist with sha256; deterministic given seed; pin cudnn + record versions |
| B1 position encoding | Learned absolute retained; problem is fixed by B4, not by the encoding |
| B2 output norm | **Per-block LayerNorm** (after the split) |
| B3 Δ conditioning | **Continuous** — log₂Δ expanded to a vector. Not claimed as novel |
| B4 anchor / Δ sampling | **Anchors over the full valid range**; Δ sampled continuously in log space; dense (anchor × Δ) pairs |
| B5 target | **RF-bounded point target**, `w_eff = min(w, Δ)`. EMA (Reg) / online (NCE) unchanged |
| B6 loss | L2 + λ·xcov, **no variance floor**. Optimizer/schedule defaults swept |
| C1 probe position | Inside the trained anchor range, asserted in code |
| C2 probe label | Label at the probed position (follows A2) |
| C3 splits | Group split + disjoint eval windows + asserts |
| C4 metrics | Absolute scores, chance stated, linear + MLP + MINE, RankMe |
| E1 τ channel | **Per-axis T_ac, min rule** (extends D1) |
| E2 channel design | Channel-mixing, documented with rationale |
| E3 HAPT τ | **Re-estimate at L=256** before fixing the Δ set |

---

## A. Data

### A1 — Window length and stride

**Decision.**

| dataset | L (patches) | window | Δ set |
|---|---|---|---|
| Synthetic | 256 | 2048 steps | {1, 4, 16, 64, 128}; continuous sampling in log space (B4) |
| **HAPT** | **256** (was 128) | **1024 samples = 20.48 s** | finalized after E3 τ re-estimation; expected {1, 4, 16, 64, 128} |
| PTB-XL | 100 | 1000 samples = 10 s | record length is fixed by the dataset |
| XJTU | 512 | 2048 samples = 80 ms | unchanged (predicted negative) |

Training windows may overlap (overlap fraction SWEPT). **Evaluation windows
must be disjoint** — non-negotiable, it is the leak-freedom condition (C3).

**Why.** The gate only produces signal where horizons straddle τ: horizons
inside τ train `z_fast`, horizons beyond τ train `z_slow`. The draft rule
(`docs/tau.md`) asks for ≥2 horizons on each side. Measured against the raw
data:

| L | window | train win | disjoint eval win | labeled positions | horizons beyond τ(39.7 p) |
|---|---|---|---|---|---|
| 128 (v1) | 10.2 s | 4296 | 2165 | 67.5% | **0–1** |
| **256** | **20.5 s** | **2104** | **1067** | **68.5%** | **2** |
| 384 | 30.7 s | 1371 | 699 | 69.3% | 2 (no gain, 35% fewer windows) |

L=256 is the first HAPT configuration in which the gate ever closes. All 61
recordings (min 198 s) fit a 20.5 s window. The earlier claim that L=256 is
"infeasible" rested on 92% of windows spanning a transition — which A2's
reversal removed as an objection.

### A2 — Label policy

**Decision.**
- **Labels are read per position**, not once per window.
- **Windows may contain anything** — activity transitions, postural-transition
  classes 7–12, and VOID. No label-based filtering of the input.
- **Training windows are cut without consulting labels at all.**
- **Probe scoring happens only at positions with a defined label.** HAPT:
  classes 1–6; positions labeled VOID or 7–12 are excluded from scoring and
  the excluded fraction is reported.
- PTB-XL (per-record diagnosis) and XJTU (per-snapshot life fraction) are
  unchanged; both are constant across their window by construction.

**Why.** The STEP 1 draft recommended keeping only windows that lie inside a
single activity segment. That was wrong, and the counter-example is our own
best case: synthetic has dwell 3000 vs window 2048, so **~50% of synthetic
windows contain a regime switch** and separation still works (leak 0.79→0.19).
The real requirements are (1) dwell ≫ Δ, (2) T_ac(fast) ≪ Δ, (3) the factor
varies in the corpus — none of which mention within-window constancy.
Enforcing it would demote the slow case to the static case and delete the
tracking behaviour we most want to demonstrate.

The label-free-in-training claim also depends on this: filtering training
windows by label would put labels back into pretraining. VOID is real sensor
signal that merely lacks annotation (27% of HAPT), so it is kept as free
pretraining data and excluded only from scoring.

### A3 — Normalization

**Decision.** **Dataset-global, per-channel** standardization: one mean and
one std per channel, computed on the **training split only**, applied
everywhere. Synthetic is fed **raw**.

**Why.** Per-window z-scoring (v1) erases exactly the signal we need. HAPT's
static classes SIT/STAND/LAY are all ≈1 g and differ *only* in which axis
carries gravity; subtracting a per-axis per-window mean deletes that. It also
destroys cross-window comparability, so any slow factor defined by absolute
level cannot survive. RevIN-style per-window normalization targets the
*opposite* goal (robustness to distribution shift); we need the slow level
preserved. Global standardization fixes scale for optimization while
preserving both within- and cross-window structure. Synthetic needs none: it
is generated on a fixed scale with a designed SNR (amplitude ≈1, u term 0.2),
and standardizing would distort that.

### A4 — Channels

**Decision.**
- HAPT: accelerometer, 3 axes. **Gyroscope deferred** (not in v2 scope).
- PTB-XL: lead II.
- XJTU: horizontal vibration.
- **Channel handling is channel-mixing, by deliberate design.** The first
  layer is `Linear(patch_len × n_channels, D_MODEL)`, so channels are mixed at
  the input and the parameter count depends on the channel count. We are
  **not** channel-independent.

**Why.** Lead II is the clinical rhythm lead because it is nearly parallel to
the heart's mean electrical axis, giving the largest positive P and QRS
deflections. XJTU's horizontal channel already contains the full AM structure
(carrier plus fault-modulated envelope), which is all the Hilbert-envelope
framing needs; the vertical channel is a largely redundant second view.

Channel-mixing is right for us because the slow factor of interest lives in
the *relation between* channels — posture is the ratio of gravity across x/y/z,
which a channel-independent encoder (PatchTST-style) would represent poorly.
The cost is that weights cannot transfer across datasets with different channel
counts; that is acceptable because we train per dataset. Gyro is deferred
because adding it changes the *fast proxy* (`accmag`) and therefore the leak
metric's definition — a metric change, not a simple ablation.

### A5 — Synthetic data persistence and determinism

**Decision.** Persist every synthetic dataset to disk with a **sha256 content
hash** recorded in the run JSON. `generate()` is deterministic given a seed
(verified: `np.random.default_rng` only, no global RNG or clock). Pin cudnn
deterministic flags and record torch/CUDA versions in the run JSON.

**Why.** v1 regenerated data on the fly and logged only the seed, so a result
named a seed rather than the bytes it was produced from. Hashing makes every
number traceable to exact data, which matters because this paper has two prior
figure↔text drift incidents (CLAUDE.md §4).

---

## B. Training task

### B1 — Position encoding

**Decision.** **Keep learned absolute position embeddings.** The problem this
item was opened for is fixed by B4 (anchor range), not by changing the
encoding. Sinusoidal or RoPE are permitted alternatives but not required.

**Why.** v1 sampled anchors only from `[64, 128)`, so `pos[128:256]` received
no gradient, targets at long Δ were computed on untrained position embeddings,
and evaluation then read position 255. **Correction to the STEP 1 draft:**
`pos[0:64]` *is* trained — a causal encoder's anchor outputs attend over it —
so only `pos[128:256]` was starved. The severity is a train/eval context
mismatch, not a catastrophic bug.

Once anchors cover the full range (B4), every position is trained and every
context length is in-distribution, which removes the starvation and the
mismatch together. RoPE would remove per-position parameters but not the
context-length mismatch, so it solves the smaller half of the problem while
adding length-extrapolation caveats. HEPA's choice (sinusoidal) is a valid
alternative for the same reason. We take the change with the fewest new
trade-offs.

### B2 — Output normalization

**Decision.** **Per-block LayerNorm**: split z into `z_slow` (d_slow) and
`z_fast` (D_Z − d_slow) first, then apply `LayerNorm(d_slow)` and
`LayerNorm(D_Z − d_slow)` independently.

**Why.** v1 applied one LayerNorm across all D_Z dims *after* the blocks were
defined, so the two blocks shared one mean and std: rising `z_fast` variance
rescales `z_slow` down. That directly fights the xcov penalty, which exists to
make the blocks independent. Per-block normalization gives each block its own
statistics and is a one-line change that does not touch the transformer
internals (the trunk keeps its standard pre-LN blocks).

### B3 — Δ conditioning

**Decision.** **Continuous conditioning.** Feed `log₂Δ` to the predictor,
**expanded into a vector** before use — either a small MLP `R → R³² → R¹⁶` or
Fourier features `[sin(ω_k log Δ), cos(ω_k log Δ)]` (which of the two: SWEEP).
Replaces `nn.Embedding(len(OFFSETS), 16)`.

**Why.** v1's embedding-by-index is CPC's per-k heads in disguise: five
unrelated vectors, no notion that Δ=16 lies between 4 and 64, and unseen
horizons are unrepresentable. Our central claim is that horizon is a
*continuous* free supervision axis; a five-slot lookup table cannot support
that claim. Expansion to a vector is required because a bare scalar entering a
linear layer contributes only a rank-1 shift (`w · logΔ`), which cannot express
the qualitatively different behaviour needed at Δ=1 versus Δ=128.

**Not a novelty claim.** HEPA also conditions the predictor on Δt as a scalar.
This is a standard choice and must be written as such.

### B4 — Anchor and Δ sampling

**Decision.**
- **Anchors: sampled over the full valid range** `[min_context, L − Δ_max)`.
- **Δ: sampled continuously**, `log Δ ~ Uniform(log Δ_min, log Δ_max)`, instead
  of drawing from a fixed 5-value grid.
- **Density: many anchors × several Δ per anchor** per forward pass (exact
  counts: SWEEP).
- The sampling procedure must be documented in code comments and in the run
  JSON config.

**Why.** v1 drew 8 anchors, each with a single random Δ — 8 (anchor, Δ) pairs
out of a 256-position forward pass, wasting ~97% of the computation. The gate's
learning signal is the *contrast across Δ at a fixed anchor*, and one Δ per
anchor barely provides it within a step. Full-range anchors simultaneously fix
B1 (no starved positions) and C1 (any probe position is in-distribution), which
is why this is the upstream decision of the three. Continuous Δ sampling is
unlocked by B3 and removes the grid entirely, densifying coverage near τ.

Δ spacing is logarithmic because the governing quantity is the ratio Δ/T_ac,
and the range to cover (T_ac(min) ≈ 5 p to T_ac(max) ≈ 82 p) spans more than
an order of magnitude; linear spacing would crowd the short end. **Recorded
honestly: this is a post-hoc justification** — v1's `[1,4,16,64,128]` was set
in the initial commit without recorded rationale.

### B5 — Target definition

**Decision.** **Receptive-field-bounded point target.**

```
w_eff = min(w, Δ)
ztgt  = TargetEncoder( x[t+Δ − w_eff : t+Δ] )[last position]
```

Target-encoder identity is unchanged and orthogonal to this: **EMA copy for
HGLP-Reg, online (both-sided, no EMA, no stop-grad) for HGLP-NCE** — D2 stays
locked. `w` is initialized from `T_ac(fast)` (label-free) and SWEPT.

**Why.** v1's target was the causal encoder read at t+Δ, whose receptive field
is `[0, t+Δ]` — it re-contains the anchor's own past `x[0…t]`. That breaks the
harmless-to-close premise by construction: `z_fast(t)` describes the signal near
t, which is *inside* the target's input, so cutting it at long Δ is not
automatically harmless. Verified against the sources, **both** reference methods
avoid this deliberately: CPC's target is `z_{t+k}` from a **local** encoder
(cumulative context appears only on the input side), and HEPA's target is a
bidirectional summary of `(t, t+Δ]`. v1 was the outlier.

`w_eff = min(w, Δ)` guarantees the target's receptive field lies inside
`(t, t+Δ]` for **every** Δ, including Δ=1 where a fixed w could not fit. At
short Δ the field shrinks to Δ, which is desirable — the gate is open there and
fast information is *supposed* to help.

Checks that cleared this decision: EMA is retained (receptive field and encoder
identity are independent axes); collapse risk is not increased (EMA asymmetry
intact, and a harder target reduces the temptation to collapse — RankMe still
monitors it); the regression loss is unchanged (the target is still one D_Z
vector); the method remains latent-prediction / JEPA-family and in fact moves
closer to the CPC lineage. The live contrast with HEPA becomes "read one
position inside the future interval vs attention-pool the whole interval."

### B6 — Loss, optimizer, schedule

**Decision.** `loss = ‖ẑ − z̄‖² + λ · L_xcov`. **No variance floor.**

| knob | default | status |
|---|---|---|
| λ (xcov) | 4 | SWEEP {0, 1, 4, 16} |
| EMA rate | 0.996 | SWEEP {0.99, 0.996, 0.999} |
| optimizer / LR | AdamW, 3e-4 | SWEEP |
| steps | — | **train to loss plateau and record the step count**; no magic number |
| batch | 64 | SWEEP |
| gate softness W | 4.0 patches | SWEEP {2, 4, 8}; soft vs hard as ablation |
| d_slow | 16 | SWEEP {8, 16, 32} |

**Why.** D7 resolved the variance floor: at `vfloor=0` there is no collapse
(RankMe *rises*, 13.8 → 22), and the term was not neutral — it also carried the
seed-0 leak outlier (0.50 → −0.09) that widened the error bars. It is therefore
removed from the loss and from the paper's loss equation, with the vfloor=1
cells demoted to an appendix ablation. Every remaining constant above is an
untuned v1 default, so each is swept rather than asserted.

---

## C. Evaluation protocol

### C1 — Probe positions

**Decision.** Probe **only at positions inside the trained anchor range**, and
`assert` that condition in code. With B4's full-range anchors this is satisfied
almost everywhere, so in practice the last position is probed.

**Why.** v1 probed `enc(x)[:, −1]` (position 255), which was outside the anchor
range `[64,128)` and carried an untrained position embedding — the probe read a
representation the training task never shaped. The assert exists so this class
of mismatch cannot silently return.

### C2 — Probe label

**Decision.** Use the label **at the probed position** (per-position), following
A2. HAPT scores classes 1–6 only; VOID and 7–12 positions are excluded and the
excluded fraction is reported.

**Why.** Follows mechanically from A2. Under end-point labeling the probe
question is ambiguous once a window spans several activities; under
per-position labeling it is always well defined, and it additionally lets us
measure *tracking* (how fast `z_slow` updates after a transition) rather than
only steady-state readout.

### C3 — Group splits, disjointness, leakage

**Decision.** Unchanged from v1, which did this correctly. **Group split**
(HAPT = subject, PTB-XL = patient, XJTU = bearing) so no recording appears in
both probe-train and probe-test; **disjoint (non-overlapping) evaluation
windows**; probes fit on the training group and scored on held-out groups.
Synthetic uses a contiguous time cut (early = train, late = test) on
non-overlapping windows. **Add asserts** that the train/test window index sets
are disjoint.

**Why.** This is the fix that made v1's results trustworthy in the first place
(the original leak came from random-splitting overlapping windows). Nothing in
the review touched it; the asserts are added to protect it.

### C4 — Metric definitions

**Decision.**
- **slow-kept** — slow-factor score from `z_slow` (activity F1 / regime acc /
  life R²).
- **leak** — *fast*-factor score from `z_slow` (u R² / accmag R² / phase R²);
  lower is better. Report **linear probe and MLP probe and MINE**, because
  linear probes overstate exclusion (v1 finding).
- **chance stated explicitly per metric**: balanced macro-F1 chance = 1/k;
  R² chance = 0. Never implied.
- **RankMe** effective rank as the collapse monitor (it replaces the removed
  variance floor as the *check*).
- **Absolute scores, never ratios.**

**Why.** Each element traces to a specific v1 review finding: ratios hid
collapse (#4), linear probes overstated exclusion (nonlinear.py: MLP leak 0.45
vs linear 0.19), and the variance floor's removal left collapse unmonitored
unless RankMe is reported.

---

## D. Sweeps

**Held fixed** (decided above, not swept): loss form (L2 + xcov, no vfloor);
RF-bounded point target with `w_eff = min(w,Δ)`; continuous Δ conditioning;
dataset-global normalization; full-range anchor sampling; per-block LayerNorm;
group-split leak-free evaluation; per-position labels.

**Swept:**

| axis | grid | question |
|---|---|---|
| τ (via ε) | ε ∈ {0.1, 0.05, 0.02} ⇒ c ∈ {2.3, 3.0, 3.9}; ×{0.5, 1, 2} | is there a broad plateau containing auto-τ |
| Δ straddle | (inside, beyond) τ ∈ {(2,0), (2,1), (2,2), (2,3)} | the horizon-count rule |
| λ (xcov) | {0, 1, 4, 16} | exclusion vs slow-kept trade-off |
| d_slow | {8, 16, 32} | absolute-block-size mechanism |
| gate | soft W ∈ {2,4,8} vs hard | gate-shape sensitivity |
| EMA | {0.99, 0.996, 0.999} | target-stability sensitivity |
| **target width w** | {1, 2, 4} × T_ac(fast) | receptive-field bound sensitivity |
| Δ encoding | MLP vs Fourier | conditioning form |
| D_MODEL / D_Z | around 96/64 | never justified; the current ratio is atypical |
| train overlap, anchor count, Δ density | TBD | throughput vs variance |

Seeds: **3 per reported cell** (data-resampling, D3). Every run JSON records
the full config, the data content hash, and library versions (A5).

---

## E. Items added during review (not in the STEP 1 draft)

### E1 — τ estimation channel

**Decision.** Estimate `T_ac` **per axis**, then combine with the **minimum**
rule — the same rule D1 already uses across derived series, extended to axes.

**Why.** A mismatch surfaced during review: the model sees 3 axes, but
`tac_real.py` measured T_ac on `‖acc‖` (magnitude). Magnitude is
rotation-invariant, so it has already discarded the gravity-direction cue —
we were tuning τ on a different signal than the model consumes. Per-axis
estimation measures the channels the model actually receives, and the min rule
requires no new decision (D1 is locked and already justifies min as a
lower-bound estimate, with lower being the safe direction per `docs/tau.md`).
Multivariate ACF is the more principled alternative and is deferred to the
workshop version.

### E2 — Channel-mixing design (see A4)

**Decision.** Keep channel-mixing; **document it as a deliberate choice** with
its cost.

**Why.** It had never been decided — it was inherited from the initial commit.
It is nonetheless the right choice here (the posture factor lives in inter-axis
relations), and the cost (no cross-dataset weight transfer) is irrelevant under
dataset-wise training. Recording it prevents it from being read as an accident.

### E3 — HAPT τ re-estimation

**Decision.** **Re-estimate τ on HAPT at L=256** before fixing the Δ set.

**Why.** τ = 39.7 patches was measured inside a 10.24 s window, and ACF cannot
see correlations longer than its own window — so that value is a
window-limited lower bound. Extending the window to 20.5 s (A1) exposes longer
structure and may move τ, which in turn decides the Δ set. This is the same
window-limitation the review identified for `T_ac(max)`.

### E4 — Harmlessness curve (new experiment)

**Decision.** Add an evaluation-layer experiment measuring the
harmless-to-close premise directly. For each Δ, fit two regressions to the
target and compare:

```
A:  s(t)        → z̄(t+Δ)          slow factor only
B:  s(t), u(t)  → z̄(t+Δ)          plus the fast factor
harmlessness(Δ) = R²(B) − R²(A)
```

Run it for **both** the v1 cumulative target and the v2 RF-bounded target.

**Why.** The premise that the entire gating argument rests on has never been
measured. It is a property of the data and the target definition, not of our
model — so it must be measured **without** the gate or the block structure (an
earlier draft of this experiment used a gate-free model's `z_fast`, which is
meaningless because nothing makes those dims "fast"). Ground-truth factors are
available on synthetic; real data uses proxies.

It yields three things: empirical support for the premise; a **direct
comparison of the two target designs** (the cumulative target should *not*
decay to zero, the RF-bounded one should) — which is the strongest available
defence of the B5 decision; and a data-driven τ estimate at the Δ where the
curve reaches zero, merging with the backlog's "model-based τ estimation."

Cost: no additional training; a few Ridge fits on embeddings we already produce.

### E5 — Deferred

| item | when |
|---|---|
| Gyroscope (acc + gyro) | after v2 — it changes the fast-proxy metric definition |
| Tracking-latency experiment (how fast `z_slow` updates after a transition) | performance-measurement stage |
| Multivariate ACF for τ (E1 option c) | workshop version |
| Two-axis applicability table (statistical screen + data-design judgement) | paper-writing stage, with the Responsible-AI section |

The two-axis applicability table is worth stating explicitly: our screen tests
*signal statistics* (timescale gap, long-horizon predictability, window
adequacy) but cannot see *data-collection design* or *label semantics*. XJTU
fails on the second axis (1.28 s recorded per minute, so the slow axis is never
observed continuously), and HAPT's quirks (27% VOID, end-point labels, separate
transition classes) are label-semantics facts. Presenting suitability as two
axes — one automatic, one requiring judgement — strengthens the responsible-
scoping pillar rather than weakening the method.

---

## Appendix — what changed from the STEP 1 draft, and why

| # | Change | Trigger |
|---|---|---|
| 1 | **A2 reversed**: "clean single-segment windows only" → per-position labels, windows may span anything | Synthetic (our best case) has ~50% of windows containing a switch and separation works; the restriction would demote slow → static |
| 2 | **B5 target redefined**: cumulative causal → RF-bounded point target | CPC and HEPA source check: both exclude the anchor's past from the target; v1 was the outlier |
| 3 | **CLAUDE.md §5 corrected**: "point target at t+Δ" → "causal cumulative summary read at t+Δ" (v1) / RF-bounded point target (v2) | Same source check; user approved the locked-text edit |
| 4 | **B1 severity corrected**: `pos[0:64]` is trained; only `pos[128:256]` was starved | Re-derivation prompted by review; the draft overstated the bug |
| 5 | **B1/B4/C1 merged**: anchor sampling is the upstream decision | Position encoding, probe position and sampling were three views of one choice |
| 6 | **B3 novelty dropped**: continuous Δ conditioning is standard | HEPA conditions on scalar Δt too |
| 7 | **A1 HAPT extended** to L=256, reinstated as a full testbed | A2's reversal removed the contamination objection; L=256 is the first config where the gate closes |
| 8 | **VOID kept in pretraining** | Filtering by label would compromise the label-free-in-training claim |
| 9 | **E1–E4 added** | Surfaced by review: τ channel mismatch, undecided channel design, window-limited τ, unmeasured premise |
| 10 | **XJTU rationale corrected**: not "no slow state in the window" but "no fast/slow contrast, so the gate contributes nothing (z_slow 0.18 = z_full 0.18)" | Structurally XJTU is a static-factor case like PTB-XL; the original wording was inconsistent |
