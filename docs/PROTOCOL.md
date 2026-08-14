# HGLP v2 — Protocol (FINAL — frozen for implementation)

**Status: STEP 2 complete; STEP 3 in progress.** Stems amended 2026-08-04
(user-approved): HGLP-Reg = **L1**+EMA, HGLP-NCE = InfoNCE+**EMA**, and BOTH are
carried in Part 1 and Part 2 (D4's "pick one" was a compute-cost rule that no
longer applies). B1 is now **RoPE**. See CLAUDE.md §2 for the evidence.

Every item is decided; items marked SWEEP have a decided *default* plus a grid.
**The B5 gate is CLEARED** — the pilot (`src/pilot_b5.py`) confirmed the
RF-bounded target is stable and materially better than the v1 cumulative target.
`src/` is refactored into `hglp/` · `prep/` · `exp/`. Done: B5 pilot, 4-stem ×
3-seed matrix on all three datasets, gate×xcov ablation for both stems.
Next: domain robustness (Part 2), then post-hoc-rotation and Part-2 baselines.

This document supersedes the STEP 1 draft. It was finalized over three review
rounds (`PROTOCOL_QA_ko.md`, `PROTOCOL_QA2_ko.md`, `PROTOCOL_QA3_ko.md`) plus a
cross-review by a second session; together they overturned three of the draft's
recommendations and corrected two factual claims (see the Appendix). The "Why"
line under each item records what settled it.

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
| B1 position encoding | **RoPE (adopted 2026-08-04).** No per-position parameter exists → starvation impossible; prefix-invariance makes B5's variable-length slices in-distribution (asserted in `model.py`) |
| B2 output norm | **Per-block LayerNorm** (after the split) |
| B3 Δ conditioning | **Continuous** — log₂Δ expanded to a vector. Not claimed as novel |
| B4 anchor / Δ sampling | **Sample Δ first, then anchor over `[min_context, L − Δ)`** (fixes the starvation bug); Δ continuous in log space; dense (anchor × Δ) pairs |
| B5 target | **RF-bounded point target**, `w_eff = min(w, Δ)`. **Gate CLEARED — adopted.** EMA target for both stems. With RoPE no slice-indexing convention is needed |
| B6 loss | **L1** + λ·xcov, **no variance floor** (L2 demoted to ablation: unstable, 0.641±0.239). Optimizer/schedule defaults swept |
| C1 probe position | **Multi-position: probe every labeled position, one score each.** Last-position readout kept for the usage protocol |
| C2 probe label | Label at the probed position (follows A2) |
| C3 splits | Group split + disjoint eval windows + asserts |
| C4 metrics | **Block × factor matrix + dimension-matched random-subspace null**; absolute scores, chance stated, linear + MLP + MINE, RankMe (amended 2026-08-05) |
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

**Decision (revised 2026-08-04 — RoPE adopted).** **Rotary position encoding.**
The fallback branch below was taken: B5's variable-length slices made the
position-tagging question unavoidable, and RoPE removes it rather than answering
it. Two properties are asserted in `model.py`:

1. **No per-position parameter exists**, so starvation is impossible by
   construction (the learned-absolute run left 12 positions untrained even after
   B4, because the anchor upper bound `L − Δ_min` never reaches the tail).
2. **Prefix invariance** — `enc(x[:, :k]) == enc(x)[:, :k]` to 1e-5. A slice
   encoded standalone equals the same slice as the prefix of a longer sequence,
   which is exactly what makes B5's variable-length target slices
   in-distribution with no indexing convention. This is FALSE for learned
   absolute embeddings.

Measured effect (single seed, synthetic slow-kept): reg+ema 0.535 → 0.583,
nce+ema 0.972 → 0.994, nce+online 0.972 → 0.992. Anchors per window were raised
8 → 16 at the same time, since with no per-position parameters anchor density is
purely about training signal rather than coverage.

**Superseded decision (kept for the record).** **Learned absolute position
embeddings are the default.** The problem this item was opened for is fixed by
B4 (anchor range), not by changing the encoding. **But B5 weakened the
premise:** feeding the target encoder a short slice raises a position-tagging
question that absolute embeddings answer awkwardly (see B5). **RoPE is the
explicit fallback** if the B5 pilot shows position-related instability.

**Why.** v1 sampled anchors only from `[64, 128)`, so `pos[128:256]` received
no gradient, targets at long Δ were computed on untrained position embeddings,
and evaluation then read position 255. **Correction to the STEP 1 draft:**
`pos[0:64]` *is* trained — a causal encoder's anchor outputs attend over it —
so only `pos[128:256]` was starved (attention trains the anchor's *past*, i.e.
lower indices; it can never reach the anchor's *future*, the higher indices).
The severity is a train/eval context mismatch, not a catastrophic bug.

Once anchors cover the full range (B4), every position is trained and every
context length is in-distribution, which removes the starvation and the
mismatch together. RoPE would remove per-position parameters but not the
context-length mismatch, so on its own it solves the smaller half of the
problem while adding length-extrapolation caveats. **However**, once B5 feeds
variable-length short slices to the target encoder, RoPE gains a second
advantage: it is relative-only, so a short slice uses only short relative
distances (always in-distribution), with no per-position parameter to
mismatch. That makes it the natural fallback rather than a co-equal
alternative. HEPA uses sinusoidal for a related reason. Default stays learned
absolute (fewest changes); the pilot decides whether to switch.

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
- **Δ is sampled first**, `log Δ ~ Uniform(log Δ_min, log Δ_max)`, instead of a
  fixed 5-value grid.
- **The anchor range is then set per-sample to** `[min_context, L − Δ)` — bound
  to the Δ just drawn, **not** a fixed `L − Δ_max`.
- **Density: many anchors × several Δ per anchor** per forward pass (exact
  counts: SWEEP).
- The sampling procedure must be documented in code comments and in the run
  JSON config.

**Why.** v1 drew 8 anchors, each with a single random Δ — 8 (anchor, Δ) pairs
out of a 256-position forward pass, wasting ~97% of the computation. The gate's
learning signal is the *contrast across Δ at a fixed anchor*, and one Δ per
anchor barely provides it within a step.

**Bug this fixes (caught in cross-review; the STEP 1 draft reproduced it).** A
fixed anchor upper bound `L − Δ_max = 256 − 128 = 128` is *identical to v1's*,
so `pos[128:256]` still starves — the "full valid range" phrasing was
self-contradictory. The cause is applying the `anchor + Δ ≤ L` constraint at
the *maximum* Δ uniformly, when it only needs to hold for the Δ actually used.
Sampling Δ first and bounding the anchor at `L − Δ` lets small Δ push anchors
to the end of the window (e.g. Δ=1 → anchors up to 254), so **every** position
becomes an anchor for some Δ and every position embedding is trained.

**Side effect to document (unavoidable geometry).** Anchor position now
correlates with Δ: large Δ forces anchors toward the front (short context) and
leaves fewer valid anchor positions, so far-horizon training signal comes from
early, less diverse anchors. This is inherent to causal prediction in a finite
window. It is mild here (Δ_max=128, L=256 ⇒ far-Δ anchors still span ~96
positions) and benign (slow state reads fine from short context), but must be
stated.

Full-range anchors simultaneously fix B1 (no starved positions) and C1 (any
probe position is in-distribution), which is why this is the upstream decision
of the three. Continuous Δ sampling is unlocked by B3 and removes the grid
entirely, densifying coverage near τ.

Δ spacing is logarithmic because the governing quantity is the ratio Δ/T_ac,
and the range to cover (T_ac(min) ≈ 5 p to T_ac(max) ≈ 82 p) spans more than
an order of magnitude; linear spacing would crowd the short end. **Recorded
honestly: this is a post-hoc justification** — v1's `[1,4,16,64,128]` was set
in the initial commit without recorded rationale.

### B5 — Target definition

**Decision.** **Receptive-field-bounded point target. GATE CLEARED by the
pilot (2026-08-04) — adopted.**

**Pilot result** (`src/pilot_b5.py`, synthetic, HGLP-Reg, 2500 steps, seed 0;
`runs_v2/pilot_b5.{json,png}`). Bounded vs cumulative target:
- **Harmlessness curve as predicted, decisively.** Bounded: 0.10 at Δ=8 →
  ~0.00 by Δ=16 and flat at 0 through Δ=128 (fast info becomes useless at long
  Δ — harmless-to-close holds *by construction*). Cumulative: stays positive
  at **every** Δ (0.024→0.008), never reaching 0 — the anchor's past keeps
  leaking into the target.
- **Bounded is also materially better on the actual goal:** leak (u R² from
  z_slow) **0.041 vs 0.66**; target RankMe **12.9 vs 5.3** (cumulative shows
  partial collapse); slow-kept 0.50 vs 0.46.
- **Stability:** bounded converges, RankMe healthy, no collapse. Minor caveats
  to watch in the full run, not blockers: a small late-training loss uptick
  (0.02→0.06; target getting harder, not collapse), and slow-kept ≈0.50 under
  per-position probing (includes transition + short-context positions; a pilot
  sanity number, not the final metric). Fallback (separate local target
  encoder) is **not** needed.

Scope note: the pilot used Δ_min = w so `w_eff == w` for every pair (see
`src/train.py`) — exactly the decided design in the Δ≥w regime where
harmlessness is claimed. The full run drops Δ_min to 1 with slices grouped by
`w_eff`.

```
w_eff = min(w, Δ)
ztgt  = TargetEncoder( x[t+Δ − w_eff : t+Δ] )[last position]   # slice indexed from 0
```

Target-encoder identity is unchanged and orthogonal to this: **EMA copy for
HGLP-Reg, online (both-sided, no EMA, no stop-grad) for HGLP-NCE** — D2 stays
locked. `w` is initialized from `T_ac(fast)` (label-free) and SWEPT.

**Slice position indexing (settled in cross-review).** Feed the slice as a
standalone short sequence, **indexed from 0** — not with the absolute indices
`t+Δ−w … t+Δ`. Reason: with learned absolute embeddings, `pos[k]` is trained
coupled to "k tokens of history"; the last slice token has only `w_eff−1`
tokens of history, so indexing from 0 (last token = `pos[w_eff−1]`) matches
that coupling and mimics a window prefix (in-distribution), whereas absolute
indexing tags a short-history token with a mid-window position (out of
distribution). RoPE, being relative-only, does this natively and is the
fallback (see B1). This corrects an earlier suggestion to keep absolute
indices.

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

**Why GATED, not locked.** B5 is the one change no one has trained yet, and it
carries a residual risk the pilot must clear: the target encoder is the EMA of
the *context* encoder, which is optimized on full windows, yet B5 applies it to
short slices. This is milder than it first sounds — a short slice is
in-distribution (it is exactly a window *prefix*, which the encoder processes
constantly, and under B4 those prefix positions are anchors too) — so
catastrophic failure is ruled out a priori. What remains empirical is whether
representations *shaped as context* also serve well *as targets*. The B5 pilot
(part of E4) checks it cheaply via RankMe on the target, target-informativeness,
and loss stability. **Fallback if it fails:** a separate small local target
encoder that only ever sees `w`-slices (CPC-style), which removes the reuse
mismatch by construction. Until the pilot clears it, B5 is the leading
candidate, not a locked decision.

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

**Decision.** **Multi-position probe.** Extract `z_slow` at **every labeled
position** inside the trained anchor range and score each against that
position's own label (`assert` the position is in-range). Cluster by window for
error bars (positions within a window are correlated — treating them as
independent would fabricate significance). The single **last-position readout
is kept, but as the *usage* readout** (§1 usage protocols), not as the
measurement.

**Why.** Two things share the word "probe" and must be separated. (1) The
*measurement* — "does `z_slow` capture the slow factor, and by long integration
rather than an end-point shortcut?" — is exactly what A2's per-position labels
exist to test; probing only the last position would re-open the shortcut A2
closed, and it cannot show tracking. (2) The *usage* readout — "what is the
state now?" — is legitimately the last position ("z_slow is a memo about now",
CLAUDE.md §5). These answer different questions and do not compete: multi-
position **subsumes** last-position (you can always read off the last score),
and it additionally yields the position-vs-performance curve and the
transition-region trajectory. The behaviour just after a transition is
**tracking latency, reported as a curve — not an error**. Multi-position is
therefore the more faithful test of our own "memo about now" thesis (checked at
every now, not only the last).

v1 probed `enc(x)[:, −1]` (position 255), which was outside the anchor range
`[64,128)` and carried an untrained position embedding — a representation the
training task never shaped. B4 fixes the training side; the in-range assert
keeps this class of mismatch from silently returning.

**Correction to the FINAL draft's earlier wording** ("in practice the last
position is probed"): that reproduced the end-point method A2 replaced. It is
superseded by multi-position here.

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

**Decision (amended 2026-08-05).** The unit of reporting is the **block × factor
matrix with a dimension-matched random-subspace null**, not a pair of `z_slow`
numbers. Every row is scored on **both** factors:

| row | expected |
|---|---|
| `z_slow` (d_slow) | slow ↑, fast ↓ |
| `z_fast` (D_Z − d_slow) | slow ↓, **fast ↑** |
| `rand<d>` — a random d-dim subspace of the SAME embedding, both block widths | the reference |
| `z_full` | ceiling |

- **slow-kept** — slow-factor score from `z_slow` (activity F1 / regime acc /
  life R²).
- **leak** — *fast*-factor score from `z_slow` (u R² / accmag R² / phase R²);
  lower is better **only when read against `rand<d_slow>`**. Report **linear probe
  and MLP probe and MINE**, because linear probes overstate exclusion (v1 finding).
- **SEP** = inclusion × allocation × exclusion, each a RAW score clipped to
  [0,1]. **No denominators anywhere** (revised 2026-08-07):
  1. **inclusion** — is the slow factor in `z_slow`? Its slow score.
  2. **allocation** — is the fast factor recoverable from `z_mix`? Its fast score.
  3. **exclusion** — is `z_slow` free of the fast factor? `1 − z_slow_fast`.
- **Why no denominator.** Inclusion and allocation never had one: SEP is a product,
  so a denominator shared by every cell in a table is a constant factor that cannot
  change the ranking and only adds a "do not compare across tables" caveat.
  Exclusion kept one until 2026-08-07, justified as stopping a model that encoded
  the fast factor NOWHERE from collecting a perfect exclusion for free. **That job
  belongs to allocation, which does it alone**: the vacuous `nce+online` cell on
  HAPT carries the fast proxy at 0.164 in its complement against 0.625 for
  `nce+ema`, and the product collapses on that term by itself (SEP 0.129 vs 0.525).
  Keeping the denominator as well made **two of the three terms move with the same
  property** — how much fast information the model represents at all — so a model
  was rewarded for it twice. Measured across the six post-hoc-rotation tables,
  dropping it **changes no winner and no top-3**, and shifts values by 0.001–0.037.
- **Two things fall out of this.** (a) `sep_index` no longer reads `z_full`, so
  "which model's `z_full`" stops being a question — it was answered wrong once,
  from a different training run. (b) The index applies **across architectures**:
  TS2Vec's 320 dims and PatchTST's `C*128` need no shared reference encoder.
- **The random-subspace row stays.** Exclusion is not normalised by
  `rand<d_slow>` either — over 30 draws its CV is 0.2–0.7% (synthetic) but 18–19%
  (PTB-XL) and 45–49% (HAPT, range [0.013, 0.222], a factor of 17), so dividing by
  it would inject that noise everywhere. It is reported as a **row**, where its
  spread is visible. On datasets where 16 dims naturally carry little of the fast
  factor, exclusion is high for everything (HAPT's random 16-dim readout already
  scores 0.042–0.109); **only the random row makes that readable.**
- **SEP is a summary, not a verdict.** Report the raw four numbers — (`z_slow`
  slow, `z_slow` fast, `z_mix` slow, `z_mix` fast) — beside it, and never settle a
  comparison on the scalar alone. In particular **always show the inclusion
  column**: SEP is a product, so a high exclusion can buy back lost inclusion, and
  exclusion is capped at 1.0 while inclusion is bounded by what the task allows.
  Measured: a representation holding 0.518 of the slow factor with a near-perfect
  exclusion ties one holding 0.754 with a partial exclusion.
- **Naming (`z_mix`, adopted 2026-08-06; code rename deferred to post-deadline
  per §4-6).** The complement block is called **`z_mix`** in the paper; the code
  keeps `z_fast` and every stored JSON key is `z_fast`. `z_fast` is a misnomer:
  the gate exposes that block only for Δ < τ, and near-horizon prediction needs
  the current slow state as well as the fast one, so it holds **both** factors
  (measured slow score 0.36–0.87). Nothing constrains its content; only its
  horizon availability is constrained. Naming it "fast" invites the reader to
  expect a symmetric split that the method never claims.
- **chance stated explicitly per metric**: balanced macro-F1 chance = 1/k;
  R² chance = 0. Never implied.
- **RankMe** effective rank as the collapse monitor (it replaces the removed
  variance floor as the *check*).
- **Absolute scores, never ratios.**

**Why.** The original one-sided form (slow high, leak low, both from `z_slow`)
cannot distinguish separation from three impostors that all produce a low leak:
a model that never encoded the fast factor anywhere (nce+online: HAPT `z_fast`
carries the proxy at 0.163), a subspace that drops it by accident because it
sits in low-variance directions (PCA on HAPT), and a fast proxy too weakly
represented for ANY subspace to score on. The `z_fast` row catches the first
two; the random null catches the third and makes "leak 0.045" readable — on
HAPT an arbitrary 16-dim readout already scores only 0.188, so a low leak there
means far less than the same number on PTB-XL, where the null is 0.445.

**Correction (2026-08-05, same day).** An earlier draft of this section claimed
that post-hoc rotations pass the allocation test "by construction" because they
are full-rank. **That is wrong, and the first two-sided run refuted it.** A
full-rank split guarantees the two halves *jointly* retain everything; it says
nothing about either half alone. Measured on synthetic (nce+ema, 3 seeds): PCA's
top-16 carries the fast factor at 0.879 — *above* the random-16 null of 0.852,
i.e. variance ranking actively concentrates it — while its complement carries
−0.006. PCA fails **both** sides. So allocation is a real constraint on the
baselines too, and the two-sided matrix is informative for them, not just for us.

The asymmetry that does survive is narrower: the gate is compared against a
*differently trained* encoder, so its `z_full` can differ from the ungated one,
whereas every post-hoc method is a re-basis of one fixed embedding and shares its
`z_full` exactly.

The rest traces to specific v1 review findings: ratios hid collapse (#4), linear
probes overstated exclusion (nonlinear.py: MLP leak 0.45 vs linear 0.19), and
the variance floor's removal left collapse unmonitored unless RankMe is reported.

**Caveat surfaced by the null (2026-08-05):** B2's per-block LayerNorm is applied
to the two blocks separately *regardless of gate/xcov*, so the `g0_x0` cell is
not a mechanism-free control — it privileges the coordinate split, and it
normalises away each block's magnitude, which is what HAPT's fast proxy is. The
random subspace is the genuinely mechanism-free reference; the gap between the
ungated block and the null measures per-block LN's own contribution.

---

## D. Sweeps

**Held fixed** (decided above, not swept): the two stems (`l1+ema`, `nce+ema`;
L2 and `nce+online` are ablation cells, not the loss form — amended 2026-08-04);
xcov, no vfloor; RF-bounded point target with `w_eff = min(w,Δ)`; continuous Δ
conditioning; dataset-global normalization; Δ-then-anchor sampling over
`[min_context, L−Δ)`; per-block LayerNorm **wherever λ > 0**; group-split
leak-free evaluation; per-position multi-position probing.

**The mechanism-free control is `g0_x0_noBlockLN`** (added 2026-08-07). Per-block
LayerNorm normalises the two blocks separately, which privileges the coordinate
split on its own — a control that keeps it is not mechanism-free. Since the LN
exists to stop block coupling from fighting xcov, it travels with the xcov arm,
which fixes the reported 2×2 as

```
g0_x0_noBlockLN | g1_x0_noBlockLN
g0_x1_LN        | g1_x1_LN
```

Corollary: the λ sweep's domain is **λ > 0** ({1, 4, 16, 64}). The λ = 0 point
would need LN off to be consistent, and that cell is already the ablation's
`g1_x0_noBlockLN`.

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

Seeds: **5 per reported cell** (data-resampling, D3) — raised from 3 on
2026-08-07 because HGLP-NCE's across-seed sd is ~2.4× HGLP-Reg's (±0.079 vs
±0.033 on SEP), so 3 seeds could not separate the stems. Tables that still carry
3 seeds (τ and d_slow sweeps on real data) **state n in the caption**; mixing n
inside one table is not allowed. Every run JSON records the full config, the
data content hash, and library versions (A5) — **including `n_seed`**, and where
practical the per-seed values, since a paired test needs them and averaging them
away made one run unusable.

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

### E4 — B5 pilot + harmlessness curve (runs FIRST, gates B5)

**Decision.** Before B5 is locked and before the full matrix, run an **early
pilot** that does two separable jobs:

1. **B5 stability** — train one small model with the v1 cumulative target and
   one with the v2 RF-bounded target; watch loss convergence, RankMe on the
   target, and probe informativeness. This is what decides whether B5 is
   adopted or falls back to a separate local target encoder.
2. **Harmlessness curve** — on those models, for each Δ fit two regressions to
   the target and compare:

   ```
   A:  s(t)        → z̄(t+Δ)          slow factor only
   B:  s(t), u(t)  → z̄(t+Δ)          plus the fast factor
   harmlessness(Δ) = R²(B) − R²(A)
   ```

**Why the reorder.** B5 is the single unvalidated change and the biggest
schedule risk of the rewrite; it must be judged before the pipeline is built on
top of it. The pilot is the judge.

**Correction to an earlier claim ("no additional training").** That was wrong.
The harmlessness curve is a property of *the data and the target definition*,
so it must be measured without the gate or block structure — but the target
`z̄` is still produced by an encoder, and comparing the two target *designs*
means training one model per design. So this is **≥2 trainings**, not a pure
eval-layer analysis. (Note: stability and the harmlessness curve are distinct —
the curve does not by itself test convergence; the pilot bundles both because
both need the same two trained models.)

It yields three things: empirical support for the premise; a **direct
comparison of the two target designs** (the cumulative target should *not*
decay to zero, the RF-bounded one should) — the strongest available defence of
the B5 decision; and a data-driven τ estimate at the Δ where the curve reaches
zero, merging with the backlog's "model-based τ estimation." Ground-truth
factors are available on synthetic; real data uses proxies.

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

### Cross-review round (after the FINAL draft was first written)

| # | Change | Trigger |
|---|---|---|
| 11 | **B4 anchor bug fixed**: fixed upper bound `L−Δ_max` reproduced v1's starvation of `pos[128:256]` → **sample Δ first, bound anchor at `L−Δ`** | A second session showed "full valid range = `[min_context, L−Δ_max)`" is self-contradictory; the fix trains every position. Anchor–Δ correlation documented as an unavoidable side effect |
| 12 | **B5 gated, not locked**: leading candidate pending an early pilot | B5 is the only untrained change; the EMA-context-encoder-on-short-slices reuse is benign a priori (a slice is a window prefix) but its target quality is empirical |
| 13 | **B5 slice indexing decided**: index from 0 (not absolute); RoPE is the native/fallback form | With learned absolute embeddings `pos[k]` couples to "k tokens of history"; absolute indexing on a short slice is OOD. This corrects the reviewer's "keep absolute (b)" suggestion |
| 14 | **C1 → multi-position probe**; last-position kept only as the usage readout | The FINAL draft's "probe the last position" reproduced the end-point method A2 replaced. Multi-position subsumes it and yields the tracking curve |
| 15 | **E4 reordered to run first and gate B5; "no additional training" retracted** (it needs ≥2 trainings) | The pilot must judge B5 before the pipeline is built on it; harmlessness-curve and stability are distinct but share the two trained models |
| 16 | **B1 fallback elevated**: RoPE is the explicit fallback if the B5 pilot shows position instability (was "keep absolute, document") | B5's variable-length slices weaken the case for learned absolute positions more than first stated |
