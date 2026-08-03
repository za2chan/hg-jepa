# HGLP v2 — Protocol (design decisions BEFORE code)

Every design choice, its value, and its justification. The v1 pipeline's
failure was not bugs — it was undocumented arbitrary choices. Anything below
marked **arbitrary — to be swept** or **DECISION NEEDED** is a choice v1 made
silently; we surface it here and either justify, sweep, or escalate it.

Provenance for the numbers: measured T_ac and segment facts come from the raw
EDA (`docs/eda/README.md`, `runs/tac_real.json`, `runs/datasets_table.json`).
Old pipeline frozen in `legacy/`; old results in `runs/`, `runs_hapt/`, … .
v2 code → `src/`, v2 results → `runs_v2/`. No result reuse.

Status legend: ✅ decided+justified · 🔁 arbitrary→sweep · ⛔ **DECISION NEEDED**.

Measured anchors used throughout (patch units unless noted):

| dataset | T_ac(min)→u-scale [fast] | T_ac(max) [slow, capped] | segment/dwell | notes |
|---|---|---|---|---|
| Synthetic | 5.25 p (≈6.25 true u) | dwell 375 p (3000 steps) | dwell 3000 steps | per-sample GT, clean |
| HAPT | ~13 p ≈ 1.06 s (gait) | ~24 p ≈ 2 s (window-capped) | basic-seg median 17.2 s; **27% VOID; 12 classes** | fast≈1 gait cycle |
| PTB-XL | ~19 p ≈ 1.9 s (beat-ish) | ~48 p (record-capped) | record = 10 s, one label | **heartbeat = coherent-periodic** |
| XJTU | ~7 samples (carrier) | degradation = minutes | 80 ms window ⟶ life const | predicted negative (no in-window gap) |

---

## A. Data

### A1 — Window length & stride ✅ (synthetic) / ⛔ (HAPT) / 🔁 (stride)

The gate needs at least two trained horizons **inside** τ and two **beyond**
(draft rule, `docs/tau.md`): the Δ set must straddle τ, and τ ≈ c·T_ac(min).
So the window must be long enough to host horizons out to ≈ T_ac(max).

- **Synthetic:** L = 256 patches (2048 steps). Justified: Δ_max = 128 p sits
  at ~0.34× dwell and ~20× the fast lifetime — 2 horizons inside τ≈15.7, 3
  beyond. Satisfies the straddle rule. **Keep.** ✅
- **Stride (eval windows):** disjoint (non-overlapping) for probe eval —
  non-negotiable for leak-freedom (C3). Training windows may overlap; overlap
  fraction is **arbitrary — to be swept** (v1 used random start offsets).🔁
- **HAPT:** v1 used L = 128 (10.24 s) with Δ_max = 32 p. EDA verdict: auto-τ
  (39.7 p) sits beyond Δ_max ⇒ 0 horizons beyond τ ⇒ the gate never closes;
  and L = 256 (20.5 s) is **infeasible** — 70% of activity segments are
  shorter than the window, 92% of windows straddle a boundary. So HAPT cannot
  host a straddling Δ set without windows longer than its own activity
  segments. ⛔ **DECISION NEEDED:** (a) keep HAPT with L≈128 and accept it can
  only ever be a "0 beyond τ" datapoint (report as a negative-config, not a
  separation claim); (b) drop HAPT as a slow-factor testbed given the EDA
  (VOID/transition contamination + instantaneously-readable posture) and use
  it only for the shift/label-efficiency probes; (c) restrict HAPT to the
  static-posture subset (SIT/STAND/LAY only) where "slow" = posture is at
  least well-defined. **Recommendation: (b)** — the EDA shows HAPT's activity
  label is largely an end-region readout, so it cannot cleanly evidence slow
  integration regardless of L.

### A2 — Label policy ⛔ (HAPT) / ✅ (PTB-XL, XJTU)

v1 HAPT: `per[end]` (end-point sample label), keep classes 1–6, silently skip
windows whose **end-point** is VOID(0) or transition(7–12) — but a *kept*
window may still contain VOID/transition samples mid-span (57% do: 25% VOID,
20% transition, 12% other-activity; `docs/eda/README.md` STEP 4). Three
undocumented choices bundled together:

1. **Label position** — end-point vs per-position. ⛔ **DECISION NEEDED.**
   Options: (i) end-point label (v1) — but the EDA shows this is locally
   readable, undermining the slow claim; (ii) **per-position labels** — probe
   z_slow at each anchor position against that position's own label, which
   both matches the causal encoder and lets us measure integration honestly;
   (iii) majority-vote over the span — hides transitions. **Recommendation:
   (ii)** per-position, paired with a "window fully inside one segment"
   restriction (below). This is the single change that would make HAPT's
   number mean "slow-factor readout" rather than "end-region readout."
2. **VOID / transition classes** — do not silently drop. ⛔ **DECISION
   NEEDED.** Options: (i) exclude any window containing VOID or 7–12 (clean
   subset only — costs ~57% of windows, n≈1250); (ii) keep transition classes
   7–12 as their own labels (12-class problem); (iii) v1's end-point-only
   filter (rejected — leaves mid-span contamination undocumented).
   **Recommendation: (i)** clean-subset-only for the separation claim, and
   separately report the excluded fraction so the restriction is explicit.
3. **Windows crossing segment boundaries** — currently allowed (that IS the
   contamination). With (ii)+(i) above, forbid it: a window must lie inside a
   single activity segment. Windows never cross file/subject boundaries (v1
   loop is per-file — verified 0 in the audit). ✅ for the file-boundary part.

- **PTB-XL:** one label per record, constant across the 10 s. Keep. Note the
  NORM-vs-all pooling is coarse but is the dataset's own granularity. ✅
- **XJTU:** life-fraction per snapshot, constant across the 80 ms window.
  Keep; state that "life" = snapshot index / (n−1), a normalized position, not
  a physical RUL (`docs/eda`). ✅

### A3 — Normalization ⛔ / ✅

v1: **per-window, per-axis z-score over time** on HAPT/PTB-XL/XJTU inputs
(`(w−w.mean(0))/w.std(0)`), and additionally `F.layer_norm` on the encoder
output (see B2). Two problems the EDA/read surfaced:

- Per-window z-scoring **erases gravity direction** — the exact posture cue
  that distinguishes SIT/STAND/LAY (all ≈1 g, differing only in direction).
  It also **destroys cross-window comparability**, so a slow factor defined by
  absolute level cannot survive. RevIN-style per-window normalization is
  designed for forecasting *distribution-shift robustness* — the opposite of
  our goal, which needs the slow level preserved.
- ⛔ **DECISION NEEDED (normalization scope):** (i) **dataset-global**
  standardization (one mean/std per channel over the training split) —
  preserves gravity direction and cross-window level, recommended; (ii)
  per-recording (per subject/patient/bearing) — preserves within-recording
  level, removes inter-subject offset; (iii) per-window (v1) — rejected for
  the reasons above. **Recommendation: (i)** dataset-global, per-channel,
  fit on the training split only. Synthetic: fed **raw** (unstandardized) in
  v1 — the amplitude cue reaches the encoder; keep raw, it is generated on a
  fixed scale. ✅

### A4 — Channels 🔁 / ✅ document

- **HAPT:** v1 uses accelerometer only; **gyroscope (61 files) is unused.**
  Deliberate simplification for v1. For v2: **arbitrary — to be swept** as an
  ablation (acc-only vs acc+gyro) if HAPT is kept as a testbed; otherwise
  document as a scoped-out modality. 🔁
- **XJTU:** Horizontal channel only; **Vertical unused.** Document as a
  deliberate simplification (the AM framing needs one channel); note it
  explicitly rather than leaving it implicit. ✅ (document)
- **PTB-XL:** lead II only of 12. Standard single-lead choice; document. ✅

### A5 — Synthetic persistence & determinism ✅

- `datagen.generate()` uses `np.random.default_rng(seed)` only — **fully
  deterministic given a seed** (no `Date.now`/global RNG). ✅
- v2: **persist each synthetic dataset to disk with a content hash**
  (`sha256` of the packed arrays) recorded in the run JSON, so every result
  names the exact data that produced it. v1 regenerated on the fly (seed
  logged, bytes not). ✅ (adopt)
- Determinism of training itself (cudnn) is **arbitrary — to be pinned**:
  set deterministic flags + record torch/cuda versions in the run JSON. 🔁

---

## B. Training task

(Read of `legacy/train.py` current state: `P=8, L=256, D_MODEL=96, D_Z=64,
OFFSETS=[1,4,16,64,128], W=4.0, EMA=0.996, STEPS=3000, BATCH=64, LR=3e-4,
n_anchor=8`.)

### B1 — Anchor range ⛔ (confirmed bug) / recommendation given

v1: `anchors = rng.integers(64, L−max(OFFSETS)) = [64, 128)`. Consequences,
confirmed by read:

- Positions **[0,64) and [128,256) are never anchors** ⇒ their position
  embeddings `pos[128:256]` (and `pos[0:64]`) **receive no gradient**.
- Targets are taken at `anchor+Δ ∈ [65, 256)`, i.e. up into the **untrained
  position-embedding range**, and the EMA target encoder inherits those
  untrained embeddings — so long-Δ targets are partly random-position noise.
- Eval then probes position **255** (`enc(x)[:,-1]`) — an untrained position
  entirely outside the anchor range (see C1). Triple inconsistency.

⛔ **DECISION NEEDED (fix):** (a) **relative position encoding** (RoPE /
relative bias) so no absolute position is special and any anchor/target
position is trained — cleanest, removes the whole class of bug;
(b) sample anchors over the full valid range `[Δ_context_min, L−max(OFFSETS))`
AND ensure every probed position is trained; (c) shorten L so the trained band
covers it. **Recommendation: (a)** relative positions — it also directly
supports B3 (horizon generalization) and makes "which position we probe"
a non-issue.

### B2 — Output normalization ⛔ / recommendation given

v1: `F.layer_norm(out, (D_Z,))` over **all 64 dims** jointly. This couples the
slow block's scale to the fast block's content (a large-variance z_fast
rescales z_slow), contradicting block independence — the property the xcov
penalty is separately trying to enforce.

⛔ **DECISION NEEDED:** (i) **per-block LayerNorm** (norm z_slow and z_fast
independently) — keeps each block self-normalized without cross-coupling;
(ii) **no output norm**, rely on the target-encoder EMA + xcov for scale
control; (iii) norm the D_MODEL trunk *before* the split into blocks.
**Recommendation: (i)** per-block norm — minimal change, directly removes the
cross-block coupling, and preserves each block's usability as a probe input.

### B3 — Δ conditioning ✅ decide (continuous) / justification

v1: `nn.Embedding(len(OFFSETS), 16)` — **five independent vectors indexed by
Δ**. This is CPC's per-k heads in disguise: no interpolation between horizons,
no notion that Δ=16 is "between" 4 and 64, and unseen horizons are
unrepresentable.

**Decision:** replace with **continuous conditioning** — feed `log2(Δ)` (a
scalar) through a small MLP, or a sinusoidal embedding of `log Δ`, into the
predictor. ✅ Justification: it buys **generalization to unseen horizons**,
which is exactly what the paper's central "horizon is a free supervision axis"
claim requires — a discrete per-Δ head cannot claim the axis is continuous.
This also lets τ be any real value (A1/D1) rather than snapping to grid Δs.
(Which encoding — MLP(logΔ) vs sinusoid — is **arbitrary → sweep**. 🔁)

### B4 — Anchor / Δ sampling density 🔁 → decide denser

v1: 8 anchors × 1 Δ each = **8 (anchor,Δ) pairs from a 256-position forward
pass** — the forward pass is ~97% wasted (only 8 of 256 positions supervised;
each anchor sees one random Δ, so the Δ axis is sampled 8×/window).

**Decision:** sample **densely** — for each sampled anchor, use **all** (or a
large random subset of) horizons Δ, and sample more anchors per window. Target
≈ (many anchors) × (full Δ set) pairs per forward pass. ✅ direction.
Exact counts (n_anchor, whether full-Δ or subset) **arbitrary → sweep** for
the throughput/variance trade-off. 🔁 Justification: the gate's signal is the
*contrast across Δ at a fixed anchor*; sampling one Δ per anchor gives the
optimizer almost no within-anchor Δ contrast per step.

### B5 — Target definition ⛔ **DECISION NEEDED — highest reviewer risk**

v1 target `z̄(t+Δ) = EMA-encoder(x)[t+Δ]` is a **causal summary of
x[0 .. t+Δ]** — it therefore **contains the anchor's own past** x[0..t]. So at
long Δ, part of "predict the future" is "recall what the anchor already
encodes," which **dilutes the gate's pressure**: the model can score by
copying persistent content it already has, without being forced to write the
slow factor into z_slow specifically.

Options:
- **(i) Keep the causal point target (v1).** Pro: it is genuinely a *point*
  target at t+Δ, which is our stated contrast vs HEPA's interval target, and
  the premise of the gate's "harmless-to-close" argument. Con: the dilution
  above; the "prediction" is partly recall.
- **(ii) Restrict the target encoder's receptive field to x[t+1 .. t+Δ]**
  (a windowed/interval target). Pro: removes the recall shortcut, sharpening
  the gate pressure. Con: this **drifts toward HEPA's interval-summary
  target** and **contradicts our point-target premise** — a reviewer will say
  we became HEPA. Also an interval target still contains fast content at long
  Δ, which is the very thing CLAUDE.md §5 says breaks the harmless-to-close
  argument.
- **(iii) Point target at exactly t+Δ with a receptive field that excludes a
  neighborhood of t** (e.g. target sees x[.. t+Δ] but the *loss* is on the
  residual after removing the anchor-predictable part) — more complex, keeps
  point semantics, needs care.

⛔ **DECISION NEEDED.** **Recommendation: (i) keep the causal point target**,
and address the dilution not by changing the target but by the gate + a
long-Δ-weighted loss (so long horizons, where only slow content survives,
dominate the gradient). Rationale: the point-vs-interval distinction is load-
bearing for the paper's identity and for the Gating argument; trading it away
to fix dilution would cost more than it buys. But this is the decision most
likely to be challenged, so it is yours to set, not mine.

### B6 — Loss, optimizer, schedule ✅ / 🔁

- **Loss = L2 + λ·xcov, no variance floor.** ✅ settled (D7: no collapse at
  vfloor=0; term removed). λ = 4 is **arbitrary → sweep** (v1 default). 🔁
- **EMA rate 0.996** — arbitrary → sweep (standard BYOL-range, untuned). 🔁
- **Optimizer AdamW, LR 3e-4** — arbitrary → sweep (untuned defaults). 🔁
- **STEPS 3000, BATCH 64** — arbitrary; pin by a convergence check (train to a
  loss plateau, record the step count) rather than a magic number. 🔁
- **Gate softness W = 4.0 patches** — arbitrary → sweep; also decide whether
  the gate is soft-sigmoid (v1) or hard (step) as an ablation. 🔁
- **d_slow = 16 of 64** — the mechanism result (v1 F4) says separation tracks
  *absolute* slow-block size; keep 16 as the anchor, sweep as ablation. 🔁

---

## C. Evaluation protocol (v1's weakest part)

### C1 — Which positions are probed ⛔

v1 probes `enc(x)[:, −1]` = **position 255**, which is (a) outside the trained
anchor range [64,128) and (b) an untrained position embedding (B1). The probe
reads a representation the training task never shaped.

⛔ **DECISION NEEDED (tie to B1):** probe **only positions inside the trained
anchor range**, or — with relative positions (B1a) — probe a fixed offset from
the window end that is guaranteed trained. **Recommendation:** with B1a
adopted, probe the last position; without it, probe the center of the trained
band. Either way, **assert probed-position ∈ trained-range** in code.

### C2 — Probing label ⛔ (ties to A2)

v1 probes the **end-point** label. If A2 moves to per-position labels, the
probe must use the label **at the probed position**, not the window end.
⛔ **DECISION NEEDED = A2.1.** Recommendation: per-position label at the probed
position, on the clean single-segment subset.

### C3 — Group splits, disjointness, leakage ✅

v1 does this correctly and v2 keeps it: **group split** (HAPT=subject,
PTB-XL=patient, XJTU=bearing) so no recording appears in both probe-train and
probe-test; **disjoint (non-overlapping) eval windows**; probes fit on the
train group, scored on held-out group. Synthetic: contiguous time-cut (early
train / late test) on non-overlapping windows. ✅ Keep, and add an explicit
assert that train/test window index sets are disjoint.

### C4 — Metric definitions ✅ / 🔁 chance

- **slow-kept** = probe score for the slow factor from z_slow (activity F1 /
  regime acc / life R²).
- **leak** = probe score for the *fast* factor from z_slow (u R² / accmag R² /
  phase R²); lower = better exclusion. Report the linear probe AND (from v1's
  honesty finding) an MLP probe + MINE, since linear probes overstate
  exclusion.
- **chance:** classification → majority-class / balanced-random baseline
  reported alongside (macro-F1 chance = 1/k for balanced); regression → R²
  chance = 0 (predicting the train mean). State chance explicitly per metric
  rather than implying 0. ✅
- **RankMe** effective rank reported as the collapse monitor (replaces the
  removed variance floor as the collapse *check*). ✅
- Absolute scores, never ratios (v1 reviewer #4). ✅

---

## D. Sweeps

**Held fixed** (decided above): loss = L2+xcov (no vfloor); point causal
target (B5, pending your call); continuous Δ conditioning (B3); dataset-global
normalization (A3, pending); relative positions (B1, pending); per-block norm
(B2, pending); group-split leak-free eval (C3).

**Swept** (the arbitraries, grid TBD after B1/B2/B5 land):

| axis | grid (proposed) | question it answers |
|---|---|---|
| τ (via ε or ×auto) | ε ∈ {0.1,0.05,0.02} ⇒ c∈{2.3,3.0,3.9}; ×{0.5,1,2} | plateau around auto-τ |
| Δ set straddle | #horizons {inside,beyond} τ ∈ {(2,0),(2,1),(2,2),(2,3)} | the horizon-count rule |
| λ (xcov) | {0, 1, 4, 16} | exclusion vs slow-kept trade |
| d_slow | {8, 16, 32} | absolute-block-size mechanism |
| gate | soft(W∈{2,4,8}) vs hard | gate-shape sensitivity |
| EMA | {0.99, 0.996, 0.999} | target-stability sensitivity |
| training overlap / n_anchor / Δ-density | TBD | throughput vs variance |
| (HAPT only) acc vs acc+gyro | if HAPT kept | modality ablation |

Seeds: 3 (data-resampling, D3) for every reported cell. Every run JSON records
the full config + data content-hash + library versions (A5).

---

## Open decisions summary (for your STEP 2 review)

⛔ **DECISION NEEDED** — I will not choose these silently:
- **A1/A2** HAPT's role: keep as weak L≈128 testbed / drop to probe-only /
  static-subset. (Rec: drop to probe-only.)
- **A2.1** label position: end-point vs **per-position** vs majority.
  (Rec: per-position, clean single-segment subset.)
- **A2.2** VOID + transition classes: exclude / keep-as-classes / v1 filter.
  (Rec: exclude, report excluded fraction.)
- **A3** normalization scope: **dataset-global** / per-recording / per-window.
  (Rec: dataset-global.)
- **B1** position encoding: **relative** / widen-anchor-range / shorten-L.
  (Rec: relative.)
- **B2** output norm: **per-block** / none / pre-split. (Rec: per-block.)
- **B5** target: **causal point (keep)** / interval / residual. (Rec: keep
  point; highest reviewer risk — your call.)

✅ decided with justification: A5, B3, B6(loss), C3, C4, and the sweep frame.
🔁 arbitrary → swept: stride/overlap, λ, EMA, LR, steps, gate W, d_slow,
Δ-encoding form, sampling density, channels.

No code until you approve this document (STEP 2).
