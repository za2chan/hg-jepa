# HGLP checklist v3 — decisions locked + 5-day plan for the Aug 7 deadline

> Delegation original (hand THIS file + `CHECKLIST_v2.md` to Claude Code).
> `CHECKLIST_v3_ko.md` is a reading mirror for the user — keep both in sync.

Repo: `za2chan/hg-jepa` @ `b4e6c70`. Today = Aug 2.
**Deadline = Aug 7, 23:59 — IEEE conference format, 8 pages incl. references,
for a Responsible-AI course.**

Structure: A. locked decisions → B. 5-day sprint (must land by Aug 7) →
C. post-deadline backlog (workshop version). Background and design rationale
live in `CHECKLIST_v2.md` — **this document wins on any conflict.**

---

## A. Locked decisions (user-approved — Claude Code must not change these)

- **D1. τ anchor rule (amended 2026-08-02, user-approved):** `estimate_tac`'s
  `chosen` = **minimum T_ac across derived series, converted to the u scale**
  (squared/energy series ×2 — squaring halves an OU correlation time;
  raw/envelope ×1). A lower-bound estimate of the fast lifetime.
  **τ = c·chosen, c = ln(1/eps), eps = 0.05 (CLI arg) ⟹ c ≈ 3.0**; `c=`
  survives only as an appendix-sweep override. Verification bar on
  synthetic: energy T_ac ≈ 3.125 patches (±25%), chosen ≈ 5.24 after
  conversion, τ ≈ 15.7 (legacy τ=16 retroactively explained — observed,
  not tuned). Derivation and finding: `docs/tau.md`.
- **D2. NCE online target:** **both-sided gradients (SimCLR-style), no EMA,
  no stop-grad.** Simplest form. If unstable, stop and report (no autonomous
  fallback).
- **D3. Seeds/data:** **(a) `make_dataset(..., seed=seed)`; full synthetic
  re-run.** Folded into the sprint's matrix run — nothing trains twice.
- **D4. Part-2 stem:** rule locked — "the stem that dominates on *both*
  slow-kept and leak in the mechanism matrix carries Part 2; if the metrics
  disagree, **keep Reg** (avoids the real-data re-run)." Apply mechanically
  when results land. **RESOLVED 2026-08-03 → Reg.** NCE-online linearly
  uninformative (0.456 vs 0.776), low leak vacuous → metrics disagree → Reg.
  Mechanism needs a stabilized latent target (nce+ema works, nce+online
  does not); loss form secondary. No real-data NCE re-runs.
- **D5. Cut order:** approved — NCE d_slow → coherence-time sweep (keep the
  reject demo) → HI → HSIC → TS2Vec (only if P2-4's answer is already known).
  Principle: never cut anything a headline claim depends on; cut duplicative /
  bonus evidence first.
- **D6. Near-term goal:** Aug 7 23:59, IEEE format, 8pp. Workshop version
  extends afterwards.
- **D7. vfloor rule:** RankMe shows collapse at `vfloor=0` → promote the term
  into the paper's loss equation / no collapse → default `vfloor=0` and remove
  it. On Day 1, run `git log -S "relu(1.0"` to record when/why it entered.
  **RESOLVED 2026-08-03 → removed.** No collapse (RankMe 13.8→22 without it);
  not neutral (also carried the s0 leak outlier). `vfloor=` default 0 in ALL
  trainers; main matrix = vf0, vf1 → appendix ablation. Entered at commit
  `d741957`, no rationale.

---

## B. 5-day sprint (Aug 3–7) — what MUST be in the Aug 7 version

### Principles
1. Every sentence in the Aug 7 paper is backed by **results that exist at that
   moment**. Unfinished experiments are honestly marked "in progress" — no
   placeholder numbers.
2. Checkpoint reuse first: `runs/model_*.pt` (encoders) and `runs/emb_*.npz`
   (embeddings + labels) already exist. **Shift (P3-4) and label-eff (P3-5)
   run at the evaluation layer with zero retraining** — two-thirds of Part 2
   is effectively free.
3. End of each day: update the list of "sentences the paper can now claim".

### Day 1 (Aug 3) — gate & small integrity fixes
- [ ] ⏱ **Timing measurement (scheduling only):** hardware confirmed —
      one dedicated H200. A 0.5M-param run should take minutes → **re-run
      path confirmed, freeze path abolished.** Measurement only sizes the
      overnight batch.
- [ ] P0-1: `tac.py` + `tau=auto` (rule D1 as amended). **AC: energy T_ac ≈
      3.125 patches (±25%), chosen ≈ 5.24 on synthetic.** If far off, record
      and report — do not silently retune `eps`.
      On pass: the existing tau∈{4,64} sweep becomes retroactively valid as
      the c-sweep (appendix material secured).
- [ ] P0-2: `vfloor=1|0` switch + run-JSON logging. Record the `git log -S`
      finding.
- [ ] P0-3 minimal: mode split `loss=reg|nce` × `target=ema|online` (spec D2).
      Keep old names as deprecated aliases. **No repo-wide rename** — the
      paper text adopts HGLP; code renaming happens after the deadline (no
      large refactors mid-sprint).
- [ ] P0-4: `make_dataset(..., seed=seed)` fix (D3). Draft the protocol
      sentence.
- [ ] P3-1 prep: `datagen.py` gains `regime_mode=freq|variance` + make the
      self-test assert conditional on mode (variance-only returns to the
      sprint thanks to the H200).
- [ ] 🌙 **Day-1 overnight batch launch:** after P0 lands, one
      `run_matrix.sh` pass — full Reg re-run (seed=seed) + new NCE(online)
      cells (gate+xcov / gate-only / xcov-only / no-gate) + vfloor=0 cells
      (both stems) + P3-1 variance-only (2 stems × 3 seeds) + NCE d_slow
      (free filler; was D5's first cut, compute cost has vanished). Whole
      batch = a few hours on the H200 — collect results in the morning.

### Day 2 (Aug 4) — matrix collection + rule application + Part 2 secured
- [x] **Collect the overnight batch & apply the rules mechanically (done
      2026-08-03):** D7 → removed (no collapse, RankMe 13.8→22); vfloor=
      default 0 propagated to all trainers; main matrix = vf0, vf1 → appendix.
      D4 → Reg carries Part 2 (NCE-online uninformative; stabilized target is
      the operative factor, three-point reg+ema/nce+ema/nce+online measured).
      No real-data NCE re-runs needed. NEW OPEN: horizon-count / Δ-set
      derivation (`docs/tau.md`); HAPT horizon extension gated on the
      raw-data audit — Option 2 (L=256) infeasible, Option 1 is the ceiling
      (awaiting user go-ahead).
- [ ] **P3-4 shift_eval.py (top priority):** generate perturbed eval data
      (OU lifetime 50→{20,150}, OU scale ×{0.5,2}, noise) → encode with the
      **fresh checkpoints** (overnight-batch outputs) → probes fit on clean
      data, applied to shifted data, z_slow vs z_full degradation.
      **The abstract's placeholder sentence is waiting for this result.**
- [ ] **P3-5 label_eff.py:** on the fresh `emb_*.npz`, subsample probe labels
      {1%, 10%, 100%}, z_slow vs z_full, **λ tuned for both** input sets.
      Zero retraining.
- [ ] P0-5 Check 1 start: `check_coherent_periodic` (ACF-envelope decay +
      Fisher's g — **on the sample-level signal**, never on patch means).
      ⛔ **STOP: if synthetic fails Check 1, return to threshold calibration;
      all screen-related paper sentences frozen until it passes.**
      **Aug-7 fallback:** if Check 1 does not stabilize within the sprint →
      the Aug 7 version ships the existing C1–C3 screen + one sentence
      "coherent-periodic detection is a planned extension". Finish it in the
      workshop version. (The injected-periodic control signal doubles as
      P3-2's demo.)

### Day 2–3 (Aug 4–5) — remaining runners + optional experiment
(The matrix body moved to Day-1 overnight. The old `cpc_*` runs are preserved
as the nce+ema control — do not delete, do not re-run.)
- [ ] Variance-only result → write the **single result sentence** into the
      paper ("separation confirmed or not under variance-only regimes + do
      the stems differ") — per the standing memory note.
- [ ] (If slack; ~half-day implementation) P3-3 interval-summary target
      ablation — compute is free but the bidirectional encoder + attention
      pooling **implementation** is the cost. If implementation is on track
      by Day-3 noon, launch as overnight batch #2; otherwise → workshop
      version.

### Day 3 (Aug 5) — risk experiment + writing starts
- [ ] **P2-4 classical baseline (FM demodulation → moving average → HMM),
      ~30 lines.** Whichever way it lands, write F1's paragraph accordingly —
      never submit without knowing the answer.
- [ ] Writing: port to IEEEtran (`report.tex` → the 8pp outline is already
      fixed: Intro / Related 2.1–2.4 / Method 3.1–3.4 / Exp 4.0–4.2 /
      Limitations / Appendix).
      **Course context (Responsible AI):** for this version, make the RAI
      significance explicit —
      (a) one abstract sentence + reword the intro payoff paragraph in
          interpretability/reliability vocabulary,
      (b) **a dedicated Discussion subsection "Relevance to Responsible AI"
          (~0.25p)**, four pillars:
          ① interpretability by design — we learn a coordinate system in
            which explanations are unnecessary rather than generating
            post-hoc ones, and interpretability is *measured* by the
            block×factor matrix (contrast with post-hoc SHAP-style methods;
            reuse the lecture framing "measurable intrinsic interpretability
            vs post-hoc explanation"),
          ② responsible scoping — screen + applicability table + predicted
            negative (XJTU) + normal-only exclusion = a built-in procedure
            for knowing **when not to use the method**,
          ③ reliability — the shift result as quantitative evidence of
            robustness under distribution change,
          ④ epistemic honesty — Exclusion's non-guarantee stated, the
            linear-level limitation disclosed, one-directional separation
            made explicit.
      **HEPA scope:** the four-way contrast (EMA/SIGReg/L1/interval) is
      demoted to one–two sentences in Related Works. **Method keeps exactly
      one sentence on target shape (point vs interval)** — that sentence is a
      premise of the gate's logic, not a comparison. Deleting the
      "as in HEPA" misattribution remains mandatory. Prose fixes in one pass:
      - replace the HEPA attribution (P1-4 wording)
      - the "conditional expectation — z_slow is a memo about now" paragraph
      - applicability table (now with the window-contamination axis: dwell-vs-
        Δ_max is one axis, "does the window straddle factor transitions?" a
        separate one — HAPT is NOT static-like-PTB-XL, it is its own category;
        source `datasets_table.py`) + the normal-only precondition sentence +
        the HAPT end-region-readout caveat (F1 magnitude ≠ slow integration;
        clean-subset F1 0.74 is the defensible figure; `hapt_anchor_probe.py`)
      - usage-protocol paragraph (probe / trajectory / per-block queries +
        the fallback sentence)
      - F3 vfloor rewording, the "one-directional by design" sentence
      - Limitations order (linear exclusion → model selection → xcov
        polarity → two-timescale)
      - Prop.→Gating/Exclusion, L_dcor→L_xcov, HGLP naming (text only)

### Day 4 (Aug 6) — figures & number sealing
- [ ] Extend `figures_paper.py`: **every** table/number generated from run
      JSONs. No hand transcription — this paper has two prior incidents of
      figure↔text number drift.
- [ ] Method figure: loss box → abstract `ℓ(·, z̄)` + Reg/NCE branches
      (edit `fig_method_pptx.py`; labels Gating/Exclusion).
- [ ] Port the concept figure (`fig_concept.py`): capsule split at the
      coherent-periodic point / per-method "what fixes the factor's
      location" labels / ordinal-axis caption sentence.
- [ ] Screen flowchart (serial Check 1 → Check 2; if the fallback was used,
      draw C1–C3 as-is).
- [ ] Shift figure (Day-2 result) — Part 2's anchor figure.
- [ ] 8pp budget check: Intro 0.75 / Related 1 / Method 1.5 / Datasets 0.5 /
      Part1 1.5 / Part2 1.25 / Limitations 0.5 / refs 1.

### Day 5 (Aug 7) — consistency pass & submission
- [ ] Regenerate all numbers → figure-vs-text cross-check (automated outputs
      only).
- [ ] Claim–evidence audit: annotate every quantitative claim with its
      backing run tag. Claims without backing are deleted or demoted to
      "in progress".
- [ ] IEEE format check (margins/fonts/reference style), PDF compile, submit.
- [ ] Buffer: keep half a day empty — the landing zone for anything that
      slipped from Days 2–4.

### Intentionally absent from the Aug 7 version (honestly marked "future work")
P3-6 HI · P4-2 TS2Vec · HSIC · coherence-time sweep · repo-wide rename ·
(slack-dependent) P3-3 interval target, P0-5 Check 1 (if the fallback ships).
— P3-1 variance-only and NCE d_slow **returned** to the sprint once the H200
was confirmed. Everything still cut is implementation-bound, not
compute-bound.

---

## C. Post-deadline backlog (workshop version) — the rest of v2

In priority order: finish P0-5 Check 1 (if the fallback shipped) → P3-3
interval-summary target (if it missed the sprint) → P0-6 Δ_max three-tier
verdict (reuse `screen_model.py`) → P1-1 repo-wide rename + tag migration →
P4-2 TS2Vec (+ the unmixing fairness twin) → P4-3 appendix sweeps
(soft/hard gate, HSIC, τ-estimator variants, re-run `select.py`) → P3-6 HI →
coherence-time sweep.

---

## Delegation preamble for Claude Code
1. "**Execute only CHECKLIST_v3.md.** `CHECKLIST_v2.md` is the reference for
   background, design rationale, and the post-deadline backlog — do not start
   a v2 item unless v3 names it. On conflict, v3 wins."
2. "⛔ At stop-the-line points (P0-5, D2 instability): halt and report. No
   autonomous workarounds."
3. "Section A's decisions D1–D7 are immutable. D4/D7 are applied mechanically
   as written once their results land."
4. "At the end of each day: report the list of secured results + the list of
   sentences the paper can now claim."
