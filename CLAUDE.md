# CLAUDE.md — HGLP (Horizon-Gated Latent Prediction)

This file is the standing context for every Claude Code session in this repo.
Read it before touching anything. It encodes what the project IS, what has
been DECIDED, and what must never be silently changed. Task lists live in
`docs/CHECKLIST_v3.md` (execute) and `docs/CHECKLIST_v2.md` (reference).

---

## 1. What this project is — the five-sentence map

1. **One-sentence idea:** to predict the far future even slightly, a model
   must write down the current state — that memo pad is `z_slow`.
2. **One mechanism:** a single cut at τ on the persistence axis. Everything
   remembered longer than τ → `z_slow`; everything forgotten faster → `z_fast`.
3. **Two honesty clauses:** inclusion is guaranteed by the objective
   (*Gating*); exclusion is NOT — the objective cannot force fast information
   out of `z_slow` (*Exclusion*); in practice the narrow block + encoder
   smoothness do it, empirically, at the linear level only.
4. **Three applicability conditions:** (i) no coherent-periodic component
   dominating the window, (ii) a timescale gap exists (a persistent component
   beyond τ), (iii) the factor of interest actually varies in the training
   corpus. Normal-only anomaly detection is explicitly out of scope.
5. **Three usage protocols:** `z_slow`-only probes (shift robustness, label
   efficiency) · `z_slow` trajectories (health indicators) · per-block
   queries (fault type on `z_fast`, point/contextual residuals).

Everything else in the paper is a footnote to one of these five.

## 2. Naming (current, mandatory)

- Project: **HGLP** (Horizon-Gated Latent Prediction). "JEPA" appears only as
  a searchable keyword in README subtitle/abstract, never as the method name.
- Two stems, same backbone/gate/penalty, different loss layer:
  - **HGLP-Reg** = `loss=reg target=ema` (L2 to an EMA target encoder;
    collapse handled by EMA asymmetry)
  - **HGLP-NCE** = `loss=nce target=online` (InfoNCE vs in-batch negatives,
    both-sided gradients, no EMA, no stop-grad; collapse handled by negatives)
- Penalty: **`L_xcov`** (squared cross-covariance between blocks). The old
  name `dcor` collides with Székely's distance correlation — do not reuse it.
- Theory sections: **Gating / Exclusion** (informal arguments). Never
  "Prop. 1 / Prop. 2" — these are not theorems.
- Old CLI names (`nepa`, `cpc`, `dcor=`) survive one commit as deprecated
  aliases only.

## 3. Locked decisions D1–D7 (user-approved; immutable)

- **D1** (amended 2026-08-02, user-approved) τ anchor: `estimate_tac`'s
  `chosen` = **min T_ac across derived series** (raw / squared-energy /
  envelope), **converted to the u scale** (squared series ×2 — squaring
  halves an OU correlation time; raw/envelope ×1); a lower-bound estimate
  of the fast lifetime (safe direction: τ too short fails toward the
  ungated control, not toward false separation). **τ = c·chosen with
  c = ln(1/eps), eps = 0.05 ⟹ c ≈ 3.0** (eps = residual fast
  autocorrelation tolerated at the gate); `c=` survives only as an
  appendix-sweep override. Sanity bar on synthetic: per-series energy
  T_ac ≈ 3.125 patches (±25%), chosen ≈ 5.24 after conversion, τ ≈ 15.7 —
  retroactively consistent with the legacy hardcoded τ=16 (observed, not
  tuned). The original 6.25-patch bar was the unobservable u's own
  lifetime — a spec error, corrected here.
- **D2** NCE online target: **both-sided gradients, no EMA, no stop-grad.**
  If unstable → ⛔ stop and report; no autonomous fallback.
- **D3** `make_dataset(..., seed=seed)`: seeds vary the data; all synthetic
  error bars regenerate; the re-run is folded into the overnight matrix.
- **D4** Part-2 stem rule: the stem dominant on BOTH slow-kept and leak
  carries Part 2; metrics disagree → keep Reg. Apply mechanically.
  **RESOLVED 2026-08-03 → Reg carries Part 2.** NCE-online (D2 pure form)
  trains stably but its embeddings are linearly uninformative on synthetic
  (slow-kept 0.456 vs Reg 0.776) and its low leak is vacuous (z_fast carries
  u at R²≈0). Metrics disagree → keep Reg. Three-point evidence
  (reg+ema / nce+ema / nce+online): the legacy `cpc_*` cell is nce+ema and
  works (0.810, z_fast u 0.857) — so the mechanism needs a **stabilized
  latent target**, the loss form is secondary. No real-data NCE re-runs.
- **D5** Cut order (approved): NCE d_slow → coherence sweep (keep reject
  demo) → HI → HSIC → TS2Vec (only if the classical baseline is answered).
- **D6** Near-term deliverable: **Aug 7, 23:59 — IEEE 8pp course paper
  (Responsible-AI course).** Workshop version extends afterwards.
- **D7** vfloor rule: RankMe collapse at `vfloor=0` → the term enters the
  paper's loss equation; no collapse → default off and remove.
  **RESOLVED 2026-08-03 → removed.** No collapse at vfloor=0 (RankMe rises
  13.8→22, i.e. *higher* effective rank without it); the term was not
  neutral — it also carried the s0 u-leak outlier (0.50→−0.09) that widened
  the error bars. `vfloor=` flag now default 0 in ALL trainers; main
  synthetic matrix = vf0 cells; vf1 cells demoted to the appendix D7
  ablation. The term is dropped from the paper's loss equation. (Origin:
  entered with the initial commit `d741957`, no dedicated rationale.)

## 4. Hard rules for agents (manager and subagents alike)

1. **Document hierarchy:** execute `docs/CHECKLIST_v3.md`; `CHECKLIST_v2.md`
   is background/backlog; v3 wins on conflict. Do not start a v2 item unless
   v3 names it.
2. **⛔ Stop-the-line:** (a) the screen's Check 1 rejects the synthetic
   testbed; (b) NCE-online training is unstable under D2. Halt, report, wait.
3. **Never fabricate or hand-transcribe numbers.** Every figure/table number
   regenerates from `runs/*.json` via `figures_paper.py`. This paper has two
   prior figure↔text drift incidents.
4. **Do not delete or re-run the old `cpc_*` runs** — after the P0-3 stem
   split they are exactly the `nce+ema` control cells.
5. **Honest reporting in the paper:** unfinished work is marked "in
   progress"; claims without a backing run tag are deleted or demoted. No
   placeholder numbers, ever.
6. No repo-wide renames or large refactors during the sprint (paper text
   adopts HGLP naming; code rename is post-deadline backlog).

## 5. Domain facts agents must not re-derive wrongly

- **Backbone attribution (re-verified against the papers 2026-08-03; the
  earlier "point target" phrasing was imprecise and is corrected here with
  user approval):**
  - **v1 (what the old code did):** the target was the causal encoder's
    output *read at* t+Δ — i.e. a **cumulative summary of x[0 … t+Δ]**, which
    re-contains the anchor's own past x[0…t]. Calling that a "point target"
    hid the receptive field. Correct phrasing: *a causal cumulative summary
    starting at 0, read at t+Δ*.
  - **v2 (decided):** **receptive-field-bounded point target** —
    `TargetEncoder(x[t+Δ−w_eff : t+Δ])` read at the last position, with
    `w_eff = min(w, Δ)` so the target's receptive field always lies inside
    `(t, t+Δ]` and never re-contains the anchor's past. This is what makes
    harmless-to-close hold *by construction* rather than by hope.
  - **CPC** [van den Oord 2018]: target = `z_{t+k}` from a **local** encoder
    `g_enc`; the cumulative context `c_t` appears only on the input side. So
    CPC also excludes the anchor's past from the target.
  - **HEPA**: interval-summary target over `(t, t+Δ]` (bidirectional +
    attention pooling), weight-shared jointly-trained target (no EMA, no
    stop-grad), SIGReg, L1, **sinusoidal absolute positions**, and **scalar
    (continuous) Δ conditioning** — so continuous horizon conditioning is a
    standard choice, not our contribution.
  - **LeNEPA**: next-step (t+1) latent target, no horizon variable.
  - **Never write "as in HEPA."** In prose: the full contrast lives in
    Related Works (1–2 sentences); Method keeps exactly one sentence on
    target shape. The live distinction vs HEPA is **"read one position inside
    the future interval (ours) vs attention-pool the whole interval (HEPA)"**
    — pooling mixes in near-future fast content, which is what weakens the
    harmless-to-close argument.
- **Why long-horizon prediction works at all:** the loss is a conditional
  expectation; the predictor cannot foresee regime switches — it can only
  lower error by encoding the current state. `z_slow` is a memo about NOW,
  not a guess about the future. (Reviewers misread this by default; the
  paper carries a 2–3 sentence paragraph on it.)
- **"Slow" is shorthand for long-horizon-predictable (persistent).**
  Persistence = slowness only under the mixing-fast assumption. Coherent
  periodic components are infinitely persistent without being slow — that is
  the screen's Check 1 target. Static factors are the infinite-persistence
  *constant* limit: they ride into `z_slow` by design (that is a feature;
  PTB-XL's diagnosis is the example), and the screen does not need to test
  for them (window-constants vanish from the ACF via mean subtraction).
- **Applicability regimes:** dwell ≈ Δ_max → separates slow (+static)
  [HAPT, synthetic]; dwell ≫ Δ_max but varies across sequences → separates
  static [PTB-XL]; invariant across the corpus → out of scope [normal-only
  settings]. XJTU-SY = the predicted negative (no timescale gap).
- **Third applicability axis — window-level factor contamination (added
  2026-08-03):** SEPARATE from dwell-vs-Δ_max. "Does the window straddle
  factor transitions?" HAPT prep cuts on a fixed stride ignoring activity
  boundaries and labels each window by the END-POINT sample (`per[end]`), so
  57% of windows straddle a transition even at L=128 — the slow factor is
  neither constant within the window nor fully observable. PTB-XL (per-record
  constant diagnosis) and XJTU (per-snapshot life fraction, ~constant over
  80 ms) have transition-fraction 0. HAPT is thus NOT "static like PTB-XL" —
  it is its own category. This axis gets its own applicability-table row.
  Regenerable table: `datasets_table.py` → `runs/datasets_table.json`.
- **HAPT activity-F1 is partly an end-region readout, not proof of slow
  integration (probe 2026-08-03, `hapt_anchor_probe.py`):** (a) z_slow F1
  rises monotonically with anchor position (0.57 near start → 0.71 at end) —
  context accumulates, but the causal encoder confounds "more context" with
  "closer to the labeled end-point sample"; (b) contaminated windows score
  AS WELL or better than clean (0.78 vs 0.74) — the label survives without a
  constant slow factor. Verdict: the F1 MAGNITUDE is not clean evidence of
  window-spanning slow-factor integration; the clean-subset F1 (0.74) is the
  defensible figure, and the paper must state this. The block SEPARATION
  claim (activity concentrates in z_slow, fast leak in z_fast) is separate
  and still holds.
- **Separation is one-directional by design:** `z_fast` retaining regime
  info (≈0.57) is expected, not a bug.
- **Horizon configuration governs gating (open question, 2026-08-03):** the
  operative quantity is the COUNT of trained horizons BEYOND τ — only those
  close the gate and give z_slow learning signal (synthetic 2-of-5; HAPT
  0-of-6 at auto-τ, hence its monotone preference for smaller τ). Ratio
  thresholds (Δ_max/fast, Δ_max/dwell) are post-hoc observations, NOT design
  criteria — never present a ratio as a hypothesis. The Δ SET itself is
  underived (synthetic {1,4,16,64,128} vs real {1,2,4,8,16,32}, no recorded
  justification) — same gap class as the old hardcoded τ. Draft rule (not
  implemented): log-span T_ac(min)→T_ac(max) with ≥2 horizons each side of τ;
  HAPT cannot satisfy it (segments shorter than the required window — a real
  applicability limit). Detail + numbers in `docs/tau.md`.
- **Screen subtleties:** `screen()` currently collapses windows to patch
  means (`W.mean(-1)`) — an implicit 8-sample low-pass. Periodicity tests
  must run on sample-level signals or short-period confounds are invisible.
  The datagen self-test asserts regime-variance ratio < 1.2; the
  variance-only mode violates it by design → the assert must be
  mode-conditional.
- **Checkpoint reuse:** `runs/model_*.pt` and `runs/emb_*.npz` exist; shift
  eval (P3-4) and label efficiency (P3-5) run at the evaluation layer with
  zero retraining.
- **Hardware:** one dedicated H200. A full matrix is an overnight batch;
  compute-bound cuts are void — implementation time is the binding
  constraint.

## 6. Paper framing constants (for any writing task)

- Thesis sentence: horizon is a persistence supervision axis already free in
  any predictive learner [predictive-information lineage: Bialek 2001,
  Creutzig 2009]; under the standard slow-fast mixing assumption
  [cf. Pavliotis & Stuart], persistence coincides with slowness; we show
  **empirically** that gating this axis separates the slow factor into a
  linear subspace, label-free **in training** (validation, as standard in
  disentanglement, is not).
- "Free" applies to the horizon variable, never to the separation itself
  (d_slow, λ are label-tuned; HAPT shows a −4pt downstream cost).
- Aug-7 course version additionally carries a "Relevance to Responsible AI"
  subsection (~0.25p): interpretability by design (measured via the
  block×factor matrix) · responsible scoping (screen + predicted negative +
  normal-only exclusion) · reliability (shift result) · epistemic honesty
  (Exclusion's non-guarantee, linear-level limitation).
- Limitations order: linear-level exclusion → unsupervised model selection →
  xcov polarity → two-timescale assumption (scaling last; the width sweep
  defends it).
- Standing note: the variance-only regime experiment gets exactly one result
  sentence in the screen/Method prose — "separation confirmed or not under
  variance-only regimes + whether the stems differ"; the mean-only theory
  footnote is demoted to the appendix or dropped depending on that result.
- **Variance-only result (2026-08-03):** separation confirmed for HGLP-Reg
  (regime 0.721±0.042) — the mean-only condition is a theoretical boundary,
  not an empirical barrier; theory footnote → appendix. State explicitly that
  NCE-online also fails here but fails on standard regimes too, so this is
  NOT evidence about conditional-mean estimators.
- **Related Works — horizon-set gap (2026-08-03):** CPC [van den Oord 2018]
  predicts k=1..K consecutive steps with a separate head per k, K set by hand
  per domain, no derivation; we use a log-spaced subset with a single
  Δ-conditioned predictor. There is no established procedure in the literature
  for choosing the horizon set — that gap is part of what the screen
  contributes (ties to the Δ-set open question in §5).

## 7. Reporting protocol

At the end of each working day, report: (1) results secured (run tags),
(2) sentences the paper can now claim, (3) anything that hit a ⛔ or deviated
from D1–D7 (deviation requires the user's explicit approval — never
self-approve).
