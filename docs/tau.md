# τ estimation: derivation, rule, and verification (D1 as amended 2026-08-02)

Appendix source for the τ-estimation paragraph. All numbers regenerate via
`python3 tac.py` (synthetic) and `python3 tac_real.py` (real data →
`runs/tac_real.json`).

## What each derived series measures

The fast factor u is latent; T_ac must be read off observable derived
series, each of which distorts u's timescale differently:

- **squared-energy** (patch-smoothed x²): tracks **u²**. Squaring an OU
  process halves its correlation time (corr_{u²}(k) = corr_u(k)², so
  e^{-k/T} → e^{-2k/T}). Its T_ac = T_u/2.
- **envelope** (patch-smoothed |hilbert(x)|): a mixture of u-linear and
  noise terms; T_ac between T_u/2 and T_u.
- **raw**: |ACF| decay measures **carrier phase coherence** (phase
  diffusion driven by u through FM), not u's own lifetime.

Estimator details: ACF renormalized at a short reference lag (removes the
white-noise spike so noise amplitude cannot dilute the crossing), T_ac =
last lag with renormalized |ACF| ≥ 1/e; energy/envelope smoothed over one
patch to remove the carrier. Units: patches.

## The rule

1. `chosen` = min T_ac over {raw, energy, envelope}, **converted to the u
   scale**: energy ×2, raw/envelope ×1. The minimum is a **lower bound**
   on the fast lifetime — every derived series can only see u through a
   distortion that shortens or preserves its timescale, never lengthens it
   (slow contamination lengthens, but the *min* discards those series).
2. A lower bound is the **safe direction**: τ too short means the gate
   closes slightly early and cuts *less* fast content — failure degrades
   toward the ungated control, not toward a false separation.
3. **τ = c · chosen with c = ln(1/ε)**, ε = the residual fast
   autocorrelation tolerated at the gate: for exponential forgetting,
   corr(τ) = ε ⟹ τ = T·ln(1/ε). Default **ε = 0.05 ⟹ c ≈ 3.0** (CLI arg
   `eps=`; `c=` survives only as an appendix-sweep override). The legacy
   hardcoded c = 2.5 was arbitrary and silently absorbed both the decay-
   tail allowance and the u²→u factor; these are now separate and each
   interpretable.

## Synthetic verification (2026-08-02; seeds 7/11/23, deterministic)

| series | T_ac (patches) | conversion | u-scale |
|---|---|---|---|
| raw | 4.4–5.0 | ×1 | 4.4–5.0 |
| **energy (argmin)** | **2.625** | **×2** | **5.25** |
| envelope | 4.125 | ×1 | 4.125 |

- Theory: u² lifetime = U_TAU/2 = 25 steps = **3.125 patches**; measured
  energy T_ac = 2.625 (within the ±25% bar; residual carrier contamination
  biases slightly low).
- chosen = 5.25 patches (theory bar 5.24 ≈ 6.25·(2.625/3.125)); true u
  lifetime 6.25 — chosen is indeed a modest lower bound.
- **τ = 3.0 × 5.25 = 15.7 patches**, landing next to the legacy hardcoded
  τ = 16 — which retroactively explains why 16 worked. Observed, not tuned.
- Day-1 history: the original spec bar of 6.25 patches was a spec error
  (u's own lifetime, unobservable from x); the estimator was right and the
  bar was corrected (D1 amendment, user-approved).

## Real-data property verification (no ground truth → properties)

Patch ↔ seconds mapping (previously missing from the paper):

| dataset | patch | rate | sec/patch |
|---|---|---|---|
| HAPT | 4 samples | 50 Hz | 0.08 s |
| PTB-XL | 10 samples | 100 Hz | 0.10 s |
| XJTU-SY | 4 samples | 25.6 kHz | 0.15625 ms |

**(1) Physical-units sanity** (`runs/tac_real.json`):

| dataset | argmin | chosen | τ (patches) | τ physical | known fast dynamics |
|---|---|---|---|---|---|
| HAPT | raw | 13.25 p | 39.7 | 3.18 s | gait cycle ~1 s — chosen = 1.06 s ≈ one gait cycle ✓ |
| PTB-XL | raw | 18.9 p | 56.6 | 5.66 s | beat ~0.8 s — chosen = 1.89 s ≈ 2.4 beats, same order ✓ |
| XJTU | envelope | 119.8 p | 358.7 | 56 ms | shaft rotation 25–29 ms — chosen = 18.7 ms ≈ 0.7 rotation ✓ |

No estimate is orders of magnitude off; chosen consistently sits at
0.7–2.4× the known fast timescale (lower-bound behavior as designed).

**Horizon-count framing (correction, 2026-08-03).** The operative quantity
is the COUNT of trained horizons lying beyond τ — those are the only ones
where the gate closes and z_slow receives learning signal. Ratio thresholds
(Δ_max/fast, Δ_max/dwell) are post-hoc observations, not design criteria,
and no ratio is presented as a hypothesis. Current state: synthetic
{1,4,16,64,128} at τ=15.7 → **2 of 5** horizons beyond τ; HAPT {1,…,32} at
auto-τ=39.7 → **0 of 6** (the gate never closes; the legacy TAU=5 gave
5 of 6). The τ-sensitivity arm's two statistically identical outside-points
are direct evidence that the count, not the margin, is what matters. The
dwell-side condition (Δ_max ≲ dwell) IS satisfied by HAPT's current config
(2.56 s vs ~12 s) — that was never the problem.

**(2) Per-group stability** (per subject / patient / bearing, τ in patches):

| dataset | n groups | median | IQR | min–max | reading |
|---|---|---|---|---|---|
| HAPT | 30 | 38.2 | 34.5–40.0 | 24.7–48.3 | tight — locking onto a real timescale |
| PTB-XL | 10* | 73.1 | 25.8–108.8 | 0.6–141.0 | wide — but *only 10 patients have ≥3 windows; per-group estimates are under-powered (3×10 s of signal), so spread confounds estimator noise with patient variation. Report as-is. |
| XJTU | 15 | 165.1 | 66.1–333.7 | 19.5–361.7 | wide — consistent with XJTU's known nonstationarity across degradation stages / absent timescale gap (the predicted negative) |

**(3) τ-sensitivity (HAPT, gate+xcov, 3 seeds, τ ∈ {0.5,1,2,4}×auto):**
in the Day-1 overnight batch (`run_matrix.sh`); the claim under test is a
broad plateau containing auto-τ, not optimality. Results land Day 2.

## The Δ SET is also underived (open question, 2026-08-03)

Same class of gap as the hardcoded τ. Synthetic uses {1,4,16,64,128}; the
real-data trainers use {1,2,4,8,16,32}; no recorded justification, no
consistency between them.

Mechanism: horizons INSIDE τ train z_fast (gate open), horizons BEYOND τ
train z_slow (gate closed). So how the Δ set splits around τ sets how much
learning signal each block receives — which is exactly why HAPT with **0
horizons beyond τ** showed a monotone preference for smaller τ (shrinking τ
was the only way to move any horizon into the gate-closed regime).

Draft rule (documented, NOT implemented): span T_ac(min) → T_ac(max)
log-spaced with **at least two horizons on each side of τ**. Check against
existing configs:
- Synthetic, τ=15.7, Δ={1,4,16,64,128}: inside τ = {1,4} (2), beyond =
  {16,64,128} (3) → satisfies (≥2 each side). Fast T_ac(min)=5.25 p,
  slow T_ac(max)≈82 p; the set roughly spans them. ✓
- HAPT, auto-τ=39.7, Δ={1,2,4,8,16,32}: inside = all 6, beyond = 0 →
  fails. Even Option 1 (Δ→{1..64}) gives beyond={64} (1), still short of 2.
  A rule-satisfying HAPT set needs Δ_max ≳ 2× τ ≈ 80 p, i.e. L≥256 — which
  the raw-data audit shows is infeasible (segment length < window). So HAPT
  cannot satisfy the draft rule without longer contiguous activity segments
  than the corpus contains. Record as a genuine applicability limit.

## Backlog (workshop version)

Model-based τ estimation — assumption-free alternative reading τ off the
model's horizon-wise error curve where the fast block stops contributing;
compare against the ε-derived τ above. (CHECKLIST_v2 P4-3.)
