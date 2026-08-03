"""tac.py — data-derived T_ac for the tau anchor (P0-1, rule D1 as amended
2026-08-02; derivation and the Day-1 finding: docs/tau.md).

D1 (amended, user-approved): chosen = min T_ac across the derived series
raw / squared-energy / envelope, CONVERTED TO THE u SCALE — squaring an OU
process halves its correlation time, so the energy series' T_ac is
multiplied by 2 (raw/envelope x1). chosen is a lower-bound estimate of the
fast lifetime (safe direction: tau too short fails toward the ungated
control). tau = c * chosen with c = ln(1/eps), eps = residual fast
autocorrelation tolerated at the gate (default 0.05 => c ~ 3.0); c= is an
appendix-sweep override only. ZCR is reported, excluded from the min.

Estimator: energy and envelope are smoothed over one patch to remove the
carrier, then T_ac = last lag where the ACF renormalized at a short
reference lag stays >= 1/e. Raw uses |ACF| (oscillation is carrier phase,
not forgetting). Units: PATCHES.

Sanity bar on synthetic: energy T_ac ~ 3.125 patches (+-25%; u^2 lifetime
= U_TAU/2 = 25 steps), argmin = energy, chosen ~ 5.24 after conversion,
tau ~ 15.7 — retroactively consistent with the legacy hardcoded 16.

Necessary-condition caveat: slow structure invisible in all three derived
series is missed; the set is deliberately not expanded.
"""
import numpy as np
from scipy.signal import hilbert

# u-scale conversion per series (D1 as amended): squared series forget x2
# faster than u, so their T_ac is doubled; raw/envelope are linear in u.
CONVERSION = {"raw": 1.0, "energy": 2.0, "envelope": 1.0}


def _acf(x, K):
    x = np.asarray(x, np.float64)
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    a = np.fft.irfft(f * np.conj(f))[:K]
    return a / (a[0] + 1e-12)


def _tac_renorm(a, kref):
    """Last lag with ACF/ACF(kref) >= 1/e, in steps."""
    an = a / (abs(a[kref]) + 1e-12)
    idx = np.flatnonzero(an[kref:] >= 1 / np.e)
    return float(kref + (idx.max() if len(idx) else 0))


def _zcr_series(x, w):
    step = max(w // 2, 1)
    idx = range(0, len(x) - w, step)
    return np.array([np.mean(np.abs(np.diff(np.sign(x[i:i + w])))) for i in idx]), step


def estimate_tac(x_or_windows, patch=8, max_lag_patches=128):
    """x_or_windows: 1-D signal or (N, L, P) windows (per-window estimates,
    median-aggregated). Returns per-series T_ac (patch units, unconverted),
    the argmin series, its u-scale conversion factor, and
    chosen = per_series[argmin] * conversion (rule D1 as amended)."""
    W = np.asarray(x_or_windows, dtype=np.float64)
    sigs = [W] if W.ndim == 1 else [w.reshape(-1) for w in W[:300]]
    per = {"raw": [], "energy": [], "envelope": [], "zcr": []}
    kern = np.ones(patch) / patch
    for s in sigs:
        K = min(max_lag_patches * patch, len(s) // 2)
        kref = patch + 2
        a = np.abs(_acf(s, K))                       # raw: coherence, not dips
        per["raw"].append(_tac_renorm(a, 1))
        en = np.convolve(s ** 2, kern, "valid")
        per["energy"].append(_tac_renorm(_acf(en, K), kref))
        ev = np.convolve(np.abs(hilbert(s)), kern, "valid")
        per["envelope"].append(_tac_renorm(_acf(ev, K), kref))
        zs, step = _zcr_series(s, 4 * patch)
        per["zcr"].append(_tac_renorm(_acf(zs, max(4, K // step)), 1) * step)
    per_series = {k: float(np.median(v)) / patch for k, v in per.items()}
    argmin = min(CONVERSION, key=lambda k: per_series[k])
    factor = CONVERSION[argmin]
    # slow scale: the max was being discarded — diagnostic only, all 4 series
    # (capped at max_lag_patches; a hit at the cap means ">= cap")
    argmax = max(per_series, key=per_series.get)
    return {"per_series": per_series, "argmin_series": argmin,
            "conversion_factor": factor,
            "chosen": per_series[argmin] * factor,
            "tac_slow": per_series[argmax], "argmax_series": argmax,
            "rule": ("D1 amended 2026-08-02: min T_ac over {raw, energy, "
                     "envelope} converted to u scale (energy x2); patch "
                     "units; zcr reported only")}


def tau_from_tac(x_or_windows, patch=8, eps=0.05, c=None):
    """tau = c * chosen. c = ln(1/eps) unless an explicit c override is
    given (appendix sweep only)."""
    r = estimate_tac(x_or_windows, patch=patch)
    r["eps"] = None if c is not None else float(eps)
    r["c"] = float(c) if c is not None else float(np.log(1.0 / eps))
    r["tau"] = float(r["c"] * r["chosen"])
    return r


if __name__ == "__main__":
    from datagen import generate
    results = []
    for seed in (7, 11, 23):
        x, s, u, phi = generate(300_000, seed=seed)
        r = tau_from_tac(x, patch=8)
        results.append(r)
        print(seed, {k: round(v, 3) for k, v in r["per_series"].items()},
              f"argmin={r['argmin_series']} x{r['conversion_factor']:g} "
              f"chosen={r['chosen']:.3f} c={r['c']:.3f} tau={r['tau']:.2f}")
    ch = [r["chosen"] for r in results]
    en = [r["per_series"]["energy"] for r in results]
    assert max(ch) - min(ch) < 1.0, ch                     # determinism/stability
    assert all(r["argmin_series"] == "energy" for r in results)
    assert all(3.125 * 0.75 < e < 3.125 * 1.25 for e in en), en  # D1 bar +-25%
    print(f"chosen median {np.median(ch):.2f} patches (bar ~5.24) | "
          f"tau {np.median([r['tau'] for r in results]):.2f} (legacy 16)")
