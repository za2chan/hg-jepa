"""Two-timescale synthetic data for the horizon-conditioned NEPA pilot.

Generative factors (recorded as ground truth):
  slow s(t): 3-state Markov regime, mean dwell ~DWELL steps.
             Each regime sets the oscillator's carrier freq and amplitude.
  fast u(t): OU process with short memory (~U_TAU steps), modulates
             instantaneous frequency and adds a small direct component.

Observation:
  x(t) = amp[s]*sin(phi(t)) + 0.2*u(t) + 0.1*eps
  phi'(t) = 2*pi*freq[s]*(1 + 0.3*u(t))
"""
import numpy as np

REGIME_FREQ = np.array([0.095, 0.100, 0.105])  # hard: +-5% gap ~ u-induced jitter
REGIME_AMP = np.array([1.0, 1.0, 1.0])       # hard: no amplitude cue
DWELL = 3000        # mean regime dwell (steps)
U_TAU = 50          # OU correlation time (steps) = fast-factor lifetime
FREQ_MOD = 0.05     # frequency modulation depth (keeps phase predictable ~1 lifetime)


def generate(n_steps, seed=0, freq_mult=None):
    """freq_mult: optional per-step multiplier on instantaneous frequency
    (used by anomaly.py to inject contextual anomalies)."""
    rng = np.random.default_rng(seed)
    # slow factor: Markov regime
    s = np.empty(n_steps, dtype=np.int64)
    cur = rng.integers(3)
    switches = rng.random(n_steps) < 1.0 / DWELL
    for t in range(n_steps):
        if switches[t]:
            cur = (cur + rng.integers(1, 3)) % 3
        s[t] = cur
    # fast factor: OU process
    u = np.empty(n_steps)
    u[0] = rng.standard_normal()
    a = np.exp(-1.0 / U_TAU)
    noise = rng.standard_normal(n_steps) * np.sqrt(1 - a * a)
    for t in range(1, n_steps):
        u[t] = a * u[t - 1] + noise[t]
    # observation
    fm = 1.0 if freq_mult is None else freq_mult
    phi = np.cumsum(2 * np.pi * REGIME_FREQ[s] * (1 + FREQ_MOD * u) * fm)
    x = REGIME_AMP[s] * np.sin(phi) + 0.2 * u + 0.1 * rng.standard_normal(n_steps)
    return x.astype(np.float32), s, u.astype(np.float32), phi.astype(np.float32)


def make_dataset(n_steps=1_000_000, seed=0, path=None):
    x, s, u, phi = generate(n_steps, seed)
    d = dict(x=x, s=s, u=u, sin_phi=np.sin(phi).astype(np.float32),
             cos_phi=np.cos(phi).astype(np.float32))
    if path:
        np.savez_compressed(path, **d)
    return d


if __name__ == "__main__":
    x, s, u, phi = generate(200_000, seed=1)
    # regime dwell sanity: mean dwell within 3x of target
    changes = np.flatnonzero(np.diff(s)) + 1
    dwells = np.diff(np.concatenate([[0], changes, [len(s)]]))
    assert DWELL / 3 < dwells.mean() < DWELL * 3, dwells.mean()
    # fast factor forgets itself: ACF(u) drops below 1/e near U_TAU
    ac = np.array([np.corrcoef(u[:-k], u[k:])[0, 1] for k in (1, U_TAU, 5 * U_TAU)])
    assert ac[0] > 0.9 and ac[1] < 0.6 and abs(ac[2]) < 0.1, ac
    # signal ACF decays fast (fast dynamics dominate raw signal)
    acx = np.array([np.corrcoef(x[:-k], x[k:])[0, 1] for k in (1, 50, 500)])
    assert abs(acx[2]) < 0.15, acx
    # hard mode: regime must NOT be readable from short-window variance
    var_by_regime = [x[s == r].var() for r in range(3)]
    assert max(var_by_regime) / min(var_by_regime) < 1.2, var_by_regime
    print("dwell mean:", dwells.mean(), "| ACF(u):", np.round(ac, 3),
          "| ACF(x):", np.round(acx, 3), "| var/regime:", np.round(var_by_regime, 3))
    print("OK")
