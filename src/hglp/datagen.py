"""Synthetic two-timescale data for HGLP v2 (protocol A5).

Slow s(t): 3-state Markov regime (mean dwell DWELL). Fast u(t): OU process
(lifetime U_TAU). Deterministic given a seed. Persists to disk with a sha256
content hash so every run names the exact bytes it trained on.
"""
import hashlib

import numpy as np

REGIME_FREQ = np.array([0.095, 0.100, 0.105])   # +-5% carrier gap
REGIME_AMP = np.array([1.0, 1.0, 1.0])
REGIME_USCALE = np.array([0.25, 1.0, 2.5])       # variance-mode OU scale
DWELL = 3000
U_TAU = 50
FREQ_MOD = 0.05


def generate(n_steps, seed=0, regime_mode="freq"):
    rng = np.random.default_rng(seed)
    s = np.empty(n_steps, dtype=np.int64)
    cur = rng.integers(3)
    switches = rng.random(n_steps) < 1.0 / DWELL
    for t in range(n_steps):
        if switches[t]:
            cur = (cur + rng.integers(1, 3)) % 3
        s[t] = cur
    u = np.empty(n_steps)
    u[0] = rng.standard_normal()
    a = np.exp(-1.0 / U_TAU)
    noise = rng.standard_normal(n_steps) * np.sqrt(1 - a * a)
    for t in range(1, n_steps):
        u[t] = a * u[t - 1] + noise[t]
    if regime_mode == "variance":
        u_eff = REGIME_USCALE[s] * u
        freq = np.full(n_steps, REGIME_FREQ[1])
    else:
        u_eff, freq = u, REGIME_FREQ[s]
    phi = np.cumsum(2 * np.pi * freq * (1 + FREQ_MOD * u_eff))
    x = REGIME_AMP[s] * np.sin(phi) + 0.2 * u_eff + 0.1 * rng.standard_normal(n_steps)
    return (x.astype(np.float32), s, u.astype(np.float32), phi.astype(np.float32))


def make_dataset(n_steps=1_000_000, seed=0, regime_mode="freq", path=None):
    x, s, u, phi = generate(n_steps, seed, regime_mode)
    d = dict(x=x, s=s, u=u, sin_phi=np.sin(phi).astype(np.float32),
             cos_phi=np.cos(phi).astype(np.float32))
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode()); h.update(np.ascontiguousarray(d[k]).tobytes())
    d["sha256"] = h.hexdigest()[:16]
    if path:
        np.savez_compressed(path, **d)
    return d


if __name__ == "__main__":
    d = make_dataset(200_000, seed=1)
    x, s, u = d["x"], d["s"], d["u"]
    ac = np.array([np.corrcoef(u[:-k], u[k:])[0, 1] for k in (1, U_TAU, 5 * U_TAU)])
    assert ac[0] > 0.9 and ac[1] < 0.6 and abs(ac[2]) < 0.1, ac
    # determinism: same seed -> same hash
    assert make_dataset(200_000, seed=1)["sha256"] == d["sha256"]
    print("hash", d["sha256"], "| ACF(u)", np.round(ac, 3), "OK")
