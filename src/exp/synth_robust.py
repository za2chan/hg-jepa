"""Does z_slow help ONLY when the corruption targets the fast factor?

On real data we can only ARGUE which factor a perturbation hits: PTB-XL gain looks
like a fast attack (amplitude changes, waveform survives), PTB-XL baseline wander
and HAPT amplitude scaling look like slow attacks (wander IS slow; on HAPT the
absolute gravity level IS posture). The measured signs matched that reading
(+0.101 / -0.037 / -0.020..-0.138), but the reading is post-hoc.

Synthetic removes the argument. We own the generator

    x(t) = sin(phi(t)) + 0.2*u(t) + 0.1*eps(t)
    phi  = cumsum(2*pi * f[s(t)] * (1 + FREQ_MOD*u(t)))

so a corruption can be BUILT to move exactly one factor:

  fast attack  u  -> sqrt(1-a^2)*u + a*u'      (u' an independent OU draw)
  slow attack  f  -> f * (1 + b*w)             (w a slow OU, timescale = DWELL)

The two attacks are NOT put on a common magnitude scale, because no honest one
exists. Matching by signal RMS -- the first thing I tried -- is badly wrong: phase
accumulates, so a carrier drift of 0.0007% decorrelates the waveform enough to
reach RMS 0.4 while moving the local frequency by 1/250 of the regime spacing.
Both attacks then looked huge and changed nothing, because the information the
probe reads was untouched. Waveform correlation fails for the same reason.

So each attack is swept on its OWN natural axis --

  fast   the FRACTION of u replaced           (0.25 .. 1.0, saturating)
  slow   carrier wander in REGIME SPACINGS    (0.25 .. 2.0)

-- and the claim is about the SIGN of (z_slow - z_full) WITHIN each attack, never
about comparing a fast strength to a slow one.

Run at gap=0.01: at the default +-5% every cell scores 0.99 and there is no
headroom for any corruption to show up.

Prediction, from the rule the real data suggested:
  fast attack -> z_slow degrades LESS than z_full
  slow attack -> z_slow degrades MORE
A failure of the first would say the central claim does not hold even under full
control, which is worth knowing before it is written down.

Usage: python3 synth_robust.py [n_seed] [steps]
Writes runs_v2/synth_robust.json
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hglp"))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from datagen import make_dataset, _freqs, FREQ_MOD, U_TAU, DWELL
from model import P, L, D_SLOW, D_Z
from probes import rand_subspace
from train import train, DEV

STEMS = ["l1+ema", "nce+ema"]
FAST_LV = (0.25, 0.5, 0.75, 1.0)   # fraction of u replaced (1.0 = fully independent)
SLOW_LV = (0.25, 0.5, 1.0, 2.0)    # carrier wander, in regime spacings
GAP = 0.01                       # hard setting; at 0.05 nothing has room to degrade
POS = tuple(range(64, 240, 12))
N_WIN, DATA_SEED = 400, 99


def _ou(n, tau, rng):
    """Unit-variance OU with the given lifetime — the generator's own fast process."""
    a = np.exp(-1.0 / tau)
    e = rng.standard_normal(n) * np.sqrt(1 - a * a)
    out = np.empty(n); out[0] = rng.standard_normal()
    for t in range(1, n):
        out[t] = a * out[t - 1] + e[t]
    return out


def rebuild(d, eps, gap=0.05, u_mix=0.0, u_alt=None, f_drift=0.0, w=None):
    """Regenerate x with exactly one factor moved, from the stored phase.

    Re-accumulating phi from scratch does not work: `u` is stored as float32, and
    cumsum over 300k steps turns that rounding into a phase drift big enough to
    decorrelate sin(phi). Instead accumulate only the phase DIFFERENCE the attack
    causes and rotate the stored phase by it, which is exact when the attack is
    zero and well-conditioned when it is not."""
    s, u = d["s"], d["u"].astype(np.float64)
    ue = u if u_mix == 0 else np.sqrt(1 - u_mix ** 2) * u + u_mix * u_alt
    f0 = _freqs(gap)[s]
    f1 = f0 * (1.0 + f_drift * w) if f_drift else f0
    dphi = np.cumsum(2 * np.pi * (f1 * (1 + FREQ_MOD * ue) - f0 * (1 + FREQ_MOD * u)))
    sin_new = d["sin_phi"] * np.cos(dphi) + d["cos_phi"] * np.sin(dphi)
    return (sin_new + 0.2 * ue + 0.1 * eps).astype(np.float32)


def corrupt_series(n=300_000, gap=GAP):
    """Clean series plus the two attack families, matched in FACTOR space."""
    d = make_dataset(n, seed=DATA_SEED, gap=gap)
    x, s, u = d["x"].astype(np.float64), d["s"], d["u"].astype(np.float64)
    eps = (x - d["sin_phi"] - 0.2 * u) / 0.1          # exact: sin_phi is stored
    assert np.allclose(rebuild(d, eps, gap=gap), x, atol=1e-5), "reconstruction mismatch"
    rng = np.random.default_rng(1234)
    u_alt, w = _ou(n, U_TAU, rng), _ou(n, DWELL, rng)
    fr = _freqs(gap)
    spacing = (fr[1] - fr[0]) / fr[1]                 # regime gap as a fractional shift
    rms = lambda y: float(np.sqrt(np.mean((y - x) ** 2)))
    out = {}
    for lv in FAST_LV:
        out[("fast", lv)] = (rebuild(d, eps, gap, u_mix=lv, u_alt=u_alt), lv, 0.0)
    for lv in SLOW_LV:
        y = rebuild(d, eps, gap, f_drift=lv * spacing, w=w)
        out[("slow", lv)] = (y, lv, rms(y))
    return x.astype(np.float32), s, out


@torch.no_grad()
def encode_windows(enc, xs, starts):
    xb = torch.from_numpy(np.stack([xs[st:st + L * P].reshape(L, P) for st in starts])).to(DEV)
    return torch.cat([enc(xb[i:i + 128]) for i in range(0, len(xb), 128)]).cpu().numpy()


def run(stem, seed, steps):
    lk, te = stem.split("+")
    r = train(loss_kind=lk, target_enc=te, seed=seed, steps=steps,
              gap=GAP, log_every=10 ** 9)
    x, s, attacks = corrupt_series()
    rng = np.random.default_rng(DATA_SEED)
    starts = rng.integers(0, len(x) - L * P - 1, N_WIN)
    y = np.concatenate([s[starts + a * P + (P - 1)] for a in POS])
    wid = np.tile(np.arange(N_WIN), len(POS))
    cut = np.sort(starts)[N_WIN // 2]
    tr = np.flatnonzero(starts[wid] + L * P <= cut)     # window-level cut (no leak)
    te_i = np.flatnonzero(starts[wid] > cut)

    Zc = encode_windows(r["enc"], x, starts)
    Q = rand_subspace(D_Z, D_SLOW, seed)
    arms = {"z_slow": lambda Z: Z[..., :D_SLOW], "z_full": lambda Z: Z,
            "rand16": lambda Z: Z @ Q}
    clf, clean = {}, {}
    for a, f in arms.items():
        F = np.concatenate([f(Zc)[:, p] for p in POS])
        clf[a] = make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=1000,
                                                  class_weight="balanced")).fit(F[tr], y[tr])
        clean[a] = float(f1_score(y[te_i], clf[a].predict(F[te_i]), average="macro"))
    out = {"clean": clean, "attacks": {}}
    for (kind, t), (xp, strength, achieved) in attacks.items():
        Zp = encode_windows(r["enc"], xp, starts)
        cell = {a: float(f1_score(y[te_i], clf[a].predict(
            np.concatenate([f(Zp)[:, p] for p in POS])[te_i]), average="macro"))
            for a, f in arms.items()}
        cell.update(strength=float(strength), rms=float(achieved))
        out["attacks"][f"{kind}@{t}"] = cell
    return out


if __name__ == "__main__":
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
    os.makedirs("../../runs_v2", exist_ok=True)
    res = {}
    for stem in STEMS:
        cells = [run(stem, s, steps) for s in range(n_seed)]
        res[stem] = cells
        print(f"\n=== {stem} ({n_seed} seeds, {steps} steps) ===", flush=True)
        keys = list(cells[0]["attacks"])
        print(f"{'조건':14} {'z_slow':>8} {'z_full':>8} {'rand16':>8} {'slow-full':>10}")
        cl = {a: np.mean([c["clean"][a] for c in cells]) for a in ("z_slow", "z_full", "rand16")}
        print(f"{'clean':14} {cl['z_slow']:8.3f} {cl['z_full']:8.3f} {cl['rand16']:8.3f} "
              f"{cl['z_slow']-cl['z_full']:+10.3f}")
        for k in keys:
            m = {a: np.mean([c["attacks"][k][a] for c in cells])
                 for a in ("z_slow", "z_full", "rand16")}
            print(f"{k:14} {m['z_slow']:8.3f} {m['z_full']:8.3f} {m['rand16']:8.3f} "
                  f"{m['z_slow']-m['z_full']:+10.3f}", flush=True)
    json.dump(dict(result=res, config=dict(n_seed=n_seed, steps=steps,
                                           gap=GAP, fast_levels=list(FAST_LV),
                                           slow_levels=list(SLOW_LV),
                                           matched_in="factor space")),
              open("../../runs_v2/synth_robust.json", "w"), indent=2)
    print("\nsaved runs_v2/synth_robust.json")
