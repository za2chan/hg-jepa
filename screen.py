"""Pre-training screening: is a time series a candidate for HC-NEPA?

The slow factor is never the raw signal value -- it is a slowly-varying
DESCRIPTOR of the signal (its energy, its frequency, ...). So we measure
timescales on descriptors, not on the raw samples. Three label-free checks
on raw windows (N, L, C):

  C1 two-timescale coexistence: tau_fast = ACF decay of the raw signal;
     tau_slow = slowest ACF decay among local descriptors (energy, ZCR).
     gap = tau_slow / tau_fast. >= ~5 => two scales coexist.
  C2 long-horizon predictability: can early descriptors predict a late
     descriptor better than a time-shuffled control? (is the slow factor a
     real signal, not a random-walk tail)
  C3 window adequacy: tau_fast < tau_slow < L (slow factor varies within,
     but is slower than the fast one).

Run on our four datasets to test whether the screen predicts observed
separation success (HAPT/synthetic strong, PTB-XL weak, XJTU none).
"""
import json

import numpy as np
from sklearn.linear_model import Ridge


def acf_decay(x, max_lag):
    x = x - x.mean()
    v = np.dot(x, x)
    if v < 1e-9:
        return 1.0
    for k in range(1, max_lag):
        if np.dot(x[:-k], x[k:]) / v < 1 / np.e:
            return float(k)
    return float(max_lag)


def local_descriptors(sig, w):
    """Sliding-window energy (amplitude factor) and zero-crossing rate
    (frequency factor), sampled every w//2. Returns (T', 2)."""
    step = max(w // 2, 1)
    idx = range(0, len(sig) - w, step)
    en = np.array([np.log(np.var(sig[i:i + w]) + 1e-6) for i in idx])
    zcr = np.array([np.mean(np.abs(np.diff(np.sign(sig[i:i + w])))) for i in idx])
    return np.stack([en, zcr], 1)


def screen(name, W):
    L = W.shape[1]
    raw = W.reshape(len(W), L, -1).mean(-1)               # (N, L) collapse channels
    # tau_fast on raw signal
    tau_fast = np.median([acf_decay(w, min(L // 2, 64)) for w in raw[:300]])
    w = int(max(4, 3 * tau_fast))
    # descriptors per window
    D = np.stack([local_descriptors(r, w) for r in raw[:400]])   # (n, T', 2)
    Tp = D.shape[1]
    # tau_slow = slowest descriptor ACF (in descriptor steps -> raw patches)
    step = max(w // 2, 1)
    taus = []
    for c in range(D.shape[2]):
        dt = np.median([acf_decay(D[i, :, c], min(Tp - 1, 40)) for i in range(len(D))])
        taus.append(dt * step)                            # convert to raw-sample units
    tau_slow = max(taus)
    gap = tau_slow / max(tau_fast, 1e-6)

    # C2: early descriptors -> late descriptor, vs time-shuffle
    half = Tp // 2
    X = D[:, :half, :].reshape(len(D), -1)
    y = D[:, -1, 0]                                        # late energy descriptor
    n = len(X) // 2
    r2 = Ridge().fit(X[:n], y[:n]).score(X[n:], y[n:])
    rng = np.random.default_rng(0)
    ysh = y[rng.permutation(len(y))]
    r2s = Ridge().fit(X[:n], ysh[:n]).score(X[n:], ysh[n:])
    lp_gain = r2 - r2s

    c1 = gap >= 5
    c2 = lp_gain > 0.05
    c3 = tau_fast < tau_slow < L
    verdict = "SUITABLE" if (c1 and c2 and c3) else "UNSUITABLE"
    notes = []
    if not c1: notes.append("no scale gap")
    if not c2: notes.append("no predictable slow signal")
    if not c3: notes.append("window mismatch")
    return dict(dataset=name, tau_fast=round(float(tau_fast), 1),
                tau_slow=round(float(tau_slow), 1), scale_gap=round(float(gap), 1),
                longpred_gain=round(float(lp_gain), 3), C1=bool(c1), C2=bool(c2),
                C3=bool(c3), verdict=verdict, note="; ".join(notes) or "ok")


def load_windows():
    out = {}
    out["HAPT"] = np.load("data/hapt.npz")["W"]
    out["PTB-XL"] = np.load("data/ptbxl.npz")["W"]
    d = np.load("data/xjtu_raw.npz")
    bs = sorted({k.split("__")[0] for k in d.files})
    out["XJTU"] = np.concatenate([d[f"{b}__W"] for b in bs])
    from datagen import generate
    x, s, u, phi = generate(300_000, seed=7)
    L, P = 256, 8
    idx = np.arange(0, len(x) - L * P, L * P)[:2000]
    out["Synthetic"] = np.stack([x[i:i + L * P].reshape(L, P) for i in idx])
    return out


if __name__ == "__main__":
    res = [screen(n, W) for n, W in load_windows().items()]
    print(f"{'dataset':10s} {'tau_f':>6} {'tau_s':>7} {'gap':>6} {'LPgain':>7} "
          f"{'C1':>5} {'C2':>5} {'C3':>5}  verdict")
    for r in res:
        print(f"{r['dataset']:10s} {r['tau_fast']:6.1f} {r['tau_slow']:7.1f} "
              f"{r['scale_gap']:6.1f} {r['longpred_gain']:7.3f} "
              f"{str(r['C1']):>5} {str(r['C2']):>5} {str(r['C3']):>5}  "
              f"{r['verdict']} ({r['note']})")
    json.dump(res, open("screen_results.json", "w"), indent=2)
