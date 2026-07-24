"""Reviewer #6: replace the naive low-pass strawman with proper envelope
analysis (Hilbert). Caveat found while doing so: the stored `env` proxy IS
|hilbert| by construction (xjtu_raw_prep.py), so probing Hilbert features
against `env` is circular (R2 ~ 1.0 trivially). The non-circular target is
the health/RUL proxy `life` (normalized lifetime fraction), evaluated with
held-out BEARINGS (group split, matching the paper's protocol).

Usage: python3 hilbert_baseline.py
Compares: Hilbert-envelope features vs low-pass features -> life R2.
(The learned embedding's life R2 comes from raw_am_train.py with probe=life.)
"""
import json

import numpy as np
from scipy.signal import hilbert
from sklearn.linear_model import Ridge

from raw_am_train import load, L, PATCH

data = load()
keys = sorted(data)
test_bearings = keys[::3]                       # every 3rd bearing held out
res = {"test_bearings": test_bearings}

def feats(W):
    flat = W.reshape(len(W), -1)
    amp = np.abs(hilbert(flat, axis=1))
    hil = amp.reshape(len(W), L, PATCH).mean(2)                      # envelope feats
    lp = np.stack([np.fft.irfft(np.fft.rfft(w)[:L // 8], n=L * PATCH)
                   .reshape(L, PATCH).mean(1) for w in flat])        # low-pass feats
    return hil, lp

Htr, Ltr, ytr, Hte, Lte, yte = [], [], [], [], [], []
for b in keys:
    h, lp = feats(data[b]["W"])
    y = data[b]["life"]
    (Hte if b in test_bearings else Htr).append(h)
    (Lte if b in test_bearings else Ltr).append(lp)
    (yte if b in test_bearings else ytr).append(y)
Htr, Hte = np.concatenate(Htr), np.concatenate(Hte)
Ltr, Lte = np.concatenate(Ltr), np.concatenate(Lte)
ytr, yte = np.concatenate(ytr), np.concatenate(yte)

res["hilbert->life_r2"] = float(Ridge().fit(Htr, ytr).score(Hte, yte))
res["lowpass->life_r2"] = float(Ridge().fit(Ltr, ytr).score(Lte, yte))
print(json.dumps(res, indent=2))
json.dump(res, open("runs_am/hilbert_baseline.json", "w"), indent=2)
