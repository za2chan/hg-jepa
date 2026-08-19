"""v2 tau estimator (protocol E1/E3): per-axis T_ac, min rule, u-scale
conversion (energy x2), tau = c*chosen, c = ln(1/eps). Measures on the raw
signal up to a window-length lag so it is not window-limited below L (E3).
"""
import numpy as np
from scipy.signal import hilbert

CONVERSION = {"raw": 1.0, "energy": 2.0, "envelope": 1.0}


def _acf(x, K):
    x = np.asarray(x, np.float64); x = x - x.mean(); n = len(x)
    f = np.fft.rfft(x, 2 * n); a = np.fft.irfft(f * np.conj(f))[:K]
    return a / (a[0] + 1e-12)


def _tac_renorm(a, kref):
    an = a / (abs(a[kref]) + 1e-12)
    idx = np.flatnonzero(an[kref:] >= 1 / np.e)
    return float(kref + (idx.max() if len(idx) else 0))


def estimate_tac_multichannel(sig, patch, max_lag_patches):
    """sig: (T, C) continuous multichannel signal. Per axis x per derived
    series T_ac (patch units); chosen = min over all, u-scale converted."""
    kern = np.ones(patch) / patch
    per = {}
    K = min(max_lag_patches * patch, len(sig) // 2)
    kref = patch + 2
    for c in range(sig.shape[1]):
        s = sig[:, c]
        per[f"raw_a{c}"] = (_tac_renorm(np.abs(_acf(s, K)), 1), 1.0)
        en = np.convolve(s ** 2, kern, "valid")
        per[f"energy_a{c}"] = (_tac_renorm(_acf(en, K), kref), 2.0)
        ev = np.convolve(np.abs(hilbert(s)), kern, "valid")
        per[f"env_a{c}"] = (_tac_renorm(_acf(ev, K), kref), 1.0)
    conv = {k: (v[0] / patch) * v[1] for k, v in per.items()}      # u-scale, patch units
    argmin = min(conv, key=conv.get)
    return dict(per_series_uscale={k: round(v, 3) for k, v in conv.items()},
                argmin=argmin, chosen=conv[argmin])


def tau(sig, patch, max_lag_patches, eps=0.05):
    r = estimate_tac_multichannel(sig, patch, max_lag_patches)
    r["c"] = float(np.log(1 / eps)); r["tau"] = r["c"] * r["chosen"]; r["eps"] = eps
    return r
