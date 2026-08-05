"""How hard is the synthetic regime task, and can we make it hard enough to measure?

Motivation. On the default settings every cell scores slow-kept ~0.99 -- the
gated model, the gate-only model, and the no-mechanism control alike. A metric
everything saturates cannot discriminate, so "z_slow keeps the slow factor" was
carrying no evidence on synthetic. A training-free FFT baseline reaches ~0.80,
confirming the task is nearly solved before learning starts.

This calibrates difficulty with the classical baseline. The regime is the carrier
frequency, spread +-`gap` around 0.100 cyc/sample. Two quantities set the floor:

  FFT resolution   with C patches of context the bin width is 1/(C*P), so
                   regimes closer than that are unresolvable in one window
  fast-factor smear u modulates instantaneous frequency by +-FREQ_MOD, which at
                   the default gap=0.05 is exactly as large as the regime spacing

Picking a gap where the classical baseline sits near chance restores headroom, so
inclusion (does z_slow keep the slow factor?) becomes measurable again.

Usage: python3 difficulty.py [fft|train]
Writes runs_v2/difficulty.json, runs_v2/fig_fft_spectrum.png
"""
import json
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hglp"))

import numpy as np

from datagen import make_dataset, REGIME_FREQ, FREQ_MOD, _freqs
from model import P, L

GAPS = [0.05, 0.03, 0.02, 0.01, 0.005]
CONTEXTS = [64, 128]                      # patches of causal context


def windows(gap, context, n=3000, seed=7):
    """Causal context windows ending at a labelled position, + the regime there."""
    d = make_dataset(400_000, seed=seed, gap=gap)
    x, s = d["x"], d["s"]
    n_s = context * P
    rng = np.random.default_rng(seed)
    end = rng.integers(n_s, len(x) - 1, n)
    idx = end[:, None] - np.arange(n_s)[None, ::-1]
    return x[idx], s[end]


def spectra(W, zpad=8):
    """One-sided power spectrum per window and the frequency axis (cyc/sample).

    No taper and 8x zero-padding: the signal is a near-pure tone, so leakage is
    not the problem and a Hann window only widens the main lobe (measured: it
    costs the baseline 8 points). Zero-padding interpolates the peak onto a finer
    grid. Both choices make the classical baseline STRONGER, which is the point --
    it is the bar our learned representation has to clear."""
    A = np.abs(np.fft.rfft(W, n=W.shape[1] * zpad, axis=1)) ** 2
    return A, np.fft.rfftfreq(W.shape[1] * zpad)


def fft_baseline(gap, context, band=(0.06, 0.14)):
    """Training-free: estimate each window's carrier, assign to the nearest regime
    centre. Two standard estimators; the baseline gets the better of the two."""
    W, y = windows(gap, context)
    A, f = spectra(W)
    m = (f >= band[0]) & (f <= band[1])
    Ab, fb = A[:, m], f[m]
    est = {"centroid": (Ab * fb).sum(1) / Ab.sum(1), "peak": fb[Ab.argmax(1)]}
    centres = _freqs(gap)
    out = {}
    for k, e in est.items():
        pred = np.abs(e[:, None] - centres[None]).argmin(1)
        out[k] = float((pred == y).mean())
    out["best"] = max(out["centroid"], out["peak"])
    out["bin_width"] = float(1.0 / (context * P))
    out["regime_spacing"] = float(centres[1] - centres[0])
    out["smear"] = float(FREQ_MOD * centres[1])       # u's frequency modulation
    return out


def figure(path):
    """Why the default task is easy and a narrow gap is not: mean spectrum per
    regime (peaks resolved vs merged) and the per-window estimator distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [0.05, 0.01]
    fig, ax = plt.subplots(2, len(show), figsize=(11, 6.5))
    for j, gap in enumerate(show):
        W, y = windows(gap, 64)
        A, f = spectra(W)
        m = (f >= 0.085) & (f <= 0.115)
        centres = _freqs(gap)
        acc = fft_baseline(gap, 64)
        for r in range(3):
            ax[0, j].plot(f[m], A[y == r][:, m].mean(0), lw=1.4,
                          label=f"regime {r} (f={centres[r]:.4f})")
            ax[0, j].axvline(centres[r], color=f"C{r}", ls=":", lw=0.8)
        ax[0, j].set_title(f"gap = ±{gap*100:g}%   ({'default' if gap == 0.05 else 'narrow'})")
        ax[0, j].set_xlabel("frequency (cycles/sample)"); ax[0, j].set_ylabel("mean power")
        ax[0, j].legend(fontsize=7)
        mb = (f >= 0.06) & (f <= 0.14)
        Ab, fb = A[:, mb], f[mb]
        cen = (Ab * fb).sum(1) / Ab.sum(1)
        for r in range(3):
            ax[1, j].hist(cen[y == r], bins=60, alpha=0.55, label=f"regime {r}")
        for c in centres:
            ax[1, j].axvline(c, color="k", ls=":", lw=0.8)
        ax[1, j].set_title(f"per-window centroid — FFT {acc['best']:.3f} (chance .333)",
                           fontsize=10)
        ax[1, j].set_xlabel("centroid (cycles/sample)"); ax[1, j].set_ylabel("windows")
        ax[1, j].legend(fontsize=7)
    fig.suptitle("Synthetic regime = carrier frequency. 64 patches of context; "
                 "FFT bin width 1/512 = 0.00195 cyc/sample", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"saved {path}")


def train_at(gap, seed=0):
    """Block x factor at a given difficulty, no mechanism vs both mechanisms."""
    from train import train
    from probes import block_factor
    from twosided import synth_feats
    out = {}
    for tag, (g, x) in (("g0_x0", (False, False)), ("g1_x1", (True, True))):
        r = train(loss_kind="nce", target_enc="ema", seed=seed, gate=g, xcov=x,
                  log_every=10 ** 9, gap=gap)
        out[tag] = block_factor(*synth_feats(r["enc"], gap=gap), seed=seed)
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "fft"
    os.makedirs("../../runs_v2", exist_ok=True)
    path = "../../runs_v2/difficulty.json"
    res = json.load(open(path)) if os.path.exists(path) else {}

    if what == "fft":
        print(f"{'gap':>7} {'context':>8} {'bin w':>9} {'spacing':>9} {'smear':>8} "
              f"{'centroid':>9} {'peak':>7}")
        res["fft"] = {}
        for gap in GAPS:
            for c in CONTEXTS:
                r = fft_baseline(gap, c)
                res["fft"][f"gap{gap}_c{c}"] = r
                print(f"{gap:7.3f} {c:8d} {r['bin_width']:9.5f} {r['regime_spacing']:9.5f} "
                      f"{r['smear']:8.5f} {r['centroid']:9.3f} {r['peak']:7.3f}")
        print("\n  chance = 0.333. A gap below the bin width cannot be resolved in one "
              "window;\n  a gap below the smear is masked by the fast factor itself.")
        figure("../../runs_v2/fig_fft_spectrum.png")
    else:
        res["train"] = {}
        for gap in (0.05, 0.01):
            print(f"\n=== gap ±{gap*100:g}% (nce+ema, seed 0) ===")
            res["train"][str(gap)] = t = train_at(gap)
            for tag, bf in t.items():
                print(f"  {tag}: z_slow {bf['z_slow'][0]:.3f}/{bf['z_slow'][1]:+.3f} "
                      f"| z_fast {bf['z_fast'][0]:.3f}/{bf['z_fast'][1]:+.3f} "
                      f"| rand16 {bf['rand16'][0]:.3f}/{bf['rand16'][1]:+.3f}")
    json.dump(res, open(path, "w"), indent=2)
    print(f"saved {path}")
