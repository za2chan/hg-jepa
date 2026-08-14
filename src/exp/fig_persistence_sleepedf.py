"""Band accessibility of the persistent factor on Sleep-EDF.

Third companion to fig_persistence_ptbxl.py and fig_persistence_hapt.py. The
question is the same: in which frequency band of the RAW signal does the
persistent factor live? Together the three answer whether "persistence coincides
with spectral slowness" holds at the signal level, which the paper asserted for
real data but never measured.

The two earlier answers disagreed with each other, so this one is not a
formality. PTB-XL's diagnosis is NOT in the slow band (low-pass view at chance)
even though it is perfectly persistent; HAPT's activity IS (low-pass view 0.570
against chance 0.167), because posture is the DC distribution of gravity.

Sleep staging is scored from classical EEG rhythms, so the informative content is
expected in named bands rather than at DC: delta (0.5-4 Hz) for N3, theta
(4-8 Hz) for N1, sigma/spindles (11-16 Hz) for N2, and the EOG channel for REM.
Bands are therefore the standard sleep bands, not the octave grid used for HAPT.

Method matches the companions: filter to one band at a time, reduce to per-patch
features (mean and log-energy per channel), fit a linear probe with a
SUBJECT-disjoint split, score macro-F1 over the five stages (chance 1/5).

python3 src/exp/fig_persistence_sleepedf.py
    -> runs_v2/fig_persistence_sleepedf.{png,json}
"""
import json
import pathlib

import numpy as np
from scipy import signal as sps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "runs_v2"
NPZ = ROOT / "data" / "sleepedf_v2.npz"

FS = 100.0                     # Hz
PATCH, N_AX = 50, 3            # 0.5 s patches; EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal
# Classical sleep bands. lo=0 keeps DC, for symmetry with the HAPT companion --
# on EEG the DC level is not expected to carry stage, and measuring that is the point.
BANDS = [(0.0, 0.5), (0.5, 4.0), (4.0, 8.0), (8.0, 11.0),
         (11.0, 16.0), (16.0, 30.0), (30.0, 49.0)]
BAND_NAME = ["DC-0.5", "delta", "theta", "alpha", "sigma", "beta", "gamma"]
LABEL_RANGE = (1, 5)           # W, N1, N2, N3, REM;  0 = unscored
CHANCE = 1.0 / 5
MAX_TRAIN = 60000
SEED = 0
C0, C1 = "#1a5fb4", "#c01c28"


def unpatch(W):
    """(n, L, PATCH*N_AX) -> (n, N_AX, L*PATCH).

    The prep wrote `x[:, s0:s0+WIN].T.reshape(L, PATCH*N_AX)`, i.e. within a patch
    the channel index varies fastest. Round-tripped in check() below."""
    n, L, _ = W.shape
    x = W.reshape(n, L, PATCH, N_AX)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2).reshape(n, N_AX, L * PATCH))


def bandpass(x, lo, hi):
    ny = FS / 2
    if lo <= 0:
        b, a = sps.butter(3, hi / ny, btype="low")
    elif hi >= ny:
        b, a = sps.butter(3, lo / ny, btype="high")
    else:
        b, a = sps.butter(3, [lo / ny, hi / ny], btype="band")
    return sps.filtfilt(b, a, x, axis=-1)


def patch_feats(xb):
    """Per-patch mean and log-energy per channel: six numbers per patch."""
    n, c, T = xb.shape
    p = xb.reshape(n, c, T // PATCH, PATCH)
    mu = p.mean(-1)
    en = np.log1p((p ** 2).mean(-1))
    return np.concatenate([mu, en], 1).transpose(0, 2, 1)


def probe(F, lab, subj, rng):
    keep = (lab >= LABEL_RANGE[0]) & (lab <= LABEL_RANGE[1])
    subj_w = np.repeat(subj[:, None], lab.shape[1], 1)
    X = F.reshape(-1, F.shape[-1])[keep.ravel()]
    y = lab.ravel()[keep.ravel()]
    g = subj_w.ravel()[keep.ravel()]
    gs = np.unique(g)
    te_g = set(rng.choice(gs, max(1, len(gs) // 3), replace=False).tolist())
    te = np.array([v in te_g for v in g])
    Xtr, ytr = X[~te], y[~te]
    if len(ytr) > MAX_TRAIN:
        idx = rng.choice(len(ytr), MAX_TRAIN, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(Xtr, ytr)
    return float(f1_score(y[te], clf.predict(X[te]), average="macro"))


def check(W):
    """The patch layout is the one assumption that would silently invalidate
    everything, so assert the round-trip rather than trusting the comment."""
    x = unpatch(W[:3])
    back = x.transpose(0, 2, 1).reshape(3, W.shape[1], PATCH * N_AX)
    assert np.allclose(back, W[:3]), "unpatch does not round-trip"


def main():
    d = np.load(NPZ, allow_pickle=True)
    W, lab, subj = d["W"], d["lab"], d["subj"]
    check(W)
    x = unpatch(W)
    print(f"windows {len(W)}  subjects {len(np.unique(subj))}  chance {CHANCE:.3f}")

    res, feats = {}, []
    for (lo, hi), nm in zip(BANDS, BAND_NAME):
        F = patch_feats(bandpass(x, lo, hi))
        feats.append(F)
        s = probe(F, lab, subj, np.random.default_rng(SEED))
        res[f"{nm} ({lo}-{hi})"] = s
        print(f"  {nm:6s} {lo:5.1f}-{hi:4.1f} Hz  macro-F1 {s:.3f}")

    lowpass = res[f"{BAND_NAME[0]} ({BANDS[0][0]}-{BANDS[0][1]})"]
    allband = probe(np.concatenate(feats, -1), lab, subj, np.random.default_rng(SEED))
    print(f"  low-pass only  {lowpass:.3f}\n  all bands      {allband:.3f}")

    out = dict(config=dict(fs=FS, bands=BANDS, band_names=BAND_NAME, chance=CHANCE,
                           seed=SEED, n_windows=int(len(W)),
                           n_subj=int(len(np.unique(subj))),
                           split="subject-disjoint, 1/3 held out"),
               band_f1=res, lowpass_view_f1=lowpass, allband_view_f1=allband)
    (OUT / "fig_persistence_sleepedf.json").write_text(json.dumps(out, indent=1))

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(range(len(BANDS)), list(res.values()), "o-", color=C0, lw=2,
            label="single band")
    ax.axhline(allband, color="k", ls="--", lw=1, label=f"all bands ({allband:.2f})")
    ax.axhline(CHANCE, color=C1, ls=":", lw=1, label=f"chance ({CHANCE:.2f})")
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels(BAND_NAME, fontsize=8)
    ax.set_ylabel("sleep stage macro-F1")
    ax.set_title("Sleep-EDF: which band carries the persistent factor?", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=.3)
    fig.savefig(OUT / "fig_persistence_sleepedf.png", dpi=180, bbox_inches="tight")
    print("  -> runs_v2/fig_persistence_sleepedf.{json,png}")


if __name__ == "__main__":
    main()
