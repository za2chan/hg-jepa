"""Band accessibility of the persistent factor on HAPT.

Companion to fig_persistence_ptbxl.py, which asked the same question for the ECG
diagnosis. The question is: in which frequency band of the RAW signal does the
persistent factor live? This is the signal-level half of the persistence /
slowness comparison; the embedding-level half is the coincidence index computed
from rotation_*.json.

PTB-XL's answer was that the diagnosis is NOT in the slow band (low-pass view
scores at chance) even though it is perfectly persistent. HAPT should differ:
posture is the distribution of gravity over the three axes, which is a
near-DC quantity, so the slow band is expected to carry it.

Method. The window tensor is un-patched back to a 50 Hz, 3-axis series, filtered
to one band at a time, and reduced to per-patch features (mean and log-energy
per axis, 6 numbers). A linear probe is fit per band with a SUBJECT-disjoint
split, scored by macro-F1 over the six basic activities (chance 1/6).

python3 src/exp/fig_persistence_hapt.py -> runs_v2/fig_persistence_hapt.{png,json}
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
NPZ = ROOT / "data" / "hapt_v2.npz"

FS = 50.0                      # Hz
PATCH, N_AX = 4, 3             # 4 samples x 3 axes = 12 per patch
# Nyquist is 25 Hz. The lowest band is where gravity direction (posture) sits.
# lo=0 means a true LOW-PASS that keeps DC. That matters here and not on the ECG:
# posture is the distribution of gravity over the three axes, a DC quantity, so a
# 0.05 Hz high-pass corner would delete exactly the factor we are looking for.
BANDS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 24.0)]
LABEL_RANGE = (1, 6)           # basic activities only; 0 = VOID, 7-12 = transitions
CHANCE = 1.0 / 6
MAX_TRAIN = 60000              # probe subsample, keeps the fit quick
SEED = 0
C0, C1 = "#1a5fb4", "#c01c28"


def unpatch(W):
    """(n, L, PATCH*N_AX) -> (n, N_AX, L*PATCH) continuous series per window."""
    n, L, _ = W.shape
    x = W.reshape(n, L, PATCH, N_AX)          # prep wrote patch-major, axis-minor
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2).reshape(n, N_AX, L * PATCH))


def bandpass(x, lo, hi):
    ny = FS / 2
    if lo <= 0:
        b, a = sps.butter(3, hi / ny, btype="low")
        return sps.filtfilt(b, a, x, axis=-1)
    if hi >= ny:
        b, a = sps.butter(3, lo / ny, btype="high")
    else:
        b, a = sps.butter(3, [lo / ny, hi / ny], btype="band")
    return sps.filtfilt(b, a, x, axis=-1)


def patch_feats(xb):
    """Band-limited series -> per-patch features: mean and log-energy per axis.

    Mean carries the near-DC content (gravity direction, i.e. posture); energy
    carries the motion intensity. Six numbers per patch."""
    n, c, T = xb.shape
    p = xb.reshape(n, c, T // PATCH, PATCH)
    mu = p.mean(-1)                                    # (n, c, L)
    en = np.log1p((p ** 2).mean(-1))                   # (n, c, L)
    return np.concatenate([mu, en], 1).transpose(0, 2, 1)   # (n, L, 2c)


def probe(F, lab, subj, rng):
    """Subject-disjoint linear probe, macro-F1 over the scored activities."""
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
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, multi_class="multinomial"))
    clf.fit(Xtr, ytr)
    return float(f1_score(y[te], clf.predict(X[te]), average="macro"))


def main():
    d = np.load(NPZ, allow_pickle=True)
    W, lab, subj = d["W"], d["lab"], d["subj"]
    x = unpatch(W)
    rng = np.random.default_rng(SEED)
    print(f"windows {len(W)}  subjects {len(np.unique(subj))}  chance {CHANCE:.3f}")

    res, feats = {}, []
    for lo, hi in BANDS:
        F = patch_feats(bandpass(x, lo, hi))
        feats.append(F)
        s = probe(F, lab, subj, np.random.default_rng(SEED))
        res[f"{lo}-{hi}"] = s
        print(f"  band {lo:5.2f}-{hi:5.1f} Hz  macro-F1 {s:.3f}")

    # the two summary views, matching the PTB-XL companion
    lowpass = probe(patch_feats(bandpass(x, 0.0, 0.5)), lab, subj,
                    np.random.default_rng(SEED))
    allband = probe(np.concatenate(feats, -1), lab, subj, np.random.default_rng(SEED))
    print(f"  low-pass only  {lowpass:.3f}\n  all bands      {allband:.3f}")

    out = dict(config=dict(fs=FS, bands=BANDS, chance=CHANCE, seed=SEED,
                           n_windows=int(len(W)), n_subj=int(len(np.unique(subj))),
                           split="subject-disjoint, 1/3 held out"),
               band_f1=res, lowpass_view_f1=lowpass, allband_view_f1=allband)
    (OUT / "fig_persistence_hapt.json").write_text(json.dumps(out, indent=1))

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ctr = [np.sqrt(max(lo, 0.02) * hi) for lo, hi in BANDS]
    ax.semilogx(ctr, list(res.values()), "o-", color=C0, lw=2, label="single band")
    ax.axhline(allband, color="k", ls="--", lw=1, label=f"all bands ({allband:.2f})")
    ax.axhline(CHANCE, color=C1, ls=":", lw=1, label=f"chance ({CHANCE:.2f})")
    ax.set_xlabel("band centre (Hz)")
    ax.set_ylabel("activity macro-F1")
    ax.set_title("HAPT: which band carries the persistent factor?", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=.3)
    fig.savefig(OUT / "fig_persistence_hapt.png", dpi=180, bbox_inches="tight")
    print("  -> runs_v2/fig_persistence_hapt.{json,png}")


if __name__ == "__main__":
    main()
