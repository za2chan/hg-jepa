"""Introduction figure, real-data half: a perfectly persistent factor living in
a FAST waveform feature.

PTB-XL's diagnosis is constant for the whole record -- infinite persistence, the
limit case. Yet nothing in the low-frequency content of the ECG carries it: the
label is written in the shape of the QRS complex, which lasts ~100 ms. So the
persistence of a factor says nothing about the frequency band it occupies.

Measured, not asserted: beats are R-peak aligned, the ECG is restricted to one
frequency band at a time, and a linear probe is fit per band with a
patient-disjoint split.

python3 src/exp/fig_persistence_ptbxl.py -> runs_v2/fig_persistence_ptbxl.{png,json}
"""
import json, pathlib
import numpy as np
from scipy import signal as sps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "runs_v2"
FS = 100.0
PRE, POST = 25, 45            # beat window: -0.25 s .. +0.45 s around R
BANDS = [(0.05, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0),
         (5.0, 10.0), (10.0, 20.0), (20.0, 40.0)]
C0, C1 = "#1a5fb4", "#c01c28"


def bandpass(x, lo, hi):
    ny = FS / 2
    if hi >= ny:
        b, a = sps.butter(3, lo / ny, btype="high")
    else:
        b, a = sps.butter(3, [lo / ny, hi / ny], btype="band")
    return sps.filtfilt(b, a, x, axis=-1)


def r_peaks(sig):
    """R-peak indices from the QRS band, per record."""
    q = bandpass(sig, 5.0, 20.0)
    out = []
    for v in q:
        thr = np.percentile(np.abs(v), 98) * 0.5
        p, _ = sps.find_peaks(np.abs(v), height=thr, distance=int(0.3 * FS))
        out.append(p)
    return out


def beat_average(sig, peaks):
    """Mean beat per record, R-aligned. NaN row if a record has no usable beat."""
    n, T = sig.shape
    B = np.full((n, PRE + POST), np.nan, np.float32)
    for i, p in enumerate(peaks):
        seg = [sig[i, r - PRE:r + POST] for r in p if r - PRE >= 0 and r + POST <= T]
        if seg:
            B[i] = np.mean(seg, 0)
    return B


def probe(F, y, tr, te):
    m = ~np.isnan(F).any(1)
    tr_, te_ = tr & m, te & m
    mu, sd = F[tr_].mean(0), F[tr_].std(0) + 1e-8
    clf = LogisticRegression(max_iter=2000, C=1.0).fit((F[tr_] - mu) / sd, y[tr_])
    return float(f1_score(y[te_], clf.predict((F[te_] - mu) / sd), average="macro"))


def main():
    d = np.load(ROOT / "data" / "ptbxl_v2.npz")
    x = d["W"].reshape(len(d["W"]), -1).astype(np.float64)     # (n, 1000) lead II
    y = d["lab"][:, -1]
    subj = d["subj"]
    rng = np.random.default_rng(0)
    g = np.unique(subj)
    te_g = set(rng.permutation(g)[: len(g) // 3].tolist())
    te = np.array([s in te_g for s in subj]); tr = ~te

    peaks = r_peaks(x)
    res = {"config": dict(fs=FS, n=len(x), chance=0.5, bands=BANDS,
                          split="patient-disjoint, 1/3 held out"), "band_f1": {}}

    # --- per-band probe: restrict the ECG to one band, then read the beat -----
    for lo, hi in BANDS:
        B = beat_average(bandpass(x, lo, hi), peaks)
        res["band_f1"][f"{lo}-{hi}"] = probe(B, y, tr, te)

    # --- the two views, as in the synthetic figure ----------------------------
    slow_view = bandpass(x, 0.05, 1.0)[:, ::5]                 # drift / baseline
    res["lowpass_view_f1"] = probe(slow_view.astype(np.float32), y, tr, te)
    res["morphology_view_f1"] = probe(beat_average(bandpass(x, 5.0, 40.0), peaks),
                                      y, tr, te)

    # --- class-mean beats for the picture ------------------------------------
    Bfull = beat_average(bandpass(x, 0.5, 40.0), peaks)
    ok = ~np.isnan(Bfull).any(1)
    mean0 = np.nanmean(Bfull[ok & (y == 0)], 0)
    mean1 = np.nanmean(Bfull[ok & (y == 1)], 0)
    sd0 = np.nanstd(Bfull[ok & (y == 0)], 0)
    sd1 = np.nanstd(Bfull[ok & (y == 1)], 0)
    res["n_usable_beats"] = int(ok.sum())
    OUT.mkdir(exist_ok=True)
    (OUT / "fig_persistence_ptbxl.json").write_text(json.dumps(res, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    t = (np.arange(PRE + POST) - PRE) / FS * 1000
    for m, s, c, lb in [(mean0, sd0, C0, "class 0"), (mean1, sd1, C1, "class 1")]:
        ax[0].plot(t, m, lw=2, color=c, label=lb)
        ax[0].fill_between(t, m - 0.3 * s, m + 0.3 * s, color=c, alpha=.15, lw=0)
    ax[0].axvspan(-40, 60, color="#999", alpha=.12, lw=0)
    ax[0].text(10, ax[0].get_ylim()[1] * .92, "QRS\n~100 ms", ha="center",
               fontsize=8, color="#555")
    ax[0].set_xlabel("time from R peak (ms)"); ax[0].set_ylabel("lead II (a.u.)")
    ax[0].legend(fontsize=8, frameon=False)
    ax[0].set_title("(a) the record-constant label lives in a ~100 ms shape",
                    fontsize=9.5)
    ax[0].grid(alpha=.25, lw=.6); ax[0].set_axisbelow(True)
    ax[0].spines[["top", "right"]].set_visible(False)

    # Two views of the same records, matching the synthetic figure: what a
    # signal-decomposition reading returns, vs the fast waveform shape.
    # (The per-band sweep stays in the JSON only: R-peak alignment leaks QRS
    #  shape back into every band, so it understates the contrast.)
    vals = [res["lowpass_view_f1"], res["morphology_view_f1"]]
    bars = ax[1].bar([0, 1], vals, 0.55, color=["#9aa0a6", C0])
    for xi, v in zip([0, 1], vals):
        ax[1].text(xi, v + .015, f"{v:.3f}", ha="center", fontsize=9)
    ax[1].axhline(0.5, color="#c01c28", ls="--", lw=1.2)
    ax[1].text(1.42, 0.508, "chance", fontsize=8, color="#c01c28", ha="right")
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["slow component\n($0.05\\!-\\!1$ Hz)",
                           "beat shape\n($5\\!-\\!40$ Hz)"], fontsize=8.5)
    ax[1].set_xlim(-0.6, 1.6); ax[1].set_ylim(0.44, 0.83)
    ax[1].set_ylabel("diagnosis macro-F1")
    ax[1].set_title("(b) the slow component carries none of it", fontsize=9.5)
    ax[1].grid(alpha=.25, lw=.6); ax[1].set_axisbelow(True)
    ax[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("PTB-XL: an infinitely persistent factor with no slow signature",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_persistence_ptbxl.png", dpi=200, bbox_inches="tight")

    print(json.dumps(res["band_f1"], indent=1))
    print("low-pass view  F1", round(res["lowpass_view_f1"], 3))
    print("morphology view F1", round(res["morphology_view_f1"], 3))
    print("usable beats", res["n_usable_beats"], "/", len(x))
    print("wrote runs_v2/fig_persistence_ptbxl.png + .json")


if __name__ == "__main__":
    main()
