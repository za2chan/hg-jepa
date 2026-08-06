"""Sleep-EDF EDA — can it replace HAPT?

Checks are exactly the axes HAPT failed or uniquely passed:
  A  slow factor varies WITHIN the window   (HAPT's only unique contribution)
  B  a fast proxy exists with T_ac above patch resolution AND uncontaminated
     by the slow label                      (HAPT's fatal failure: T_ac=1 patch)
  C  no coherent-periodic component dominating the window  (screen Check 1)
  D  timescale gap: T_ac(fast) << dwell(slow)
Run at SAMPLE level, never on patch means (CLAUDE.md 5: patch means low-pass away
short-period confounds).

python3 src/prep/eda_sleepedf.py  ->  runs_v2/eda_sleepedf.json + .png
"""
import json, pathlib, warnings
import numpy as np
from scipy import signal as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA, OUT = ROOT / "data" / "sleepedf", ROOT / "runs_v2"

FS = 100.0            # Fpz-Cz / Pz-Oz / EOG sampling rate
EPOCH = 30.0          # hypnogram label resolution (s)
PATCH = 100           # 1 s patches -> L=256 window is 4.3 min
STAGES = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
          "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4}
SNAME = ["W", "N1", "N2", "N3", "REM"]
PATCH_SWEEP = [4, 10, 25, 50, 100, 200]   # samples per patch (0.04 s .. 2 s)


def load_rec(psg, hyp):
    raw = mne.io.read_raw_edf(psg, preload=True, verbose=False)
    ann = mne.read_annotations(hyp)
    ch = {c: raw.ch_names.index(c) for c in raw.ch_names}
    pick = [c for c in raw.ch_names if "Fpz" in c or "Pz-Oz" in c or "EOG" in c]
    x = raw.get_data(picks=pick)                       # (C, T) at 100 Hz
    # per-sample stage label from the annotation intervals
    y = np.full(x.shape[1], -1, dtype=np.int8)
    for on, du, de in zip(ann.onset, ann.duration, ann.description):
        if de not in STAGES:
            continue
        a, b = int(on * FS), int((on + du) * FS)
        y[a:min(b, len(y))] = STAGES[de]
    # drop the long W padding at both ends (recordings include hours of lights-on wake)
    scored = np.flatnonzero(y >= 0)
    if len(scored) == 0:
        return None
    sl = slice(scored[0], scored[-1] + 1)
    x, y = x[:, sl], y[sl]
    nz = np.flatnonzero(y != 0)
    if len(nz):                                        # keep 30 min of wake margin
        m = int(30 * 60 * FS)
        a, b = max(0, nz[0] - m), min(len(y), nz[-1] + m)
        x, y = x[:, a:b], y[a:b]
    return x, y, pick


def acf_time(v, max_lag, thresh=1 / np.e):
    """First lag where the autocorrelation drops below 1/e, in samples."""
    v = np.asarray(v, float)
    v = v - v.mean()
    if v.std() < 1e-12:
        return np.nan, np.zeros(max_lag)
    n = len(v)
    f = np.fft.rfft(v, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:max_lag].real
    ac /= ac[0]
    below = np.flatnonzero(ac < thresh)
    return (float(below[0]) if len(below) else float(max_lag)), ac


def band_env(x, lo, hi):
    b, a = sps.butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return np.abs(sps.hilbert(sps.filtfilt(b, a, x)))


def r2_from_labels(proxy, y):
    """How much of the fast proxy is explained by the slow label alone (one-way ANOVA R²).
    HAPT's alternative proxies scored 0.58-0.90 here and were unusable."""
    m = y >= 0
    proxy, y = proxy[m], y[m]
    gm = proxy.mean()
    ss_t = ((proxy - gm) ** 2).sum()
    ss_b = sum(((proxy[y == k].mean() - gm) ** 2) * (y == k).sum()
               for k in np.unique(y) if (y == k).sum() > 1)
    return float(ss_b / ss_t) if ss_t > 0 else np.nan


def main():
    recs = sorted(DATA.glob("*-PSG.edf"))
    assert recs, f"no PSG files in {DATA}"
    res = {"config": dict(fs=FS, patch=PATCH, epoch=EPOCH, n_rec=len(recs)),
           "recordings": [], "dwell_s": [], "acf": {}, "proxy": {}}
    dwell_all, trans_all, stage_hist = [], {}, np.zeros(5)
    Ls = [64, 128, 256, 512]        # candidate window lengths, in PATCHES
    for L in Ls:
        trans_all[L] = []
    acc_acf, proxies, patch_tac = {}, {}, {}

    for p in recs:
        h = next(DATA.glob(p.name[:6] + "*-Hypnogram.edf"), None)
        if h is None:
            continue
        got = load_rec(p, h)
        if got is None:
            continue
        x, y, pick = got
        eeg = x[0]                                   # Fpz-Cz
        n = len(y)

        # ---- A. slow-factor dwell + within-window transition fraction ----
        chg = np.flatnonzero(np.diff(y) != 0) + 1
        segs = np.diff(np.concatenate([[0], chg, [n]])) / FS
        segs = segs[segs > 0]
        dwell_all += list(segs)
        for L in Ls:
            W = L * PATCH
            starts = np.arange(0, n - W, W // 2)
            frac = np.mean([len(np.unique(y[s:s + W])) > 1 for s in starts]) if len(starts) else np.nan
            trans_all[L].append(float(frac))
        for k in range(5):
            stage_hist[k] += (y == k).sum()

        # ---- B/C/D. sample-level ACF of raw + candidate fast proxies ----
        seg = eeg[: int(600 * FS)]                   # 10 min for the ACF probe
        ml = int(60 * FS)
        for nm, v in [("raw EEG", seg),
                      ("EEG energy", seg ** 2),
                      ("beta env (16-30Hz)", band_env(seg, 16, 30)),
                      ("alpha env (8-12Hz)", band_env(seg, 8, 12)),
                      ("delta env (0.5-4Hz)", band_env(seg, 0.5, 4)),
                      ("EOG energy", x[-1][: int(600 * FS)] ** 2)]:
            t, ac = acf_time(v, ml)
            acc_acf.setdefault(nm, []).append(t)
            if nm not in res["acf"]:
                res["acf"][nm] = {"acf_curve": ac[::10].tolist()}

        # ---- B'. T_ac of the PATCH-LEVEL proxy series, per candidate patch size.
        # This is the number comparable to HAPT's "accmag T_ac = 1 patch": the model
        # never sees samples, it sees patch features. Too small -> unmeasurable fast
        # factor (HAPT's failure); too large -> window can't span a stage.
        for w in PATCH_SWEEP:
            npw = len(eeg) // w
            for nm, v in [("EEG energy", eeg ** 2),
                          ("alpha env (8-12Hz)", band_env(eeg, 8, 12)),
                          ("EOG energy", x[-1] ** 2)]:
                vp = v[: npw * w].reshape(npw, w).mean(1)
                t, _ = acf_time(vp, min(2000, npw // 4))
                patch_tac.setdefault((nm, w), []).append(t)

        # ---- B. contamination of each proxy by the slow label ----
        # proxy sampled once per patch so it matches what the model would see
        npatch = n // PATCH
        yp = y[: npatch * PATCH].reshape(npatch, PATCH)
        yp = np.array([np.bincount(r[r >= 0], minlength=5).argmax() if (r >= 0).any() else -1
                       for r in yp])
        for nm, v in [("EEG energy", eeg ** 2),
                      ("beta env (16-30Hz)", band_env(eeg, 16, 30)),
                      ("alpha env (8-12Hz)", band_env(eeg, 8, 12)),
                      ("EOG energy", x[-1] ** 2)]:
            vp = v[: npatch * PATCH].reshape(npatch, PATCH).mean(1)
            proxies.setdefault(nm, []).append(r2_from_labels(vp, yp))

        res["recordings"].append(dict(file=p.name, hours=round(n / FS / 3600, 2),
                                      channels=pick,
                                      stages=[int((y == k).sum()) for k in range(5)]))

    d = np.array(dwell_all)
    res["dwell_s"] = dict(median=float(np.median(d)), mean=float(d.mean()),
                          p25=float(np.percentile(d, 25)), p75=float(np.percentile(d, 75)),
                          n_segments=len(d))
    res["transition_fraction"] = {str(L): float(np.mean(trans_all[L])) for L in Ls}
    res["stage_share"] = {SNAME[k]: float(stage_hist[k] / stage_hist.sum()) for k in range(5)}
    res["acf_tac_samples"] = {k: float(np.mean(v)) for k, v in acc_acf.items()}
    res["acf_tac_patches"] = {k: float(np.mean(v)) / PATCH for k, v in acc_acf.items()}
    res["proxy_label_contamination_r2"] = {k: float(np.nanmean(v)) for k, v in proxies.items()}
    # patch-level T_ac, and the window length (in patches) needed to span one dwell
    med_dwell = float(np.median(d))
    res["patch_level"] = {}
    for (nm, w), v in patch_tac.items():
        tac = float(np.mean(v))
        res["patch_level"].setdefault(f"{w}", {})[nm] = dict(
            tac_patches=tac, patch_s=w / FS,
            L_for_1_dwell=int(np.ceil(med_dwell * FS / w)))
    OUT.mkdir(exist_ok=True)
    (OUT / "eda_sleepedf.json").write_text(json.dumps(res, indent=1))

    # ---------------- figure ----------------
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    ax[0].hist(np.clip(d, 0, 900), bins=60, color="#1a5fb4")
    ax[0].axvline(np.median(d), color="#c01c28", ls="--",
                  label=f"median {np.median(d):.0f}s")
    ax[0].axvline(17.2, color="#e8a33d", ls=":", label="HAPT median 17.2s")
    ax[0].set_xlabel("stage dwell (s)"); ax[0].set_ylabel("segments"); ax[0].legend(fontsize=8)
    ax[0].set_title("A. slow factor dwell", fontsize=10)

    for nm, tl in res["acf"].items():
        ax[1].plot(np.arange(len(tl["acf_curve"])) * 10 / FS, tl["acf_curve"], lw=1.5, label=nm)
    ax[1].axhline(1 / np.e, color="#888", ls="--", lw=1)
    ax[1].set_xlim(0, 20); ax[1].set_xlabel("lag (s)"); ax[1].set_ylabel("autocorr")
    ax[1].legend(fontsize=7); ax[1].set_title("C/D. sample-level ACF", fontsize=10)

    for nm in ["EEG energy", "alpha env (8-12Hz)", "EOG energy"]:
        ws = sorted(int(k) for k in res["patch_level"])
        ax[2].plot([w / FS for w in ws],
                   [res["patch_level"][str(w)][nm]["tac_patches"] for w in ws],
                   "o-", label=nm)
    ax[2].axhline(1.0, color="#c01c28", ls="--", lw=1.2, label="1 patch = unmeasurable (HAPT)")
    ax[2].axhline(4.0, color="#3a9e6e", ls=":", lw=1.2, label="4 patches = workable")
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel("patch size (s)"); ax[2].set_ylabel("fast $T_{ac}$ (patches)")
    ax[2].legend(fontsize=6.5); ax[2].set_title("B. patch size sets measurability", fontsize=10)

    ks = list(res["proxy_label_contamination_r2"])
    vs = [res["proxy_label_contamination_r2"][k] for k in ks]
    ax[3].barh(range(len(ks)), vs, color=["#3a9e6e" if v < .3 else "#c01c28" for v in vs])
    ax[3].axvline(.30, color="#888", ls="--", label="usable if < 0.3")
    ax[3].set_yticks(range(len(ks))); ax[3].set_yticklabels(ks, fontsize=8)
    ax[3].set_xlabel("$R^2$ explained by stage label"); ax[3].legend(fontsize=7)
    ax[3].set_title("B. proxy contamination", fontsize=10)
    fig.suptitle("Sleep-EDF EDA — can it replace HAPT?", fontsize=12, y=1.03)
    fig.tight_layout(); fig.savefig(OUT / "eda_sleepedf.png", dpi=180, bbox_inches="tight")

    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("recordings", "acf")}, indent=1))


if __name__ == "__main__":
    main()
