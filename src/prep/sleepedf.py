"""Sleep-EDF v2 prep (protocol A1/A2/A4), same contract as prep/hapt.py.

Emits W (n, L, patch*n_ax) / lab (n, L) / fast (n, L) / subj (n,), so
hglp/train_real.py consumes it unchanged: labels are 1..5 and its
`(lab >= 1) & (lab <= 6)` mask already covers that range, 0 = unscored.

Config from runs_v2/eda_sleepedf.json:
  patch 50 (0.5 s)  -> fast proxy T_ac ~3 patches, well above the 1-patch floor
                       that made HAPT's accmag unmeasurable
  L 256 (128 s)     -> spans ~2 stage dwells (median 60 s), so the slow factor
                       VARIES inside the window (HAPT's one unique role)
  tau ~6 patches    -> 5 of 8 horizons land beyond tau (HAPT: 0 of 6)

Windows are cut on a fixed stride WITHOUT looking at labels (A2); unscored
positions stay in for pretraining and are excluded only at probe time.

python3 src/prep/sleepedf.py [--max-rec N]  ->  data/sleepedf_v2.npz
"""
import argparse, pathlib, warnings
import numpy as np
from scipy import signal as sps
import mne

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "sleepedf"
OUT = ROOT / "data" / "sleepedf_v2.npz"

FS = 100.0
PATCH, L, N_AX = 50, 256, 3          # 0.5 s patches, 128 s window, 3 channels
WIN = PATCH * L
STRIDE = (L // 2) * PATCH            # 50% overlap, as in HAPT prep
CHANS = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
# W / N1 / N2 / N3+N4 / REM  -> 1..5;  everything else (incl. "?" and MOVEMENT) -> 0
STAGES = {"Sleep stage W": 1, "Sleep stage 1": 2, "Sleep stage 2": 3,
          "Sleep stage 3": 4, "Sleep stage 4": 4, "Sleep stage R": 5}
SNAME = ["unscored", "W", "N1", "N2", "N3", "REM"]
WAKE_MARGIN_S = 30 * 60              # keep 30 min of lights-on wake at each end


def band_env(x, lo, hi):
    b, a = sps.butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return np.abs(sps.hilbert(sps.filtfilt(b, a, x)))


def load_rec(psg, hyp):
    raw = mne.io.read_raw_edf(psg, preload=True, verbose=False)
    missing = [c for c in CHANS if c not in raw.ch_names]
    if missing:
        return None
    x = raw.get_data(picks=CHANS).astype(np.float32)      # (3, T) volts @100 Hz
    y = np.zeros(x.shape[1], dtype=np.int64)
    for on, du, de in zip(*[getattr(mne.read_annotations(hyp), k)
                            for k in ("onset", "duration", "description")]):
        if de in STAGES:
            a, b = int(on * FS), min(int((on + du) * FS), len(y))
            y[a:b] = STAGES[de]
    # Recordings run ~22 h with many hours of lights-on wake at both ends; keeping
    # them would make W trivially dominant and waste most windows.
    sleep = np.flatnonzero((y >= 2) & (y <= 5))
    if len(sleep) < WIN:
        return None
    m = int(WAKE_MARGIN_S * FS)
    a, b = max(0, sleep[0] - m), min(len(y), sleep[-1] + m)
    x, y = x[:, a:b], y[a:b]
    # ponytail: robust clip at 20x MAD, else a single movement artifact dominates the
    # train-split normalization in train_real. Widen if it ever clips real slow waves.
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1, keepdims=True) + 1e-12
    x = np.clip(x, med - 20 * mad, med + 20 * mad)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rec", type=int, default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    pairs = []
    for psg in sorted(RAW.glob("*-PSG.edf")):
        hyp = next(RAW.glob(psg.name[:6] + "*-Hypnogram.edf"), None)
        if hyp is not None:
            pairs.append((psg, hyp))
    pairs = pairs[: a.max_rec]
    assert pairs, f"no complete PSG/Hypnogram pairs in {RAW}"

    Ws, labs, fasts, subjs = [], [], [], []
    patch_ends = np.arange(L) * PATCH + (PATCH - 1)       # end-of-patch offsets (A4)
    for psg, hyp in pairs:
        got = load_rec(psg, hyp)
        if got is None:
            print("skip", psg.name); continue
        x, y = got
        subj = int(psg.name[3:5])                          # SC4<ss><night>
        # Fast proxy: EOG energy. Chosen by EDA -- T_ac 3 patches (measurable) and
        # only R^2=0.095 explained by the stage label (HAPT's alternatives: 0.58-0.90).
        fast = band_env(x[2], 0.5, 20.0) ** 2
        fast = np.log1p(fast / (np.median(fast) + 1e-12)).astype(np.float32)
        for s0 in range(0, x.shape[1] - WIN, STRIDE):      # no label filter (A2)
            # (T, n_ax) then reshape -> axis varies fastest, the interleaved layout
            # train_real._norm_stats assumes (cols = a + n_ax*k). Same as HAPT prep.
            Ws.append(x[:, s0:s0 + WIN].T.reshape(L, PATCH * N_AX).copy())
            ends = s0 + patch_ends
            labs.append(y[ends]); fasts.append(fast[ends]); subjs.append(subj)
        print(f"{psg.name}  subj {subj:2d}  {x.shape[1]/FS/3600:.1f} h  "
              f"windows so far {len(Ws)}")

    W = np.stack(Ws).astype(np.float32)
    lab = np.stack(labs).astype(np.int64)
    fastarr = np.stack(fasts).astype(np.float32)
    subj = np.array(subjs, np.int64)
    np.savez(a.out, W=W, lab=lab, fast=fastarr, subj=subj)

    labeled = (lab >= 1) & (lab <= 6)
    share = {SNAME[k]: round(float((lab == k).mean()), 3) for k in range(6)}
    print(f"\nwindows {W.shape} | recordings {len(pairs)} | subjects {len(np.unique(subj))}")
    print(f"labeled-position frac {labeled.mean():.3f} | "
          f"windows spanning a stage transition {(np.array([len(np.unique(r[r > 0])) > 1 for r in lab])).mean():.3f}")
    print("per-position stage share", share)
    print("saved", a.out, f"({pathlib.Path(a.out).stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
