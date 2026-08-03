"""Preprocess XJTU-SY into per-bearing snapshot sequences with proxy factors.

Each 1.28s snapshot (32768 @ 25.6kHz, horizontal channel) -> one patch:
  decimated raw waveform (PATCH samples) after per-snapshot z-norm.
Ground-truth proxies per snapshot t of a bearing with life T:
  slow: life fraction t/T (RUL proxy), 3-class health stage
  fast: kurtosis, dominant-freq index of the snapshot (instantaneous signature)
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy import signal, stats

BASE = ("/mnt/workspace/data/MFM_data/awesome_industrial_dataset/"
        "XJTU-SY Bearing Datasets/XJTU-SY_Bearing_Datasets")
CONDS = ["35Hz12kN", "37.5Hz11kN", "40Hz10kN"]
PATCH = 256
OUT = "data/xjtu.npz"


def snapshot_features(x):
    kurt = stats.kurtosis(x)
    f = np.abs(np.fft.rfft(x * signal.windows.hann(len(x))))
    dom = np.argmax(f[1:]) + 1                    # skip DC
    rms = np.sqrt(np.mean(x ** 2))
    return kurt, dom / len(f), rms


def log_spectrum(x):
    """Full-band log power spectrum binned to PATCH bins (preserves highs)."""
    mag = np.abs(np.fft.rfft(x * signal.windows.hann(len(x))))
    binned = mag[1:].reshape(PATCH, -1).mean(1)     # drop DC, mean-pool bands
    return np.log1p(binned).astype(np.float32)


def load_bearing(path):
    files = sorted(glob.glob(path + "/*.parquet"),
                   key=lambda p: int(os.path.basename(p)[:-8]))
    patches, kurts, doms, rmss = [], [], [], []
    for f in files:
        x = pd.read_parquet(f)["Horizontal_vibration_signals"].to_numpy().astype(np.float32)
        k, d, r = snapshot_features(x)
        patches.append(log_spectrum(x)); kurts.append(k); doms.append(d); rmss.append(r)
    n = len(patches)
    return (np.stack(patches), np.arange(n) / (n - 1),
            np.array(kurts, np.float32), np.array(doms, np.float32), np.array(rmss, np.float32))


def main():
    data = {}
    for cond in CONDS:
        for b in sorted(os.listdir(os.path.join(BASE, cond))):
            p, life, kurt, dom, rms = load_bearing(os.path.join(BASE, cond, b))
            data[b] = dict(patch=p, life=life, kurt=kurt, dom=dom, rms=rms)
            print(f"{b:12s} n={len(p):4d} patch{p.shape} "
                  f"kurt[{kurt.min():.1f},{kurt.max():.1f}] rms_trend={np.corrcoef(life,rms)[0,1]:.2f}")
    np.savez_compressed(OUT, **{f"{b}__{k}": v for b, d in data.items() for k, v in d.items()})
    print("saved", OUT, "bearings:", len(data))


if __name__ == "__main__":
    main()
