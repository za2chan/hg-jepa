"""Raw within-snapshot windows for the amplitude-modulation framing.

Each 1.28s snapshot (32768 @ 25.6kHz) contains two genuine timescales:
  fast  = kHz vibration carrier (oscillates every few samples)
  slow  = its amplitude ENVELOPE (modulated at fault frequencies ~100-200Hz);
          this envelope is where the bearing fault signature lives.
We cut raw windows and store per-patch ground-truth proxies via the Hilbert
transform: envelope |analytic| (slow) and the raw carrier value (fast).
Degradation (life fraction) labels each window at the snapshot level.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.signal import hilbert

BASE = ("/mnt/workspace/data/MFM_data/awesome_industrial_dataset/"
        "XJTU-SY Bearing Datasets/XJTU-SY_Bearing_Datasets")
CONDS = ["35Hz12kN", "37.5Hz11kN", "40Hz10kN"]
PATCH = 4
L = 512
WIN = PATCH * L                 # 2048 samples = 80 ms
WINS_PER_SNAP = 6
SNAPS_PER_BEARING = 16
OUT = "data/xjtu_raw.npz"


def proxies(w):
    """Per-patch envelope (slow) and carrier (fast) from analytic signal."""
    a = hilbert(w)
    env = np.abs(a).reshape(L, PATCH).mean(1)          # slow: AM envelope
    car = w.reshape(L, PATCH).mean(1)                   # fast: carrier value
    return env.astype(np.float32), car.astype(np.float32)


def load_bearing(path):
    files = sorted(glob.glob(path + "/*.parquet"),
                   key=lambda p: int(os.path.basename(p)[:-8]))
    n = len(files)
    idx = np.linspace(0, n - 1, SNAPS_PER_BEARING).round().astype(int)
    W, ENV, CAR, LIFE = [], [], [], []
    rng = np.random.default_rng(0)
    for i in idx:
        x = pd.read_parquet(files[i])["Horizontal_vibration_signals"].to_numpy().astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-6)
        for s in rng.integers(0, len(x) - WIN, WINS_PER_SNAP):
            w = x[s:s + WIN]
            env, car = proxies(w)
            W.append(w.reshape(L, PATCH)); ENV.append(env); CAR.append(car); LIFE.append(i / (n - 1))
    return np.stack(W), np.stack(ENV), np.stack(CAR), np.array(LIFE, np.float32)


def main():
    store = {}
    for cond in CONDS:
        for b in sorted(os.listdir(os.path.join(BASE, cond))):
            W, ENV, CAR, LIFE = load_bearing(os.path.join(BASE, cond, b))
            store[f"{b}__W"] = W; store[f"{b}__env"] = ENV
            store[f"{b}__car"] = CAR; store[f"{b}__life"] = LIFE
            print(f"{b:12s} windows {W.shape} env_amp[{ENV.mean():.2f}] "
                  f"life-env corr {np.corrcoef(LIFE, ENV.mean(1))[0,1]:.2f}")
    np.savez_compressed(OUT, **store)
    print("saved", OUT)


if __name__ == "__main__":
    main()
