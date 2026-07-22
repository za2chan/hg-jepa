"""HAPT continuous raw acc -> windows with two genuine timescales:
  fast = gait/motion oscillation (T_ac ~5-10 samples at 50Hz)
  slow = activity class (persists ~12s / ~600 samples per segment)
Slow proxy = activity label at window end (6 locomotion/static classes).
Fast proxy = instantaneous acc magnitude at window end.
"""
import glob
import os

import numpy as np

RAW = "data/hapt/RawData"
PATCH, L = 4, 128            # 4 samples/patch, 128 patches = 512 samples = 10.2s @50Hz
WIN = PATCH * L
STRIDE = 64                 # patches -> 256 samples between windows
OUT = "data/hapt.npz"


def main():
    labels = np.loadtxt(f"{RAW}/labels.txt", dtype=int)   # exp,user,act,start,end
    W, act, accmag = [], [], []
    for accf in sorted(glob.glob(f"{RAW}/acc_exp*.txt")):
        exp = int(accf.split("exp")[1][:2])
        acc = np.loadtxt(accf).astype(np.float32)          # (T,3), gravity units
        per = np.zeros(len(acc), dtype=int)                # per-sample activity
        for _, _, a, s, e in labels[labels[:, 0] == exp]:
            per[s:e + 1] = a
        mag = np.linalg.norm(acc, axis=1)
        for s in range(0, len(acc) - WIN, STRIDE * PATCH):
            end = s + WIN - 1
            a = per[end]
            if not 1 <= a <= 6:                            # keep 6 basic activities
                continue
            w = acc[s:s + WIN]
            w = (w - w.mean(0)) / (w.std(0) + 1e-6)
            W.append(w.reshape(L, PATCH * 3)); act.append(a - 1); accmag.append(mag[end])
    W = np.stack(W); act = np.array(act); accmag = np.array(accmag, np.float32)
    np.savez_compressed(OUT, W=W, act=act, accmag=accmag)
    print("windows", W.shape, "classes", np.bincount(act), "-> saved", OUT)


if __name__ == "__main__":
    main()
