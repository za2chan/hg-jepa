"""HAPT prep with the gyroscope added: 6 axes instead of 3.

Why. `hapt.py` loads only `acc_exp*.txt`, so the encoder never sees the
gyroscope, even though `RawData/` ships one gyro file per accelerometer file.
Static postures (sitting / standing / lying) separate on the gravity direction,
which total acceleration carries, so they are fine. Walking, walking upstairs
and walking downstairs are not: all three produce periodic acceleration of
similar amplitude, and what distinguishes them is largely trunk angular
velocity. That lives in the gyroscope. Any failure to tell the three walking
classes apart therefore has a candidate explanation that is nothing to do with
the gate, and this file exists to test it.

Everything else is held fixed so the comparison isolates the channel set:
same window length, same stride, same labelling, and the same transient proxy
(accelerometer magnitude) as `hapt.py`. Axis order per patch is
[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z] repeated per sample, which is the
interleaving `train_real._norm_stats` expects (cols = a + n_ax*k).

Writes a NEW file; `data/hapt_v2.npz` and every number computed from it are
untouched.

python3 src/prep/hapt_gyro.py -> data/hapt_v2_gyro.npz   (train with n_ax=6)
"""
import glob

import numpy as np

RAW = "data/hapt/RawData"
PATCH, L, N_AX = 4, 256, 6
WIN = PATCH * L                      # 1024 samples = 20.48 s @50Hz
STRIDE = (L // 2) * PATCH            # 512 samples, 50% overlap
OUT = "data/hapt_v2_gyro.npz"


def per_sample_labels(exp, n, labels):
    per = np.zeros(n, dtype=np.int64)
    for _, _, a, s, e in labels[labels[:, 0] == exp]:
        per[s:e + 1] = a
    return per


def main():
    labels = np.loadtxt(f"{RAW}/labels.txt", dtype=int)
    W, lab, fast, subj = [], [], [], []
    patch_ends = np.arange(L) * PATCH + (PATCH - 1)
    for accf in sorted(glob.glob(f"{RAW}/acc_exp*.txt")):
        gyrof = accf.replace("acc_exp", "gyro_exp")
        exp = int(accf.split("exp")[1][:2]); user = int(accf.split("user")[1][:2])
        acc = np.loadtxt(accf).astype(np.float32)         # (T, 3), g
        gyr = np.loadtxt(gyrof).astype(np.float32)        # (T, 3), rad/s
        n = min(len(acc), len(gyr))                       # the pair is same-length
        acc, gyr = acc[:n], gyr[:n]
        six = np.concatenate([acc, gyr], axis=1)          # (T, 6)
        per = per_sample_labels(exp, n, labels)
        mag = np.linalg.norm(acc, axis=1)                 # proxy unchanged: acc only
        for s0 in range(0, n - WIN, STRIDE):
            w = six[s0:s0 + WIN].reshape(L, PATCH * N_AX)  # (L, 24), channel-mixed
            ends = s0 + patch_ends
            W.append(w); lab.append(per[ends]); fast.append(mag[ends]); subj.append(user)
    W = np.stack(W).astype(np.float32)
    lab = np.stack(lab).astype(np.int64)
    fast = np.stack(fast).astype(np.float32)
    subj = np.array(subj, np.int64)
    np.savez_compressed(OUT, W=W, lab=lab, fast=fast, subj=subj)
    labeled = (lab >= 1) & (lab <= 6)
    print(f"windows {W.shape} | subjects {len(np.unique(subj))} | "
          f"labeled-position frac {labeled.mean():.3f} | "
          f"windows spanning a transition/void {(labeled.sum(1) < L).mean():.3f}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
