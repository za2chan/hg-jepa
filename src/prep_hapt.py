"""HAPT v2 prep (protocol A1/A2/A4). L=256 windows (20.5 s), cut WITHOUT
looking at labels (VOID/transition kept for pretraining), per-PATCH labels
stored so probing scores only labeled positions. 3 accelerometer axes,
channel-mixing input. Raw windows saved; A3 global normalization is applied
at train time on the train-subject split. No window crosses a recording.
"""
import glob

import numpy as np

RAW = "data/hapt/RawData"
PATCH, L, N_AX = 4, 256, 3
WIN = PATCH * L                      # 1024 samples = 20.48 s @50Hz
STRIDE = (L // 2) * PATCH            # 512 samples, 50% overlap
OUT = "data/hapt_v2.npz"


def per_sample_labels(exp, n, labels):
    per = np.zeros(n, dtype=np.int64)
    for _, _, a, s, e in labels[labels[:, 0] == exp]:
        per[s:e + 1] = a
    return per


def main():
    labels = np.loadtxt(f"{RAW}/labels.txt", dtype=int)
    W, lab, fast, subj = [], [], [], []
    patch_ends = np.arange(L) * PATCH + (PATCH - 1)      # end-of-patch offsets
    for accf in sorted(glob.glob(f"{RAW}/acc_exp*.txt")):
        exp = int(accf.split("exp")[1][:2]); user = int(accf.split("user")[1][:2])
        acc = np.loadtxt(accf).astype(np.float32)         # (T, 3), g units
        per = per_sample_labels(exp, len(acc), labels)
        mag = np.linalg.norm(acc, axis=1)
        for s0 in range(0, len(acc) - WIN, STRIDE):       # NO label filter (A2)
            w = acc[s0:s0 + WIN].reshape(L, PATCH * N_AX)  # (L, 12) channel-mixed
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
