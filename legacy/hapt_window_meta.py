"""Reconstruct per-window metadata for data/hapt.npz, in the SAME order the
prep loop appended windows: whether the window span carries a single activity
label (clean) or straddles a transition/void (contaminated), and the fraction
of the span that is the end-point label. Replicates hapt_prep.main() exactly.
Writes data/hapt_meta.npz (spans_transition, end_frac, subj, act).
"""
import glob

import numpy as np

RAW = "data/hapt/RawData"
PATCH, L = 4, 128
WIN = PATCH * L                 # 512 samples
STRIDE = 64


def main():
    labels = np.loadtxt(f"{RAW}/labels.txt", dtype=int)     # exp,user,act,start,end
    spans, end_frac, subj, act = [], [], [], []
    for accf in sorted(glob.glob(f"{RAW}/acc_exp*.txt")):
        exp = int(accf.split("exp")[1][:2])
        user = int(accf.split("user")[1][:2])
        n = sum(1 for _ in open(accf))
        per = np.zeros(n, dtype=int)
        for _, _, a, s, e in labels[labels[:, 0] == exp]:
            per[s:e + 1] = a
        for s in range(0, n - WIN, STRIDE * PATCH):
            end = s + WIN - 1
            a = per[end]
            if not 1 <= a <= 6:                             # prep's keep filter
                continue
            span = per[s:s + WIN]
            # contaminated = any sample in span differs from the end-point label
            # (covers both a different activity AND labeled-void 0 samples)
            contaminated = bool((span != a).any())
            spans.append(contaminated)
            end_frac.append(float((span == a).mean()))      # fraction matching label
            subj.append(user); act.append(a - 1)
    np.savez("data/hapt_meta.npz", spans_transition=np.array(spans),
             end_frac=np.array(end_frac, np.float32),
             subj=np.array(subj), act=np.array(act))
    print(f"windows {len(spans)} | contaminated {np.mean(spans):.3f} | "
          f"median end-label coverage {np.median(end_frac):.3f}")


if __name__ == "__main__":
    main()
