"""PTB-XL ECG -> windows with two timescales:
  fast = within-beat waveform (heart rate ~1 Hz, T_ac ~ few patches)
  slow = diagnosis (NORM vs abnormal), constant across the 10 s record.
Slow proxy = NORM label; fast proxy = instantaneous ECG value at window end.
Lead II, 100 Hz, 10 s = 1000 samples.
"""
import ast

import numpy as np
import pandas as pd
import wfdb

B = "data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
PATCH, L = 10, 100           # 10 samples/patch, 100 patches = 1000 samples = 10 s
N_PER_CLASS = 2500
LEAD = 1                     # lead II
OUT = "data/ptbxl.npz"


def main():
    df = pd.read_csv(f"{B}/ptbxl_database.csv")
    df["norm"] = df.scp_codes.apply(lambda s: int("NORM" in ast.literal_eval(s)))
    rng = np.random.default_rng(0)
    sel = pd.concat([df[df.norm == c].sample(min(N_PER_CLASS, (df.norm == c).sum()),
                                              random_state=0) for c in (0, 1)])
    W, norm, ecgend = [], [], []
    for _, r in sel.iterrows():
        sig, _ = wfdb.rdsamp(f"{B}/{r.filename_lr}")     # (1000, 12)
        x = sig[:, LEAD].astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-6)
        if len(x) < L * PATCH:
            continue
        x = x[:L * PATCH]
        W.append(x.reshape(L, PATCH)); norm.append(int(r.norm)); ecgend.append(x[-1])
    W = np.stack(W); norm = np.array(norm); ecgend = np.array(ecgend, np.float32)
    np.savez_compressed(OUT, W=W, norm=norm, ecgend=ecgend)
    print("windows", W.shape, "norm balance", np.bincount(norm), "-> saved", OUT)


if __name__ == "__main__":
    main()
