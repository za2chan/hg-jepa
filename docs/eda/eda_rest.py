"""PTB-XL / XJTU / Synthetic EDA + per-dataset ACF (STEP 1-3). Observation
only. Saves figures to docs/eda/figs/. Run from repo root."""
import glob
import os
import sys

sys.path.insert(0, os.getcwd())          # datagen.py lives at repo root

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

PB = "data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
XBASE = ("/mnt/workspace/data/MFM_data/awesome_industrial_dataset/"
         "XJTU-SY Bearing Datasets/XJTU-SY_Bearing_Datasets")


def acf_full(x, K):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    a = np.fft.irfft(f * np.conj(f))[:K]
    return a / (a[0] + 1e-12)


def tac_cross(a, kref=1):
    an = a / (abs(a[kref]) + 1e-12)
    idx = np.flatnonzero(an[kref:] >= 1 / np.e)
    return kref + (idx.max() if len(idx) else 0)


# ---------------- PTB-XL ----------------
def ptbxl():
    import pandas as pd, ast, wfdb
    df = pd.read_csv(f"{PB}/ptbxl_database.csv")
    df["norm"] = df.scp_codes.apply(lambda s: int("NORM" in ast.literal_eval(s)))
    print("=== PTB-XL ===")
    print(f"records {len(df)} | patients {df.patient_id.nunique()} | "
          f"NORM balance {df.norm.value_counts().to_dict()}")
    print("label granularity: per-record diagnosis, CONSTANT across the 10 s; "
          "NORM = 'normal ECG' present in scp_codes; no within-record time annotation")
    fig, ax = plt.subplots(2, 2, figsize=(13, 5), sharex=True, sharey=True)
    norm_ids = df[df.norm == 1].iloc[:2]
    ab_ids = df[df.norm == 0].iloc[:2]
    for j, (_, r) in enumerate(list(norm_ids.iterrows()) + list(ab_ids.iterrows())):
        sig, meta = wfdb.rdsamp(f"{PB}/{r.filename_lr}")
        x = sig[:, 1]                                  # lead II
        a = ax[j // 2, j % 2]
        a.plot(np.arange(len(x)) / 100.0, x, lw=0.6)
        a.set_title(f"{'NORM' if r.norm else 'non-NORM'} (rec {r.ecg_id}), lead II @100Hz")
        a.set_xlabel("time (s)")
    ax[0, 0].set_ylabel("mV"); ax[1, 0].set_ylabel("mV")
    plt.tight_layout(); plt.savefig("docs/eda/figs/ptbxl_records.png", dpi=110); plt.close()
    # ACF of one record
    sig, _ = wfdb.rdsamp(f"{PB}/{df.iloc[0].filename_lr}")
    return sig[:, 1]


# ---------------- XJTU ----------------
def xjtu():
    print("\n=== XJTU-SY ===")
    b = "Bearing1_1"; cond = "35Hz12kN"
    files = sorted(glob.glob(f"{XBASE}/{cond}/{b}/*.parquet"),
                   key=lambda p: int(os.path.basename(p)[:-8]))
    n = len(files)
    print(f"{b}: {n} snapshots (1/min), each 32768 @ 25.6kHz = 1.28 s, 2 channels")
    print("label = life fraction = snapshot_index/(n-1) [normalized position in run, "
          "NOT physical RUL]; ~constant across an 80 ms window")
    import pandas as pd
    early = pd.read_parquet(files[2])["Horizontal_vibration_signals"].to_numpy()
    late = pd.read_parquet(files[-3])["Horizontal_vibration_signals"].to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for a, x, tag in [(ax[0], early, f"early (life {2/(n-1):.2f})"),
                      (ax[1], late, f"late (life {(n-3)/(n-1):.2f})")]:
        seg = x[:2048]                                 # 80 ms window
        t = np.arange(len(seg)) / 25600 * 1000
        a.plot(t, seg, lw=0.4, label="carrier")
        a.plot(t, np.abs(hilbert(seg)), lw=1.2, color="crimson", label="|Hilbert| envelope")
        a.set_title(f"{tag}, 80 ms"); a.set_xlabel("time (ms)"); a.legend(fontsize=7)
    ax[0].set_ylabel("vibration (raw)")
    plt.tight_layout(); plt.savefig("docs/eda/figs/xjtu_snapshots.png", dpi=110); plt.close()
    return early.astype(float)


# ---------------- Synthetic ----------------
def synth():
    print("\n=== Synthetic ===")
    from datagen import generate
    x, s, u, phi = generate(20000, seed=1)
    print("ground truth PER TIMESTEP: regime s(t) (Markov, 3-state), OU factor "
          "u(t), phase phi(t). Granularity = per-sample (unlike real per-window/record).")
    seg = slice(0, 3000)
    t = np.arange(seg.stop)
    fig, ax = plt.subplots(3, 1, figsize=(13, 5), sharex=True)
    ax[0].step(t, s[seg], where="post", color="navy"); ax[0].set_ylabel("regime s"); ax[0].set_yticks([0, 1, 2])
    ax[0].set_title("Synthetic: regime (slow) / OU u (fast) / observed x, aligned")
    ax[1].plot(t, u[seg], color="darkorange", lw=0.7); ax[1].set_ylabel("OU u")
    ax[2].plot(t, x[seg], color="k", lw=0.4); ax[2].set_ylabel("x"); ax[2].set_xlabel("step")
    plt.tight_layout(); plt.savefig("docs/eda/figs/synth_window.png", dpi=110); plt.close()
    return x.astype(float)


# ---------------- ACF panel (all datasets, log-lag) ----------------
def acf_panel(hapt_sig, ptb_sig, xjtu_sig, synth_sig):
    import pandas as pd
    series = {
        "HAPT |acc|": (hapt_sig, 50.0, 400),
        "PTB-XL leadII": (ptb_sig, 100.0, 400),
        "XJTU carrier": (xjtu_sig, 25600.0, 2000),
        "Synthetic x": (synth_sig, None, 400),
    }
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    for i, (name, (sig, fs, K)) in enumerate(series.items()):
        raw = acf_full(sig, K)
        en = acf_full(np.convolve(np.asarray(sig, float) ** 2, np.ones(8) / 8, "valid"), K)
        ev = acf_full(np.convolve(np.abs(hilbert(sig)), np.ones(8) / 8, "valid"), K)
        lags = np.arange(1, K)
        ax[i].semilogx(lags, np.abs(raw[1:]), lw=0.9, label="raw |ACF|")
        ax[i].semilogx(lags, np.abs(en[1:]), lw=0.9, label="energy")
        ax[i].semilogx(lags, np.abs(ev[1:]), lw=0.9, label="envelope")
        ax[i].axhline(1 / np.e, color="gray", ls=":", lw=0.8)
        traw = tac_cross(np.abs(raw), 1)
        ax[i].axvline(traw, color="C0", ls="--", lw=0.8)
        unit = f" ({traw/fs*1000:.0f} ms)" if fs else " (steps)"
        ax[i].set_title(f"{name}\nraw T_ac={traw}{unit}", fontsize=9)
        ax[i].set_xlabel("lag (samples, log)"); ax[i].legend(fontsize=6)
    ax[0].set_ylabel("|ACF|")
    plt.tight_layout(); plt.savefig("docs/eda/figs/acf_all.png", dpi=110); plt.close()
    print("\nsaved ACF panel: acf_all.png")


if __name__ == "__main__":
    p = ptbxl(); x = xjtu(); sy = synth()
    # HAPT |acc| signal for ACF
    import numpy as np
    from numpy.linalg import norm
    acc = np.loadtxt("data/hapt/RawData/acc_exp01_user01.txt").astype(np.float32)
    hmag = norm(acc, axis=1)
    acf_panel(hmag, p, x, sy)
    print("saved: ptbxl_records.png, xjtu_snapshots.png, synth_window.png")
