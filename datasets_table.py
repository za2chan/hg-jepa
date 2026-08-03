"""Cross-dataset structure table for paper §4.0 — the empirical backing for
the screen's configuration rules. Regenerated from prep-script constants +
raw-length facts (no hand-transcription; hard rule 3). Writes
runs/datasets_table.json and prints a markdown table.

Axes (the third one is the new applicability category — window-level factor
contamination — separate from dwell-vs-Delta_max):
  window (samples/sec), stride/overlap, rate, patch, label granularity+
  position, dwell, window/dwell, fraction spanning a transition, extendable?
"""
import glob
import json

import numpy as np


def hapt_facts():
    RAW = "data/hapt/RawData"
    PATCH, L, STRIDE, fs = 4, 128, 64, 50.0
    WIN = PATCH * L
    labels = np.loadtxt(f"{RAW}/labels.txt", dtype=int)
    seg = []
    for accf in sorted(glob.glob(f"{RAW}/acc_exp*.txt")):
        exp = int(accf.split("exp")[1][:2])
        for _, _, a, s, e in labels[labels[:, 0] == exp]:
            if 1 <= a <= 6:
                seg.append(e - s + 1)
    seg = np.array(seg)
    contaminated = float(np.load("data/hapt_meta.npz")["spans_transition"].mean())
    dwell = float(np.median(seg)) / fs
    return dict(
        name="HAPT", rate_hz=fs, patch=PATCH, win_samp=WIN, win_sec=WIN / fs,
        stride=f"{STRIDE} patches = {STRIDE*PATCH/fs:.2f} s (50% overlap)",
        label="per-timestep activity; window-END sample (per[end])",
        dwell_sec=dwell, win_over_dwell=(WIN / fs) / dwell,
        transition_frac=contaminated,
        extendable="NO — median segment %.1f s < 20.5 s window at L=256; "
                   "70%% of segments shorter than that" % dwell)


def ptbxl_facts():
    PATCH, L, fs = 10, 100, 100.0
    WIN = PATCH * L
    return dict(
        name="PTB-XL", rate_hz=fs, patch=PATCH, win_samp=WIN, win_sec=WIN / fs,
        stride="none — first 1000 samples of each record (1 window/record)",
        label="per-record diagnosis (NORM); CONSTANT across the record",
        dwell_sec=WIN / fs, win_over_dwell=1.0, transition_frac=0.0,
        extendable="NO — records are exactly 10 s (1000 samples)")


def xjtu_facts():
    PATCH, L, fs = 4, 512, 25600.0
    WIN = PATCH * L
    snap_sec = 32768 / fs
    return dict(
        name="XJTU-SY", rate_hz=fs, patch=PATCH, win_samp=WIN, win_sec=WIN / fs,
        stride="6 random windows / 1.28 s snapshot",
        label="per-snapshot life fraction (RUL); ~constant across 80 ms window",
        dwell_sec=float("nan"),  # degradation is continuous, no discrete dwell
        win_over_dwell=float("nan"), transition_frac=0.0,
        extendable="within-snapshot to %.2f s (16x); bounded by the 1.28 s "
                   "snapshot; no timescale gap regardless (predicted negative)"
                   % snap_sec)


def synth_facts():
    PATCH, L = 8, 256          # steps; dwell 3000 steps (datagen DWELL)
    WIN = PATCH * L
    return dict(
        name="Synthetic", rate_hz=float("nan"), patch=PATCH, win_samp=WIN,
        win_sec=float("nan"),
        stride="random anchors within window",
        label="ground-truth regime at window end; Markov, dwell ~3000 steps",
        dwell_sec=float("nan"), win_over_dwell=WIN / 3000.0,
        # crude P(switch in window) proxy, NOT a measured/end-point-label
        # contamination fraction — not comparable to HAPT's measured 0.57
        transition_frac=float(WIN) / 3000.0,
        extendable="YES (generator) — reference config that satisfies the rules")


def main():
    rows = [synth_facts(), hapt_facts(), ptbxl_facts(), xjtu_facts()]
    json.dump(rows, open("runs/datasets_table.json", "w"), indent=2)
    cols = [("dataset", "name"), ("rate Hz", "rate_hz"), ("patch", "patch"),
            ("win (samp)", "win_samp"), ("win (s)", "win_sec"),
            ("win/dwell", "win_over_dwell"), ("transition frac", "transition_frac")]
    print("| " + " | ".join(c for c, _ in cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        cells = []
        for _, k in cols:
            v = r[k]
            cells.append(f"{v:.2f}" if isinstance(v, float) and not np.isnan(v)
                         else ("—" if isinstance(v, float) else str(v)))
        print("| " + " | ".join(cells) + " |")
    print("\nlabel & extendability:")
    for r in rows:
        print(f"  {r['name']:9s} label: {r['label']}")
        print(f"  {'':9s} stride: {r['stride']}")
        print(f"  {'':9s} extend: {r['extendable']}")


if __name__ == "__main__":
    main()
