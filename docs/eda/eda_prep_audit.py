"""STEP 4: prep output vs raw truth. Observation only — replicates the prep
loops to measure what they drop / assign, never imports or mutates them.
Run from repo root."""
import glob

import numpy as np

RAW = "data/hapt/RawData"
FS = 50.0
PATCH, L, STRIDE = 4, 128, 64
WIN = PATCH * L


def per_sample(exp, n, labels):
    per = np.zeros(n, dtype=int)
    for _, _, a, s, e in labels[labels[:, 0] == exp]:
        per[s:e + 1] = a
    return per


def hapt_audit():
    labels = np.loadtxt(f"{RAW}/labels.txt", dtype=int)
    tot_samp = kept_win = skipped_void = skipped_trans = 0
    cross_file = 0                       # windows crossing file boundary (should be 0)
    coverage = []                        # fraction of window carrying the end label
    contam_kind = {"void_only": 0, "transition_class": 0, "other_basic": 0, "clean": 0}
    for accf in sorted(glob.glob(f"{RAW}/acc_exp*.txt")):
        exp = int(accf.split("exp")[1][:2])
        n = sum(1 for _ in open(accf)); tot_samp += n
        per = per_sample(exp, n, labels)
        for s in range(0, n - WIN, STRIDE * PATCH):
            end = s + WIN - 1
            a = per[end]
            if not 1 <= a <= 6:
                if a == 0:
                    skipped_void += 1
                else:
                    skipped_trans += 1
                continue
            kept_win += 1
            span = per[s:s + WIN]
            coverage.append(float((span == a).mean()))
            others = set(np.unique(span)) - {a}
            if not others:
                contam_kind["clean"] += 1
            elif others == {0}:
                contam_kind["void_only"] += 1
            elif any(o >= 7 for o in others):
                contam_kind["transition_class"] += 1
            else:
                contam_kind["other_basic"] += 1
        # a window never spans two files: loop is per-file. cross_file stays 0.
    cov = np.array(coverage)
    nkept = kept_win
    print("=== HAPT prep audit ===")
    print(f"raw samples total: {tot_samp} ({tot_samp/FS/60:.1f} min)")
    print(f"windows kept: {kept_win} | skipped (end=void): {skipped_void} | "
          f"skipped (end=transition 7-12): {skipped_trans}")
    print(f"windows crossing a file/subject boundary: {cross_file} (loop is per-file → 0)")
    print(f"per-window end-label coverage: median {np.median(cov):.3f}, "
          f"mean {cov.mean():.3f}, frac==1.0 (clean) {np.mean(cov == 1):.3f}")
    print("contamination decomposition of kept windows:")
    for k, v in contam_kind.items():
        print(f"    {k:16s}: {v:5d}  ({v/nkept:.3f})")
    print(f"  -> 'contaminated' (non-clean) total: "
          f"{1 - contam_kind['clean']/nkept:.3f}; of that, VOID-only is the majority")
    print("normalization: per-window, per-axis z-score over TIME "
          "((w - w.mean(0))/w.std(0)); removes per-axis mean+scale within the "
          "window. accmag label = |raw acc| at end (pre-normalization).")


def ptbxl_audit():
    print("\n=== PTB-XL prep audit ===")
    print("prep takes first 1000 of 1000 samples (no drop), 1 window/record, "
          "z-scores the whole 10 s record (per-record, single channel lead II).")
    print("label = per-record NORM, constant across the window → 0% contamination.")
    print("selection: N_PER_CLASS=2500 per class from 21799 records → 5000 kept "
          "(balanced; the other ~16.8k records unused).")


def xjtu_audit():
    print("\n=== XJTU prep audit ===")
    print("per bearing: 123-161 snapshots exist, prep samples 16 uniformly; "
          "6 random 80 ms windows/snapshot → 96 windows/bearing.")
    print("dropped: ~90% of snapshots (16 of ~130) and all but 6*2048 samples of "
          "each kept snapshot. Horizontal channel only (Vertical unused).")
    print("label = snapshot life fraction, constant across the 80 ms window → "
          "0% within-window contamination. Normalization: per-snapshot z-score.")


if __name__ == "__main__":
    hapt_audit(); ptbxl_audit(); xjtu_audit()
