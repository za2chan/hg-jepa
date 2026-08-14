"""Synthetic data in the REAL-dataset npz format, so the SSL baselines can run on it.

The baseline adapters (`src/baselines/*_adapter.py`) consume an npz with W/lab/fast/
subj -- they cannot take the in-memory series `synth_robust.py` uses. Part 2 asks
for TS2Vec/PatchTST on all three datasets, so synthetic needs the same container.

Perturbation note: the Part 2 protocol perturbs with `scale` (W * (1+s)). On this
generator that is a well-targeted fast-factor attack -- the regime label is carried
by the carrier FREQUENCY, which a global gain does not touch, while the OU component
u is an amplitude. So the slow factor is preserved exactly and only amplitudes move.
That is NOT true on HAPT, where the activity label itself depends on acceleration
magnitude (see docs/NIGHT_REPORT_2026-08-04_ko.md).

Groups (`subj`) are contiguous TIME BLOCKS, not random: windows overlap by up to
L*P-1 samples, so a random split would leak almost the whole test set into train.
Window starts are held at least one full window before their block's end so no
window straddles a block boundary.

Usage: python3 make_synth_npz.py [gap] [n_win]
Writes data/synth_v2.npz
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hglp"))

import numpy as np

from datagen import make_dataset
from model import P, L

N_BLOCK = 30            # matches HAPT's subject count, so the C3 group split behaves alike
SERIES_SEED = 99        # the held-out evaluation series, never the training draw


def build(gap=0.03, n_win=3000, n_steps=300_000, seed=SERIES_SEED):
    d = make_dataset(n_steps, seed=seed, gap=gap)
    x, s, u = d["x"].astype(np.float32), d["s"], d["u"].astype(np.float32)
    span = L * P
    edges = np.linspace(0, n_steps - span - 1, N_BLOCK + 1).astype(np.int64)

    rng = np.random.default_rng(seed)
    starts, subj = [], []
    per = n_win // N_BLOCK
    for b in range(N_BLOCK):
        lo, hi = edges[b], edges[b + 1] - span      # no window crosses into block b+1
        if hi <= lo:
            raise ValueError("block shorter than one window; raise n_steps")
        starts.append(rng.integers(lo, hi, per))
        subj.append(np.full(per, b))
    starts = np.concatenate(starts); subj = np.concatenate(subj).astype(np.int64)

    idx = starts[:, None] + np.arange(span)[None]           # (N, L*P)
    W = x[idx].reshape(len(starts), L, P)
    # label / fast proxy read at the LAST sample of each patch -- the same position
    # convention train.py uses for its targets
    at = starts[:, None] + np.arange(L)[None] * P + (P - 1)
    lab = (s[at] + 1).astype(np.int64)                      # 1..3, so 0 stays "unlabeled"
    fast = u[at].astype(np.float32)
    return dict(W=W, lab=lab, fast=fast, subj=subj)


if __name__ == "__main__":
    gap = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
    n_win = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    out = build(gap, n_win)
    root = pathlib.Path(__file__).resolve().parent.parent.parent
    np.savez_compressed(root / "data/synth_v2.npz", **out)

    W, lab, subj = out["W"], out["lab"], out["subj"]
    print(f"W {W.shape}  lab {lab.shape} values {np.unique(lab)}  subj {len(np.unique(subj))} blocks")
    print(f"클래스 분포 (마지막 위치): {np.bincount(lab[:, -1])[1:]}")
    print(f"fast(u) 범위 [{out['fast'].min():.3f}, {out['fast'].max():.3f}]  sd {out['fast'].std():.3f}")
    # the split this npz will produce, and the leak it must not have
    rng = np.random.default_rng(0)
    test_g = set(rng.permutation(np.unique(subj))[:len(np.unique(subj)) // 3].tolist())
    is_te = np.array([g in test_g for g in subj])
    print(f"C3 분할: 학습 {int((~is_te).sum())}창 / 시험 {int(is_te.sum())}창, "
          f"시험 블록 {len(test_g)}개")
    assert lab.min() >= 1 and lab.max() <= 6, "label_range (1,6) 밖"
    assert not np.isnan(W).any(), "NaN in W"
    print(f"saved data/synth_v2.npz  (gap={gap}, {n_win} windows)")
