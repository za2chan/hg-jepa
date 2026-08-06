"""Do standard SSL representations separate the two factors, given the same
label-free post-hoc unmixing our rotation baseline uses?

This is the surviving Part 1 question applied to TS2Vec and PatchTST. The
existing baseline runs (baseline_eval, baseline_robust) attached them to the
label-efficiency and robustness claims, both of which are now retracted, so they
say nothing about the mechanism. Here they are scored by exactly the evaluation
that carries Part 1: a 16-dim "slow" half, its complement, and a dimension-matched
random null.

Fairness, unchanged from the other baseline drivers and for the same measured
reason (both baselines are totally non-causal):
  * scored at the LAST position, where a causal encoder has seen the whole window
    too, so their access to the future buys nothing;
  * their slow half is PCA-16 fit on TRAIN rows only -- label-free, and the same
    kind of post-hoc unmixing rotation.py already applies to our own ungated
    embedding. SFA and slowness-ranked ICA are included too, since those are the
    strong ones on our own embedding.

Synthetic is absent because the adapters consume an npz with W/lab/fast/subj and
the synthetic generator is not stored that way; PTB-XL is the informative case
anyway (its random-16 null is 0.40, against HAPT's 0.04).

Usage: python3 baseline_separation.py [ptbxl|hapt] [steps] [n_seed]
Writes runs_v2/baseline_separation_<dataset>.json
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hglp"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "baselines"))

import numpy as np
import torch
from sklearn.decomposition import PCA

from model import D_SLOW, D_Z
from probes import score, rand_subspace
from rotation import unmixings
from baseline_robust import DSET, last

SEEDS_DEFAULT = 3


def halves(Z, tr, d=D_SLOW):
    """Label-free splits of one embedding into a d-dim 'slow' half + complement.
    PCA/ICA/SFA all fit on TRAIN rows only; `random` is the null."""
    n = Z.shape[1]
    pairs = np.stack([np.arange(len(tr) - 1), np.arange(1, len(tr))], 1)  # adjacent records
    out = {}
    for m, (a, b) in unmixings(Z[tr], pairs, d).items():
        out[m] = (a, b)
    Q = rand_subspace(n, n, 0)
    out["random"] = (Q[:, :d], Q[:, d:])
    return out


def run(dataset, seed, steps):
    npz, n_ax, kw = DSET[dataset]
    from train_real import train_real
    from common import load_windows
    Wt, lab, fast, tr, te = load_windows(npz, n_ax, seed=seed)
    y, z = lab[:, -1], fast[:, -1]
    arg = lambda F: (F[tr], y[tr], z[tr], F[te], y[te], z[te])
    out = {}

    for stem in ("l1+ema", "nce+ema"):                 # our gate, coordinate split
        lk, tg = stem.split("+")
        r = train_real(npz, n_ax=n_ax, seed=seed, loss_kind=lk, target_enc=tg,
                       steps=steps, log_every=10 ** 9, **kw)
        Z = last(r["enc"], Wt)
        out[f"ours-{stem}/gate"] = (score(*arg(Z[:, :D_SLOW])), score(*arg(Z[:, D_SLOW:])))
        Q = rand_subspace(D_Z, D_Z, seed)
        out[f"ours-{stem}/random"] = (score(*arg(Z @ Q[:, :D_SLOW])),
                                      score(*arg(Z @ Q[:, D_SLOW:])))

    for name in ("ts2vec", "patchtst"):
        mod = __import__(f"{name}_adapter")
        b = mod.fit_encoder(npz, n_ax, train_idx=tr, seed=seed, steps=steps)
        Z = last(b["embed"], Wt)
        for m, (Ws, Wf) in halves(Z, tr).items():
            out[f"{name}/{m}"] = (score(*arg(Z @ Ws)), score(*arg(Z @ Wf)))
        print(f"  {name}: D={b['D']} causal={b['meta']['causal']}", flush=True)
    return out


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "ptbxl"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
    n_seed = int(sys.argv[3]) if len(sys.argv) > 3 else SEEDS_DEFAULT
    os.makedirs("../../runs_v2", exist_ok=True)
    cells = []
    for s in range(n_seed):
        print(f"=== {ds} seed {s} ===", flush=True)
        cells.append(run(ds, s, steps))
    keys = list(cells[0])
    agg = {k: dict(slow=float(np.mean([c[k][0][0] for c in cells])),
                   fast=float(np.mean([c[k][0][1] for c in cells])),
                   compl_slow=float(np.mean([c[k][1][0] for c in cells])),
                   compl_fast=float(np.mean([c[k][1][1] for c in cells]))) for k in keys}
    json.dump(dict(agg=agg, cells=[{k: [list(v[0]), list(v[1])] for k, v in c.items()}
                                   for c in cells],
                   config=dict(dataset=ds, steps=steps, n_seed=n_seed,
                               scored_at="last position", d_slow=D_SLOW)),
              open(f"../../runs_v2/baseline_separation_{ds}.json", "w"), indent=2)

    # A negative fast R2 means the ridge does worse than predicting the mean, i.e.
    # the subspace carries nothing -- it does NOT mean "more excluded than empty".
    # Left unclipped it turns a bad fit into an exclusion bonus (measured: TS2Vec
    # +SFA showed a margin of 1.13 off an R2 of -0.22).
    clip = lambda v: max(v, 0.0)
    print(f"\n=== {ds}: 16차원 '느린' 절반 + 여집합 ({n_seed} seeds, 마지막 위치) ===")
    print(f"{'소스/방법':26}{'느림↑':>9}{'빠름↓':>9}{'기준선':>9}{'배제여유':>10}{'여집합빠름↑':>12}")
    for k, v in agg.items():
        src = k.split("/")[0]
        null = clip(agg[f"{src}/random"]["fast"])
        print(f"{k:26}{v['slow']:9.3f}{clip(v['fast']):9.3f}{null:9.3f}"
              f"{null - clip(v['fast']):10.3f}{clip(v['compl_fast']):12.3f}")
    print("\n  빠른 인자 R²는 0에서 자릅니다 — 음수는 '더 배제'가 아니라 '적합 실패'입니다.")
    print(f"\nsaved runs_v2/baseline_separation_{ds}.json")
