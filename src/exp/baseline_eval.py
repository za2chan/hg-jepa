"""Run TS2Vec / PatchTST against our encoder on the SAME windows, splits and probes.

Two fairness corrections are applied here rather than argued about in prose,
because both baselines turned out to be totally non-causal (measured in the
adapters: TS2Vec moves 100% of strictly-earlier positions, PatchTST moves the
first patch by 9x the signal scale when the last patch is perturbed):

  LAST-POSITION scoring — at position L-1 a causal encoder has also seen the whole
    window, so the baselines' access to the future buys them nothing there. At any
    interior position it does, which is why the multi-position numbers are
    reported as the baseline-FAVOURABLE bound rather than as the comparison.
  PCA-d matching — D is 320 (TS2Vec) and 1536 (PatchTST) against our 64, and a
    linear probe's label appetite is dominated by its input dimension. Comparing
    our 16-dim z_slow against a 1536-dim readout would flatter us for a reason
    that has nothing to do with the method. PCA is fit on TRAIN windows only and
    is label-free, so it is the same kind of post-hoc unmixing the rotation
    baseline already uses.

Usage: python3 baseline_eval.py [hapt] [steps] [n_draw] [seed]
Writes runs_v2/baseline_label_efficiency_<dataset>.json
"""
import json
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hglp"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "baselines"))

import numpy as np
from sklearn.decomposition import PCA

from model import D_SLOW, D_Z
from probes import rand_subspace
import label_efficiency as LE

DSET = {"hapt": ("../../data/hapt_v2.npz", 3,
                 dict(tau=40.0, w=12, dmin=12, dmax=128, min_context=16))}
PCA_DIMS = (D_SLOW, D_Z)          # 16 = matched to d_slow, 64 = matched to z_full
FIT_ROWS = 50_000                 # rows PCA is fitted on; full transform is batched


def pca_arms(Z, tr, dims=PCA_DIMS, seed=0):
    """Label-free PCA-d projections, FIT ON TRAIN WINDOWS ONLY, memory-bounded.

    PatchTST's native embedding is (n, L, 1536) ~ 3.3 GB on HAPT, so the fit is on
    a subsample of train rows and the transform runs window-batched."""
    n, L, D = Z.shape
    rng = np.random.default_rng(seed)
    rows = Z[tr].reshape(-1, D)
    if len(rows) > FIT_ROWS:
        rows = rows[rng.choice(len(rows), FIT_ROWS, replace=False)]
    out = {}
    for d in dims:
        if d >= D:                                     # nothing to project down to
            continue
        p = PCA(n_components=d).fit(rows)
        out[d] = np.concatenate([p.transform(Z[i:i + 256].reshape(-1, D))
                                 .reshape(-1, L, d).astype(np.float32)
                                 for i in range(0, n, 256)])
    return out


def our_arms(npz, n_ax, kw, stem, seed, steps):
    from train_real import train_real
    from probes import encode_all
    lk, te_ = stem.split("+")
    g = train_real(npz, n_ax=n_ax, seed=seed, loss_kind=lk, target_enc=te_,
                   steps=steps, log_every=10 ** 9, **kw)
    u = train_real(npz, n_ax=n_ax, seed=seed, loss_kind=lk, target_enc=te_,
                   steps=steps, gate=False, xcov=False, log_every=10 ** 9, **kw)
    Zg, Zu = encode_all(g["enc"], g["Wt"]), encode_all(u["enc"], u["Wt"])
    Q = rand_subspace(D_Z, D_SLOW, seed)              # the null every 16-dim arm faces
    return ({"ours/z_slow": Zg[..., :D_SLOW], "ours/z_full": Zg,
             "ours/rand16": Zg @ Q, "ungated/z_full": Zu,
             "ungated/pca16": pca_arms(Zu, g["tr"], (D_SLOW,), seed)[D_SLOW]},
            g["lab"], g["tr"], g["te"], g["Wt"])


def baseline_arms(name, npz, n_ax, train_idx, seed, steps):
    """One SSL baseline -> {arm: (n, L, d)}. Native width kept only if affordable."""
    mod = __import__(f"{name}_adapter")
    r = mod.fit_encoder(npz, n_ax, train_idx=train_idx, seed=seed, steps=steps)
    from common import embed_all
    Z = embed_all(r["embed"], r["Wt"]).astype(np.float32)
    arms = {f"{name}/pca{d}": V for d, V in pca_arms(Z, train_idx, seed=seed).items()}
    if Z.nbytes < 1.5e9:                              # TS2Vec 690 MB yes, PatchTST 3.3 GB no
        arms[f"{name}/native{Z.shape[-1]}"] = Z
    return arms, r["meta"]


def run(dataset="hapt", steps=2500, n_draw=LE.N_DRAW, seed=0, stem="nce+ema"):
    npz, n_ax, kw = DSET[dataset]
    feats, lab, tr, te, Wt = our_arms(npz, n_ax, kw, stem, seed, steps)
    metas = {}
    for name in ("ts2vec", "patchtst"):
        a, m = baseline_arms(name, npz, n_ax, tr, seed, steps)
        feats.update(a); metas[name] = m
        print(f"  {name}: {list(a)} causal={m['causal']} params={m['params']}", flush=True)
    L = lab.shape[1]
    out = {}
    for tag, c_min in (("multi_position", LE.C_MIN), ("last_position", L - 1)):
        cells, meta = LE.label_efficiency(feats, lab, tr, te, n_draw=n_draw,
                                          seed=seed, c_min=c_min)
        agg = LE.summarize([cells], meta["budgets"])
        out[tag] = dict(agg=agg, meta=meta, crossings=LE.crossings(agg))
        print(f"\n### {tag} (c_min={c_min}) ###", flush=True)
        LE.report(agg, meta, out[tag]["crossings"])
    return dict(result=out, baselines=metas,
                config=dict(dataset=dataset, stem=stem, steps=steps,
                            n_draw=n_draw, seed=seed))


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "hapt"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
    nd = int(sys.argv[3]) if len(sys.argv) > 3 else LE.N_DRAW
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    os.makedirs("../../runs_v2", exist_ok=True)
    res = run(ds, steps, nd, seed)
    p = f"../../runs_v2/baseline_label_efficiency_{ds}_s{seed}.json"
    json.dump(res, open(p, "w"), indent=2)
    print("saved", p)
