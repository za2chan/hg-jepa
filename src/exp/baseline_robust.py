"""Perturbation robustness with TS2Vec / PatchTST alongside our encoder.

The claim being tested is narrow and was inferred from real data post-hoc:
z_slow helps when the corruption targets the FAST factor (PTB-XL gain, +0.101 at
strength 2.0) and hurts when it targets the slow one (baseline wander, -0.037).
A reviewer's next question is whether a standard SSL representation would show
the same thing for free, so the baselines get the identical treatment.

Two fairness corrections, same as baseline_eval.py and for the same measured
reason (both baselines are totally non-causal):
  * scored at the LAST position only, where a causal encoder has seen the whole
    window too, so their access to the future buys nothing;
  * PCA-16 / PCA-64, fit on CLEAN TRAIN embeddings only, because D is 320 and
    1536 against our 64 and probe capacity would otherwise drive the comparison.

PCA is fit on clean data and then applied to perturbed data — refitting it per
strength would let the baseline adapt to the corruption, which our z_slow cannot.

Usage: python3 baseline_robust.py [ptbxl|hapt] [steps] [n_seed]
Writes runs_v2/baseline_robust_<dataset>.json
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import D_SLOW, D_Z
from probes import rand_subspace

STRENGTHS = (0.0, 0.25, 0.5, 1.0, 2.0)
KINDS = ("noise", "scale", "wander")
DSET = {"ptbxl": ("../../data/ptbxl_v2.npz", 1,
                  dict(tau=16.0, w=8, dmin=8, dmax=48, min_context=8)),
        "sleepedf": ("../../data/sleepedf_v2.npz", 3,
                     dict(tau=24.0, w=12, dmin=12, dmax=128, min_context=16)),
        "hapt": ("../../data/hapt_v2.npz", 3,
                 dict(tau=40.0, w=12, dmin=12, dmax=128, min_context=16)),
        # synthetic in the same container so the SSL baselines can run on it too
        # (built by src/prep/make_synth_npz.py; hyperparameters are train.py's own
        # synthetic defaults, so the model is unchanged -- only the data path is)
        "synth": ("../../data/synth_v2.npz", 1,
                  dict(tau=16.0, w=8, dmin=8, dmax=128, min_context=16))}


def perturb(W, kind, s, g, n_ax=None):
    if kind == "noise":
        return W + s * torch.randn(W.shape, generator=g, device=W.device)
    if kind == "scale":
        return W * (1.0 + s)
    if kind == "rotate":
        # Rotate the sensor frame. This is the physically real shift for a
        # body-worn accelerometer -- the device sits at a different angle -- and it
        # is the right control for our claim: a rotation is norm-preserving, so the
        # transient proxy ||a|| is INVARIANT under it. The gain perturbation we used
        # before rescales ||a|| directly, which means it moves the very quantity the
        # exclusion term is measured against; the shift and the measurement were
        # entangled. Here they are not.
        #
        # `s` is the rotation angle in radians about a fixed axis (0.5*s*pi keeps
        # s=2 at a right angle, matching the old sweep's endpoint in spirit).
        assert n_ax == 3, "rotate is defined for 3-axis accelerometer data only"
        th = 0.5 * s * np.pi
        c, sn = float(np.cos(th)), float(np.sin(th))
        R = torch.tensor([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]],
                         dtype=W.dtype, device=W.device)
        n, L_, D = W.shape                       # D = patch_len * n_ax, axis fastest
        return (W.reshape(n, L_, D // n_ax, n_ax) @ R.T).reshape(n, L_, D)
    ph = torch.linspace(0, 2 * np.pi, W.shape[1], device=W.device)[None, :, None]
    return W + s * torch.sin(ph)                 # baseline wander: ~1 cycle per record


@torch.no_grad()
def last(embed, W, bs=64):
    """Last-position embedding — the only position where causal and bidirectional
    encoders have seen the same input."""
    out = []
    for i in range(0, len(W), bs):
        z = embed(W[i:i + bs])
        out.append((z if torch.is_tensor(z) else torch.as_tensor(z))[:, -1].cpu().numpy())
    return np.concatenate(out)


def build(dataset, seed, steps, stem="nce+ema", baselines=True):
    """{arm: (embed_fn, projection or None)} plus the shared windows and split."""
    npz, n_ax, kw = DSET[dataset]
    from train_real import train_real
    from common import load_windows
    lk, tgt = stem.split("+")
    Wt, lab, fast, tr, te = load_windows(npz, n_ax, seed=seed)
    r = train_real(npz, n_ax=n_ax, seed=seed, loss_kind=lk, target_enc=tgt,
                   steps=steps, log_every=10 ** 9, **kw)
    assert np.array_equal(r["tr"], tr), "our split differs from the shared one"
    Q = rand_subspace(D_Z, D_SLOW, seed)
    arms = {"ours/z_slow": (r["enc"], np.eye(D_Z)[:, :D_SLOW]),
            "ours/z_full": (r["enc"], None),
            "ours/rand16": (r["enc"], Q)}
    for name in (("ts2vec", "patchtst") if baselines else ()):
        mod = __import__(f"{name}_adapter")
        b = mod.fit_encoder(npz, n_ax, train_idx=tr, seed=seed, steps=steps)
        Zc = last(b["embed"], Wt)
        for d in (D_SLOW, D_Z):                  # PCA fit on CLEAN TRAIN rows only
            arms[f"{name}/pca{d}"] = (b["embed"], PCA(n_components=d)
                                      .fit(Zc[tr]).components_.T)
        print(f"  {name}: D={b['D']} causal={b['meta']['causal']}", flush=True)
    return arms, Wt, lab, tr, te


def run(dataset, seed, steps, stem="nce+ema", baselines=True):
    arms, Wt, lab, tr, te = build(dataset, seed, steps, stem, baselines)
    n_ax = DSET[dataset][1]
    # A sensor-axis rotation is only defined for 3-axis accelerometry, and it is the
    # one perturbation here that leaves ||a|| untouched -- `scale` at s=1.0 doubles
    # the norm, i.e. it moves the very quantity the transient proxy is built from.
    kinds = KINDS + (("rotate",) if n_ax == 3 else ())
    y = lab[:, -1] if dataset == "ptbxl" else lab[:, -1]
    g = torch.Generator(device=Wt.device).manual_seed(seed)
    clf, out = {}, {}
    for a, (embed, P_) in arms.items():
        Z = last(embed, Wt)
        F = Z if P_ is None else Z @ P_
        clf[a] = make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=1000,
                                                  class_weight="balanced")).fit(F[tr], y[tr])
    for kind in kinds:
        for s in STRENGTHS:
            Wp = perturb(Wt[te], kind, s, g, n_ax=n_ax)
            for a, (embed, P_) in arms.items():
                Z = last(embed, Wp)
                F = Z if P_ is None else Z @ P_
                out.setdefault(a, {})[f"{kind}@{s}"] = float(
                    f1_score(y[te], clf[a].predict(F), average="macro"))
        print(f"  {kind} done", flush=True)
    return out


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "ptbxl"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
    n_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    stem = sys.argv[4] if len(sys.argv) > 4 else "nce+ema"
    baselines = (sys.argv[5] != "0") if len(sys.argv) > 5 else True
    os.makedirs("../../runs_v2", exist_ok=True)
    cells = []
    for s in range(n_seed):
        print(f"=== {ds} seed {s} ({stem}) ===", flush=True)
        cells.append(run(ds, s, steps, stem, baselines))
    arms = list(cells[0])
    keys = list(cells[0][arms[0]])
    agg = {a: {k: float(np.mean([c[a][k] for c in cells])) for k in keys} for a in arms}
    json.dump(dict(agg=agg, cells=cells,
                   config=dict(dataset=ds, steps=steps, n_seed=n_seed, stem=stem,
                               baselines=baselines,
                               strengths=list(STRENGTHS), kinds=sorted({k.split("@")[0] for k in keys}),
                               scored_at="last position", pca="fit on clean train")),
              open(f"../../runs_v2/robust_{ds}_{stem}"
                   f"{'_wb' if baselines else ''}.json", "w"), indent=2)
    print(f"\n{'조건':14}" + "".join(f"{a:>18}" for a in arms))
    for k in keys:
        print(f"{k:14}" + "".join(f"{agg[a][k]:18.3f}" for a in arms))
    print(f"\nsaved runs_v2/robust_{ds}_{stem}{'_wb' if baselines else ''}.json")
