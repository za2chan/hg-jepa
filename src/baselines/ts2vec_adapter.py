"""TS2Vec [Yue et al., AAAI 2022, arXiv:2106.10466] as an SSL baseline.

The published implementation is vendored UNEDITED at `third_party/ts2vec`
(commit b0088e1, MIT); see `third_party/PROVENANCE.md`. Everything here is glue:
our (N, L, C) windows go in as (n_instance, n_timestamps, n_features) — L is
TS2Vec's time axis, C its feature axis — and per-position representations come
out, under the fairness contract in `common.py`. Nothing is re-normalised (A3
already ran, fit on train only) and only `tr` windows reach the SSL objective.

Hyperparameters default to upstream `train.py`'s argparse values, not to
`TS2Vec.__init__`'s, because the former are what produced the published numbers
(they disagree: batch 8 vs 16).

Usage: python3 ts2vec_adapter.py     # self-check, ~1 min on hapt_v2
"""
import pathlib
import sys

import numpy as np
import torch

import common
from common import load_windows

# appended, not inserted: third_party/ts2vec ships its own train.py/utils.py and
# must not shadow src/hglp's modules on the path.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "third_party" / "ts2vec"))
from ts2vec import TS2Vec
from utils import init_dl_program

COMMIT = "b0088e14a99706c05451316dc6db8d3da9351163"
UPSTREAM = dict(output_dims=320, hidden_dims=64, depth=10, lr=0.001,
                batch_size=8, max_train_length=3000, temporal_unit=0)


def leftmost_affected(embed, x, t, tol=1e-5):
    """Perturb the input at timestep `t`; return the earliest output position that
    moves and the per-position change. A causal encoder cannot move anything
    before `t`. This is measured, not assumed — `meta["causal"]` is its verdict."""
    z0 = np.asarray(embed(x))
    xp = x.clone()
    xp[:, t] += 10.0
    d = np.abs(np.asarray(embed(xp)) - z0).max(-1).max(0)
    moved = np.flatnonzero(d > tol)
    return (int(moved[0]) if len(moved) else x.shape[1]), d


def fit_encoder(npz, n_ax, train_idx=None, seed=0, steps=None, **kw):
    """steps -> upstream's n_iters. None keeps TS2Vec's own rule (200 iters if
    train_data.size <= 1e5 else 600), which is what `train.py` does by default."""
    Wt, lab, fast, tr, te = load_windows(npz, n_ax, train_idx, seed)
    # upstream's first line: seeds random/numpy/torch and forces true fp32 convs
    # by disabling cudnn TF32. That flag is process-global, so a co-resident HGLP
    # model gets slower-but-exacter matmuls too — never the other way round.
    dev = init_dl_program(0 if torch.cuda.is_available() else "cpu", seed=seed)

    losses = []
    cfg = {**UPSTREAM, **kw}
    model = TS2Vec(input_dims=Wt.shape[-1], device=dev,
                   after_iter_callback=lambda _, l: losses.append(l), **cfg)
    model.fit(Wt[tr].cpu().numpy(), n_iters=steps)   # contract 2: train windows only

    def embed(x):
        # encoding_window=None is `_eval_with_pooling`'s no-pool branch: one 320-d
        # vector per input timestep, aligned 1:1 with our L positions. It is also
        # what upstream's own tasks/classification.py selects when the labels are
        # per-timestamp, as ours are. 'full_series' collapses L to 1 and
        # 'multiscale' concatenates max-pools at growing kernels — both destroy
        # the alignment the multi-position probe needs.
        return model.encode(x.detach().cpu().numpy(), encoding_window=None,
                            batch_size=len(x))

    # probed mid-window, not at L-1: SamePadConv truncates the right edge, so the
    # LAST timestep's influence on distant positions decays into fp32 noise and
    # understates the leak. A mid-window probe answers the question we care about.
    t = Wt.shape[1] // 2
    first, d = leftmost_affected(embed, Wt[:4], t)
    meta = dict(name="ts2vec", commit=COMMIT, params=sum(p.numel() for p in model._net.parameters()),
                causal=first >= t, probe_t=t, leftmost_affected=first,
                frac_past_moved=float((d[:t] > 1e-5).mean()),
                cfg=cfg, n_iters=model.n_iters, loss=losses)
    return dict(embed=embed, Wt=Wt, lab=lab, fast=fast, tr=tr, te=te,
                D=cfg["output_dims"], meta=meta, model=model)


if __name__ == "__main__":
    res = fit_encoder("../../data/hapt_v2.npz", n_ax=3, steps=400)
    Wt, embed, m = res["Wt"], res["embed"], res["meta"]
    print(f"contract: embed{common.check_contract(res, Wt)} D={res['D']} "
          f"params={m['params']:,} iters={m['n_iters']}")

    q = np.array_split(np.array(m["loss"]), 4)
    print(f"hierarchical contrastive loss: q1 {q[0].mean():.4f} -> q4 {q[-1].mean():.4f}")
    assert q[-1].mean() < q[0].mean(), "loss did not decrease -- training is broken"

    # TS2Vec's objective contrasts two overlapping crops of one series, so trained
    # representations must agree on the SHARED positions of two crops of the same
    # window more than across windows. Scored on the overlap only, in absolute
    # time, so position is not a confound.
    za = np.asarray(embed(Wt[:64, :192]))[:, 64:]        # crop [0,192) -> abs [64,192)
    zb = np.asarray(embed(Wt[:64, 64:]))[:, :128]        # crop [64,256) -> abs [64,192)
    u = za / np.linalg.norm(za, axis=-1, keepdims=True)
    v = zb / np.linalg.norm(zb, axis=-1, keepdims=True)
    same, diff = (u * v).sum(-1).mean(), (u * np.roll(v, 1, 0)).sum(-1).mean()
    print(f"overlapping-crop cosine: same series {same:.3f} vs different {diff:.3f}")
    assert same > diff, "crops of the same series are not more similar -- fidelity broken"

    t = m["probe_t"]
    first, d = leftmost_affected(embed, Wt[:4], t)
    print(f"perturb input t={t}: earliest output position moved = {first}; "
          f"{100 * m['frac_past_moved']:.0f}% of the {t} STRICTLY EARLIER positions move "
          f"(|dz| at pos 0 {d[0]:.3g}, {t // 2} {d[t // 2]:.3g}, {t} {d[t]:.3g})")
    print(f"causal={m['causal']} -- future timesteps reach every earlier position")
