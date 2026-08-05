"""Shared data plumbing for the SSL baselines (TS2Vec, PatchTST).

FAIRNESS CONTRACT — every baseline must obey this, or the comparison is void:

1. Same input tensor. Windows come from the same npz, are normalised by the same
   A3 statistics (`train_real._norm_stats`, fit on the TRAIN split only), and are
   handed over as (N, L, C) exactly as our own encoder receives them. A baseline
   does NOT get to re-patch, re-normalise, or see a longer context.
2. Same training set. `train_idx` selects the windows the SSL objective may see.
   No baseline may train on windows outside it (domain_shift depends on this).
3. Same evaluation. The adapter returns per-position embeddings; scoring is done
   by the untouched probes in `hglp/probes.py`, at the same C_MIN floor.
4. Vendored upstream code is NOT edited. Adapters live here; the pinned commit and
   licence of each upstream repo are recorded in `third_party/PROVENANCE.md`.

Adapters must expose:

    fit_encoder(npz, n_ax, train_idx=None, seed=0, steps=..., **kw) -> dict with
        embed : callable(Wt_slice: (B, L, C) float tensor) -> (B, L, D) numpy/tensor
        Wt    : (N, L, C) normalised windows, on DEV
        lab   : (N, L) int   labels          (passed straight through)
        fast  : (N, L) float fast proxy      (passed straight through)
        tr, te: index arrays
        D     : int, embedding width
        meta  : dict — at minimum {"name", "commit", "causal": bool, "params": int}

`causal` matters: our encoder is causal, TS2Vec and PatchTST are bidirectional by
default. That is a real advantage for them at a given position, and it must be
reported next to the numbers rather than quietly ignored.
"""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hglp"))

from train_real import _norm_stats, _apply_norm, DEV      # identical normalisation


def load_windows(npz, n_ax, train_idx=None, seed=0):
    """(Wt, lab, fast, tr, te) — the same tensors train_real builds, same split rule."""
    d = np.load(npz)
    Wall, lab, fast, grp = d["W"], d["lab"], d["fast"], d["subj"]
    if train_idx is None:                       # C3 group split: hold out ~1/3 of groups
        rng = np.random.default_rng(seed)
        groups = np.unique(grp)
        test_g = set(rng.permutation(groups)[:max(1, len(groups) // 3)].tolist())
        is_test = np.array([g in test_g for g in grp])
    else:
        is_test = np.ones(len(grp), bool); is_test[train_idx] = False
    tr = np.flatnonzero(~is_test)
    mu, sd, per = _norm_stats(Wall[tr], n_ax)   # A3: fit on train only
    Wt = torch.from_numpy(_apply_norm(Wall, mu, sd, per, n_ax)).to(DEV)
    return Wt, lab, fast, tr, np.flatnonzero(is_test)


@torch.no_grad()
def embed_all(embed, Wt, bs=64):
    """Run an adapter's `embed` over every window -> (N, L, D) numpy."""
    out = []
    for i in range(0, len(Wt), bs):
        z = embed(Wt[i:i + bs])
        out.append(z.detach().cpu().numpy() if torch.is_tensor(z) else np.asarray(z))
    return np.concatenate(out)


def check_contract(res, Wt):
    """Cheap assertions every adapter's self-check should run."""
    X = Wt[:8]                                  # compare against the slice we embedded,
    Z = embed_all(res["embed"], X)              # not against the full N (never matched)
    assert Z.ndim == 3 and Z.shape[:2] == tuple(X.shape[:2]), \
        f"embed must return (B, L, D), got {Z.shape} for input {tuple(X.shape[:2])}"
    assert Z.shape[2] == res["D"], f"D mismatch: {Z.shape[2]} vs {res['D']}"
    assert np.isfinite(Z).all(), "non-finite embeddings"
    assert not np.allclose(Z.std(axis=(0, 1)), 0), "collapsed embedding (zero variance)"
    for k in ("Wt", "lab", "fast", "tr", "te", "meta"):
        assert k in res, f"missing key {k}"
    for k in ("name", "commit", "causal", "params"):
        assert k in res["meta"], f"meta.{k} missing"
    return Z.shape
