"""Shared probe / evaluation logic (protocol C1-C4).

C1: MULTI-POSITION probing with a CONTEXT FLOOR. Every labeled position at or
    beyond `c_min` is one evaluation sample, scored against that position's own
    label. The floor exists because position p has only p+1 patches of context;
    below some floor the slow state is not yet observable and scoring there
    measures the floor, not the representation. C_MIN is reported in the paper.
C3: probes fit on the train group, scored on held-out groups.
C4: absolute scores, chance stated, RankMe as the collapse monitor.
"""
import numpy as np
import torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import D_Z, D_SLOW

C_MIN = 64          # context floor in patches (protocol C1; swept in the appendix)
# NOTE the paper calls "z_fast" **z_mix** (decided 2026-08-06): the gate exposes
# that block only for dT < tau, and near-horizon prediction needs the current
# slow state as much as the fast one, so it holds BOTH factors (slow score
# 0.36-0.87 measured). Only its horizon availability is constrained, not its
# content. The code key stays `z_fast` -- every stored run JSON uses it and
# CLAUDE.md 4-6 bans repo-wide renames during the sprint. Rename post-deadline.
BLOCKS = {"z_slow": slice(0, D_SLOW), "z_fast": slice(D_SLOW, D_Z),
          "z_full": slice(0, D_Z)}


def rankme(Z, eps=1e-7):
    s = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
    p = s / (s.sum() + eps) + eps
    return float(np.exp(-(p * np.log(p)).sum()))


@torch.no_grad()
def encode_all(enc, Wt, bs=128):
    return torch.cat([enc(Wt[i:i + bs]) for i in range(0, len(Wt), bs)]).cpu().numpy()


def _subsample(rng, n, cap):
    return rng.choice(n, cap, replace=False) if (cap and n > cap) else np.arange(n)


def multi_position(Z, lab, fast, tr, te, block="z_slow", c_min=C_MIN,
                   label_range=(1, 6), max_samples=60000, seed=0,
                   classification=True):
    """Z: (n, L, D_Z) encodings. lab/fast: (n, L) per-position label / fast proxy.
    Returns slow-kept, leak, RankMe, plus the chance level and n used."""
    Lw = Z.shape[1]
    assert c_min < Lw, f"context floor {c_min} >= window {Lw}"
    B = Z[:, :, BLOCKS[block]]
    lo, hi = label_range
    ok = (lab >= lo) & (lab <= hi)
    ok[:, :c_min] = False                      # C1: enforce the context floor
    rng = np.random.default_rng(seed)

    def gather(idx):
        m = ok[idx]
        f, y, z = B[idx][m], lab[idx][m] - lo, fast[idx][m]
        s = _subsample(rng, len(f), max_samples)
        return f[s], y[s], z[s]

    ftr, ytr, ztr = gather(tr); fte, yte, zte = gather(te)
    out = dict(block=block, c_min=int(c_min), n_train=int(len(ftr)), n_test=int(len(fte)))
    if classification:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, class_weight="balanced"))
        lbl = np.unique(ytr)                                      # C4: fixed label set
        out["slow_kept_f1"] = float(f1_score(yte, clf.fit(ftr, ytr).predict(fte),
                                             average="macro", labels=lbl, zero_division=0))
        out["chance_f1"] = float(1.0 / len(np.unique(ytr)))       # balanced macro-F1
    else:
        rg = make_pipeline(StandardScaler(), Ridge())
        out["slow_kept_r2"] = float(rg.fit(ftr, ytr).score(fte, yte))
        out["chance_r2"] = 0.0
    out["leak_r2"] = float(make_pipeline(StandardScaler(), Ridge())
                           .fit(ftr, ztr).score(fte, zte))
    out["rankme"] = float(rankme(fte[:5000]))
    return out


def score(ftr, ytr, ztr, fte, yte, zte, classification=True):
    """(slow-kept, fast-carried) for ONE feature block. Both factors, always."""
    if classification:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, class_weight="balanced"))
        kept = float(f1_score(yte, clf.fit(ftr, ytr).predict(fte), average="macro",
                              labels=np.unique(ytr), zero_division=0))
    else:
        kept = float(make_pipeline(StandardScaler(), Ridge()).fit(ftr, ytr).score(fte, yte))
    fast = float(make_pipeline(StandardScaler(), Ridge()).fit(ftr, ztr).score(fte, zte))
    return kept, fast


def rand_subspace(D, d, seed):
    """Orthonormal basis of a random d-dim subspace of R^D (QR of a Gaussian)."""
    g = np.random.default_rng(seed).standard_normal((D, d))
    return np.linalg.qr(g)[0]


def block_factor(Ftr, ytr, ztr, Fte, yte, zte, d_slow=D_SLOW, n_rand=3, seed=0,
                 classification=True, extra=None, max_samples=60000):
    """BLOCK x FACTOR matrix with a DIMENSION-MATCHED RANDOM-SUBSPACE NULL.

    Scoring only z_slow (slow high, fast low) cannot distinguish real separation
    from three impostors: a model that never encoded the fast factor anywhere, a
    subspace that excludes it by accident (low-variance directions), and a fast
    proxy so weak that ANY subspace scores low on it. Each row here is scored on
    BOTH factors, and `rand<d>` is what an arbitrary d-dim readout of the SAME
    embedding gives -- the reference every other row must beat.

    Ftr/Fte are the FULL (n, D_Z) embeddings at the scored positions.
    `extra`: {name: (train_feats, test_feats)} for post-hoc subspaces.
    """
    # same cap as multi_position (C1): this fits ~20 probes per call, and HAPT has
    # ~270k labelled positions, so the uncapped version is hours of LogisticRegression
    rng = np.random.default_rng(seed)
    a, b = _subsample(rng, len(Ftr), max_samples), _subsample(rng, len(Fte), max_samples)
    Ftr, ytr, ztr, Fte, yte, zte = Ftr[a], ytr[a], ztr[a], Fte[b], yte[b], zte[b]
    D = Ftr.shape[1]
    rows = {"z_slow": (Ftr[:, :d_slow], Fte[:, :d_slow]),
            "z_fast": (Ftr[:, d_slow:], Fte[:, d_slow:]),
            "z_full": (Ftr, Fte)}
    rows.update(extra or {})
    out = {k: score(a, ytr, ztr, b, yte, zte, classification) for k, (a, b) in rows.items()}
    for d in (d_slow, D - d_slow):                     # null at BOTH block widths
        cells = [score(Ftr @ Q, ytr, ztr, Fte @ Q, yte, zte, classification)
                 for Q in (rand_subspace(D, d, seed + 1000 * i) for i in range(n_rand))]
        out[f"rand{d}"] = (float(np.mean([c[0] for c in cells])),
                           float(np.mean([c[1] for c in cells])))

    # `rand<d>` above are INDEPENDENT draws at each width, not a split -- pairing
    # them would mix two unrelated subspaces, so they cannot form a SEP. This is a
    # real random SPLIT: one full rotation cut into d and D-d, i.e. the same
    # coordinate-split operation applied to an arbitrary basis. It is the null SEP
    # every cell must beat, and on HAPT it is high enough (~0.34-0.54) that a table
    # without it reads far too favourably.
    pairs = []
    for i in range(n_rand):
        Q = rand_subspace(D, D, seed + 7919 * i)       # full-rank rotation
        pairs.append((score(Ftr @ Q[:, :d_slow], ytr, ztr, Fte @ Q[:, :d_slow], yte, zte,
                            classification),
                      score(Ftr @ Q[:, d_slow:], ytr, ztr, Fte @ Q[:, d_slow:], yte, zte,
                            classification)))
    mean = lambda f: (float(np.mean([f(p)[0] for p in pairs])),
                      float(np.mean([f(p)[1] for p in pairs])))
    out["randsplit_slow"] = mean(lambda p: p[0])       # the "z_slow" half of the split
    out["randsplit_fast"] = mean(lambda p: p[1])       # its complement
    return out


def sep_index(bf, fast_ceiling=None, fast_null=None):
    """One scalar summarising the block x factor matrix. In [0, 1], higher better.
    All three terms are RAW scores clipped to [0,1] -- no denominators anywhere.

        inclusion  z_slow's slow-factor score   -- is the slow factor IN z_slow?
        allocation z_fast's fast-factor score   -- is the fast factor recoverable
                   from the complement? (`z_mix` in the paper -- see BLOCKS above)
        exclusion  1 - z_slow's fast-factor score -- is z_slow free of it?

    Why no denominator on exclusion (revised 2026-08-07). Exclusion used to divide
    by the model's own z_full, to stop a model that encoded the fast factor NOWHERE
    from collecting a perfect exclusion for free. That job actually belongs to
    ALLOCATION, which does it entirely on its own: the vacuous nce+online cell on
    HAPT carries the fast proxy at 0.164 in its complement against 0.625 for
    nce+ema, and the product collapses on that term alone (SEP 0.129 vs 0.525).
    Keeping the denominator as well made two of the three terms move with the same
    property -- how much fast information the model represents at all -- so a model
    was rewarded twice for it. Measured across the six rotation tables, dropping it
    changes no winner and no top-3, and shifts values by 0.001-0.037.

    Two things fall out of having no denominator:
      * sep_index no longer needs z_full, so "which model's z_full" stops being a
        question (it was answered wrong once, from a different training run)
      * the index applies across architectures -- TS2Vec's 320 dims and PatchTST's
        C*128 need no shared reference encoder

    Consequence to state in the paper: on datasets where 16 dims naturally carry
    little of the fast factor, exclusion is high for everything (HAPT's random
    16-dim readout already scores 0.042-0.109 on the fast proxy). The random-subspace
    ROW is what makes that readable; the index alone cannot.

    SEP is a SUMMARY of the block x factor matrix, not a substitute for it. Report
    the raw four numbers -- (z_slow slow, z_slow fast, z_mix slow, z_mix fast) --
    beside it, and never settle a comparison on the scalar alone.

    `fast_ceiling` / `fast_null` are accepted and ignored; kept so older callers
    do not break.
    """
    (Ss, Fs), (_, Ff) = bf["z_slow"], bf["z_fast"]
    c = lambda v: float(min(max(v, 0.0), 1.0))
    incl, alloc, excl = c(Ss), c(Ff), c(1.0 - Fs)
    return dict(inclusion=incl, allocation=alloc, exclusion=excl,
                sep=incl * alloc * excl)


def context_curve(Z, lab, fast, tr, te, block="z_slow", positions=None,
                  label_range=(1, 6), max_samples=20000, seed=0):
    """Performance vs context length (appendix figure + justification for C_MIN).
    Fits one probe per position band so the curve is not confounded by pooling."""
    Lw = Z.shape[1]
    positions = positions if positions is not None else list(range(8, Lw, 16))
    curve = []
    for p in positions:
        r = multi_position(Z[:, p:p + 1], lab[:, p:p + 1], fast[:, p:p + 1],
                           tr, te, block=block, c_min=0, label_range=label_range,
                           max_samples=max_samples, seed=seed)
        curve.append(dict(position=int(p), context=int(p + 1),
                          f1=r.get("slow_kept_f1"), leak=r["leak_r2"],
                          n_test=r["n_test"]))
    return curve
