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
    return out


def sep_index(bf, fast_ceiling):
    """One scalar summarising the block x factor matrix. In [0, 1], higher better.

        inclusion  z_slow keeps the slow factor, relative to the whole embedding
        allocation z_fast actually HOLDS the fast factor -- normalised by the
                   ceiling ACROSS the comparison set, because a model that
                   encodes the fast factor nowhere has a low ceiling of its own
                   and would otherwise score a perfect ratio against itself
        exclusion  z_slow holds LESS fast information than a random block of the
                   same width; 0 if it holds as much (i.e. the gate did nothing)
    """
    (Ss, Fs), (_, Ff), (Sfull, _) = bf["z_slow"], bf["z_fast"], bf["z_full"]
    Fr = bf[f"rand{D_SLOW}"][1]
    incl = Ss / max(Sfull, 1e-6)
    alloc = Ff / max(fast_ceiling, 1e-6)
    excl = 1.0 - Fs / Fr if Fr > 1e-3 else float(Fs <= 1e-3)
    c = lambda v: float(min(max(v, 0.0), 1.0))
    return dict(inclusion=c(incl), allocation=c(alloc), exclusion=c(excl),
                sep=c(incl) * c(alloc) * c(excl))


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
