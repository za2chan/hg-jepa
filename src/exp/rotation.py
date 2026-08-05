"""Post-hoc rotation baseline — the sharpest objection to our contribution.

The objection: latent-prediction losses are ~invariant to rotations of the
embedding, so an UNGATED model has no reason to align factors to coordinate
blocks. If a LABEL-FREE linear unmixing (PCA / ICA / SFA) of the ungated
embedding recovers a low-dim subspace that keeps the slow factor and excludes
the fast one as well as our gate does, then our contribution reduces to a
coordinate choice obtainable by post-processing.

Design: train the g0_x0 cell (no gate, no xcov) — the same ungated control the
2x2 ablation uses — then fit each unmixing on the TRAIN split only, take a
d_slow-dim subspace by that method's own label-free ordering, and probe it with
the identical protocol used for z_slow. Compare against the gated model's
z_slow (g1_x1) from the same seeds.

Label-free subspace selection per method:
  PCA   top-d by variance          (no notion of slowness — the naive control)
  ICA   d components, ranked by temporal slowness of the source
  SFA   slowest-d directions       (Wiskott & Sejnowski 2002 — strongest: it is
                                    explicitly built to order by slowness)
Usage: python3 rotation.py [synth|hapt] [stem]
Writes runs_v2/rotation_<dataset>_<stem>.json
"""
import json
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hglp"))

import numpy as np
from sklearn.decomposition import PCA, FastICA

SEEDS = [0, 1, 2]


def sfa(X, d, pairs):
    """Slow Feature Analysis (Wiskott & Sejnowski 2002): whiten, then take the
    eigenvectors of the DERIVATIVE covariance with the smallest eigenvalues.

    `pairs` is (M, 2) of row indices that are ADJACENT IN TIME. Derivatives are
    taken only over those pairs. Differencing raw adjacent rows of the flattened
    array (as an earlier version did) is not a time derivative at all and
    silently degenerates SFA into low-variance PCA."""
    Xc = X - X.mean(0)
    C = np.cov(Xc, rowvar=False) + 1e-6 * np.eye(Xc.shape[1])
    ev, EV = np.linalg.eigh(C)
    Wht = EV / np.sqrt(np.maximum(ev, 1e-12))          # whitening
    Y = Xc @ Wht
    dY = Y[pairs[:, 1]] - Y[pairs[:, 0]]               # true temporal derivative
    Cd = np.cov(dY, rowvar=False)
    dv, DV = np.linalg.eigh(Cd)                        # ascending = slowest first
    return Wht @ DV[:, :d]


def slowness_rank(S, d, pairs):
    """Rank components by temporal slowness over time-adjacent pairs."""
    dS = S[pairs[:, 1]] - S[pairs[:, 0]]
    score = (dS ** 2).mean(0) / (S.var(0) + 1e-12)
    return np.argsort(score)[:d]


def subspaces(Xfit, pairs, apply_to, d):
    """Fit each label-free unmixing on the TIME-ORDERED training embeddings
    `Xfit` (with `pairs` giving time-adjacent rows), then project every array in
    `apply_to`. Fitting on the full temporal stream and scoring at the probe
    positions keeps the temporal methods on the data they were designed for."""
    out = {}
    p = PCA(n_components=d).fit(Xfit)
    out["PCA"] = [p.transform(A) for A in apply_to]
    try:
        ica = FastICA(n_components=d, random_state=0, max_iter=500).fit(Xfit)
        keep = slowness_rank(ica.transform(Xfit), d, pairs)     # label-free, temporal
        out["ICA-slow"] = [ica.transform(A)[:, keep] for A in apply_to]
    except Exception as e:                              # FastICA can fail to converge
        print(f"    ICA skipped: {type(e).__name__}", flush=True)
    Wsfa = sfa(Xfit, d, pairs)
    mu = Xfit.mean(0)
    out["SFA"] = [(A - mu) @ Wsfa for A in apply_to]
    return out


def score(ftr, ytr, ztr, fte, yte, zte, classification):
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if classification:
        lbl = np.unique(ytr)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, class_weight="balanced"))
        kept = float(f1_score(yte, clf.fit(ftr, ytr).predict(fte), average="macro",
                              labels=lbl, zero_division=0))
    else:
        kept = float(make_pipeline(StandardScaler(), Ridge()).fit(ftr, ytr).score(fte, yte))
    leak = float(make_pipeline(StandardScaler(), Ridge()).fit(ftr, ztr).score(fte, zte))
    return kept, leak


def run_synth(lk, te):
    from train import train
    from datagen import make_dataset
    from model import P, L, D_SLOW, D_Z
    from train import DEV
    import torch
    res = {m: [] for m in ("gate(z_slow)", "PCA", "ICA-slow", "SFA", "ungated(z_slow)")}
    for s in SEEDS:
        gated = train(loss_kind=lk, target_enc=te, seed=s, log_every=10 ** 9)
        ungated = train(loss_kind=lk, target_enc=te, seed=s, gate=False, xcov=False,
                        log_every=10 ** 9)
        d = make_dataset(300_000, seed=99)
        x, sreg, u = d["x"], d["s"], d["u"]
        rng = np.random.default_rng(99)
        starts = rng.integers(0, len(x) - L * P - 1, 400)
        xb = torch.from_numpy(np.stack([x[st:st + L * P].reshape(L, P)
                                        for st in starts])).to(DEV)
        pos = list(range(64, 240, 12))
        cut = np.sort(starts)[len(starts) // 2]
        wid = np.tile(np.arange(len(starts)), len(pos))
        tr = np.flatnonzero(starts[wid] + L * P <= cut)
        teI = np.flatnonzero(starts[wid] > cut)
        ys = np.concatenate([sreg[starts + a * P + (P - 1)] for a in pos])
        yu = np.concatenate([u[starts + a * P + (P - 1)] for a in pos])

        def feats(enc, sl):
            with torch.no_grad():
                Z = torch.cat([enc(xb[i:i + 128]) for i in range(0, len(xb), 128)]).cpu().numpy()
            return np.concatenate([Z[:, a, sl] for a in pos])
        # our method
        F_ = feats(gated["enc"], slice(0, D_SLOW))
        res["gate(z_slow)"].append(score(F_[tr], ys[tr], yu[tr], F_[teI], ys[teI], yu[teI], True))
        # ungated: raw first-16 (the "arbitrary block" control) + post-hoc rotations
        Fu_full = feats(ungated["enc"], slice(0, D_Z))
        res["ungated(z_slow)"].append(
            score(Fu_full[tr][:, :D_SLOW], ys[tr], yu[tr], Fu_full[teI][:, :D_SLOW],
                  ys[teI], yu[teI], True))
        # 시간 인접 쌍: 같은 윈도우의 이웃한 probe 위치. F_ 는 위치-major 배치
        # (row = pos_idx * n_win + win_idx) 이므로 인접 위치는 n_win 만큼 떨어져 있다.
        nw, npos = len(starts), len(pos)
        rank = -np.ones(len(Fu_full), int); rank[tr] = np.arange(len(tr))
        pm = np.arange(npos)[:, None] * nw + np.arange(nw)[None, :]          # (npos, nw)
        pr = np.stack([rank[pm[p]] for p in range(npos)])                    # (npos, nw)
        ok = (pr[:-1] >= 0) & (pr[1:] >= 0)             # 두 시점 모두 학습 split 인 쌍만
        pairs = np.stack([pr[:-1][ok], pr[1:][ok]], 1)
        Xfit = Fu_full[tr]
        for m, (a, b) in ((k, v) for k, v in
                          subspaces(Xfit, pairs, [Fu_full[tr], Fu_full[teI]], D_SLOW).items()):
            res[m].append(score(a, ys[tr], yu[tr], b, ys[teI], yu[teI], True))
        print(f"  seed {s} done", flush=True)
    return res


def run_real(npz, n_ax, kw, lk, te, static=False):
    """Real data: same comparison, using the shared multi-position probe pipeline
    so the gated and post-hoc numbers are produced by identical scoring code."""
    from train_real import train_real
    from probes import encode_all, C_MIN, BLOCKS
    from model import D_SLOW, D_Z
    res = {m: [] for m in ("gate(z_slow)", "PCA", "ICA-slow", "SFA", "ungated(z_slow)")}
    for s in SEEDS:
        gated = train_real(npz, n_ax=n_ax, seed=s, loss_kind=lk, target_enc=te,
                           log_every=10 ** 9, **kw)
        ungated = train_real(npz, n_ax=n_ax, seed=s, loss_kind=lk, target_enc=te,
                             gate=False, xcov=False, log_every=10 ** 9, **kw)
        lab, fast, tr, teI = gated["lab"], gated["fast"], gated["tr"], gated["te"]

        def probe_set(enc, sl, idx):
            """Features actually SCORED: labeled positions (HAPT) or the last
            position of each record (PTB-XL, static label)."""
            Z = encode_all(enc, gated["Wt"])[:, :, sl]
            if static:
                return Z[idx, -1], lab[idx, -1], fast[idx, -1]
            ok = (lab >= 1) & (lab <= 6); ok[:, :C_MIN] = False
            m = ok[idx]
            return Z[idx][m], lab[idx][m] - 1, fast[idx][m]

        def fit_stream(enc, idx, max_win=1500):
            """Time-ordered embeddings the unmixings are FIT on: every position
            from C_MIN onward, window-major so positions stay contiguous in time.
            Returns (X, pairs) with pairs = time-adjacent rows within a window."""
            Z = encode_all(enc, gated["Wt"])
            w = idx if len(idx) <= max_win else np.random.default_rng(0).choice(
                idx, max_win, replace=False)
            lo = 0 if static else C_MIN
            Zs = Z[w][:, lo:]                                   # (nw, T, D)
            nw, T, D = Zs.shape
            base = np.arange(nw)[:, None] * T
            pairs = np.stack([(base + np.arange(T - 1)).ravel(),
                              (base + np.arange(1, T)).ravel()], 1)
            return Zs.reshape(-1, D), pairs

        cls = not static or True            # both are classification here
        a, ya, za = probe_set(gated["enc"], slice(0, D_SLOW), tr)
        b, yb, zb = probe_set(gated["enc"], slice(0, D_SLOW), teI)
        res["gate(z_slow)"].append(score(a, ya, za, b, yb, zb, cls))

        Ftr, ytr, ztr = probe_set(ungated["enc"], slice(0, D_Z), tr)
        Fte, yte, zte = probe_set(ungated["enc"], slice(0, D_Z), teI)
        Xfit, pairs = fit_stream(ungated["enc"], tr)        # unmixings see the SERIES
        res["ungated(z_slow)"].append(
            score(Ftr[:, :D_SLOW], ytr, ztr, Fte[:, :D_SLOW], yte, zte, cls))
        for m, (p, q) in subspaces(Xfit, pairs, [Ftr, Fte], D_SLOW).items():
            res[m].append(score(p, ytr, ztr, q, yte, zte, cls))
        print(f"  seed {s} done", flush=True)
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "synth"
    stem = sys.argv[2] if len(sys.argv) > 2 else "nce+ema"
    lk, te = stem.split("+")
    os.makedirs("../../runs_v2", exist_ok=True)
    print(f"=== post-hoc rotation baseline: {which}, stem {stem}, {len(SEEDS)} seeds ===")
    if which == "synth":
        res = run_synth(lk, te)
    elif which == "ptbxl":
        res = run_real("../../data/ptbxl_v2.npz", 1,
                       dict(tau=16.0, w=8, dmin=8, dmax=48, min_context=8), lk, te, static=True)
    else:
        res = run_real("../../data/hapt_v2.npz", 3,
                       dict(tau=40.0, w=12, dmin=12, dmax=128, min_context=16), lk, te)
    agg = {m: dict(slow_kept=(float(np.mean([c[0] for c in v])), float(np.std([c[0] for c in v]))),
                   leak=(float(np.mean([c[1] for c in v])), float(np.std([c[1] for c in v]))))
           for m, v in res.items() if v}
    json.dump(agg, open(f"../../runs_v2/rotation_{which}_{stem}.json", "w"), indent=2)
    print(f"\n=== can label-free post-hoc unmixing of the UNGATED embedding match the gate? ===")
    print(f"{'method':18s} {'slow-kept':>16} {'leak':>16}")
    for m, r in agg.items():
        a, b = r["slow_kept"], r["leak"]
        print(f"{m:18s} {a[0]:8.3f}±{a[1]:.3f} {b[0]:+9.3f}±{b[1]:.3f}")
    g = agg["gate(z_slow)"]
    best = min((m for m in agg if m not in ("gate(z_slow)",)),
               key=lambda m: agg[m]["leak"][0] - agg[m]["slow_kept"][0])
    print(f"\n  gate: kept {g['slow_kept'][0]:.3f} leak {g['leak'][0]:+.3f}")
    print(f"  best post-hoc ({best}): kept {agg[best]['slow_kept'][0]:.3f} "
          f"leak {agg[best]['leak'][0]:+.3f}")
