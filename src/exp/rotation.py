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

# Synthetic difficulty; 0.05 default keeps existing files byte-identical.
GAP = float(os.environ.get("HGLP_GAP", 0.05))
import json as _json
LAM = _json.loads(os.environ.get("HGLP_LAM", "{}"))  # per loss kind; see twosided.py
from sklearn.decomposition import PCA, FastICA

SEEDS = list(range(int(os.environ.get("HGLP_NSEED", 5))))
# The gate row is measured across the whole xcov-weight range, not at one default.
# The post-hoc unmixings have no comparable knob (their subspace width is matched to
# d_slow), so pinning our lambda to a single value either flatters or handicaps us
# depending on the setting -- at +-3% synth L1, lam=4 gives inclusion 0.497 and
# lam=1 gives 0.961. Reporting the curve makes the asymmetry visible instead of
# hiding it in a hyperparameter choice.
LAMS = [float(x) for x in os.environ.get("HGLP_LAMS", "0,1,4,16,64").split(",")]
ICA_ITER = int(os.environ.get("HGLP_ICA_ITER", 200))
ICA_ROWS = int(os.environ.get("HGLP_ICA_ROWS", 50000))
PROBE_ROWS = int(os.environ.get("HGLP_PROBE_ROWS", 200000))


def sfa_basis(X, pairs):
    """Slow Feature Analysis (Wiskott & Sejnowski 2002): whiten, then order the
    eigenvectors of the DERIVATIVE covariance by ascending eigenvalue (slowest
    first). Returns the FULL basis so the complement is available too.

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
    return Wht @ DV


def slowness_order(S, pairs):
    """Order components by temporal slowness over time-adjacent pairs."""
    dS = S[pairs[:, 1]] - S[pairs[:, 0]]
    return np.argsort((dS ** 2).mean(0) / (S.var(0) + 1e-12))


def unmixings(Xfit, pairs, d):
    """Fit each label-free unmixing on the TIME-ORDERED training embeddings
    `Xfit` (with `pairs` giving time-adjacent rows) and return
    {method: (W_slow, W_fast)} -- a FULL-RANK basis split into the method's own
    d "slowest" directions and the remaining D-d.

    The complement matters: a method is only a real alternative to the gate if
    its slow subspace excludes the fast factor AND its complement RETAINS it.
    Excluding by discarding is not separating. Means are dropped because the
    probe standardises features anyway."""
    D = Xfit.shape[1]
    out = {}
    C = PCA(n_components=D).fit(Xfit).components_.T          # cols by variance
    out["PCA"] = (C[:, :d], C[:, d:])
    try:
        # FastICA dominated the wall clock: 64 components over ~288k rows with
        # max_iter=1000 ran for tens of minutes per seed and did not converge
        # anyway. n_components is deliberately NOT reduced -- that would change the
        # basis and break comparability with the patch-50 runs. Instead the fit
        # sees a subsample of the rows and gives up sooner; both only affect how
        # precisely the same basis is estimated.
        Xi = Xfit if len(Xfit) <= ICA_ROWS else Xfit[
            np.random.default_rng(0).choice(len(Xfit), ICA_ROWS, replace=False)]
        ica = FastICA(n_components=D, random_state=0, max_iter=ICA_ITER,
                      tol=1e-3).fit(Xi)
        M = ica.components_.T[:, slowness_order(ica.transform(Xfit), pairs)]
        out["ICA-slow"] = (M[:, :d], M[:, d:])
    except Exception as e:                              # FastICA can fail to converge
        print(f"    ICA skipped: {type(e).__name__}", flush=True)
    S = sfa_basis(Xfit, pairs)
    out["SFA"] = (S[:, :d], S[:, d:])
    return out


def add(res, key, val):
    res.setdefault(key, []).append(val)


def full_score(a, ya, za, b, yb, zb, cls=True):
    """The WHOLE embedding on both factors. This is the exclusion denominator: how
    much fast information the model that produced this subspace actually holds.
    Stored per row-source so a model that encoded the fast factor nowhere cannot
    collect a free perfect exclusion (see probes.sep_index)."""
    from probes import score
    return (score(a, ya, za, b, yb, zb, cls), None)


def two_sided(a, ya, za, b, yb, zb, W, cls):
    """Score BOTH halves of a split on BOTH factors -> ((s,f) slow-half, (s,f) fast-half)."""
    from probes import score
    Ws, Wf = W
    return (score(a @ Ws, ya, za, b @ Ws, yb, zb, cls),
            score(a @ Wf, ya, za, b @ Wf, yb, zb, cls))


def _split_mats(D, d, seed):
    """Coordinate split (first d vs rest) and a random-rotation split of the same widths."""
    I = np.eye(D)
    from probes import rand_subspace
    Q = rand_subspace(D, D, seed)
    return (I[:, :d], I[:, d:]), (Q[:, :d], Q[:, d:])


def run_synth(lk, te):
    from train import train
    from datagen import make_dataset
    from model import P, L, D_SLOW, D_Z
    from train import DEV
    import torch
    # HGLP_GATE_SYM=1 mutes z_slow below tau instead of leaving it always on. The
    # one-sided default is why the Exclusion clause cannot be argued from the
    # objective, so this is the direct structural alternative to the xcov penalty.
    sym = os.environ.get("HGLP_GATE_SYM", "0") == "1"
    if sym:
        print("  gate_sym=ON  (z_slow weighted by 1-g)", flush=True)
    res = {}
    for s in SEEDS:
        gated = {lam: train(loss_kind=lk, target_enc=te, seed=s, gap=GAP, lam=lam,
                            gate_sym=sym, log_every=10 ** 9) for lam in LAMS}
        # no gate, no xcov AND no per-block LayerNorm: per-block LN normalises the two
        # blocks separately, which privileges the coordinate split on its own -- an
        # ungated control that keeps it is not mechanism-free
        ungated = train(loss_kind=lk, target_enc=te, seed=s, gate=False, xcov=False,
                        blocknorm=False, gap=GAP, log_every=10 ** 9)
        d = make_dataset(300_000, seed=99, gap=GAP)   # must match the training gap
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

        def feats(enc):
            with torch.no_grad():
                Z = torch.cat([enc(xb[i:i + 128]) for i in range(0, len(xb), 128)]).cpu().numpy()
            return np.concatenate([Z[:, a] for a in pos])

        coord, randsplit = _split_mats(D_Z, D_SLOW, s)
        arg = lambda F: (F[tr], ys[tr], yu[tr], F[teI], ys[teI], yu[teI])
        for lam in LAMS:                    # NOT `L` -- that is the window length
            Fg = feats(gated[lam]["enc"])
            add(res, f"gate@lam{lam:g}", two_sided(*arg(Fg), coord, True))
            add(res, f"gate@lam{lam:g}_full", full_score(*arg(Fg)))
        Fu = feats(ungated["enc"])                       # ungated: arbitrary block + rotations
        add(res, "ungated(block)", two_sided(*arg(Fu), coord, True))
        add(res, "random", two_sided(*arg(Fu), randsplit, True))
        add(res, "ungated_full", full_score(*arg(Fu)))
        # 시간 인접 쌍: 같은 윈도우의 이웃한 probe 위치. F 는 위치-major 배치
        # (row = pos_idx * n_win + win_idx) 이므로 인접 위치는 n_win 만큼 떨어져 있다.
        nw, npos = len(starts), len(pos)
        rank = -np.ones(len(Fu), int); rank[tr] = np.arange(len(tr))
        pm = np.arange(npos)[:, None] * nw + np.arange(nw)[None, :]          # (npos, nw)
        pr = np.stack([rank[pm[p]] for p in range(npos)])                    # (npos, nw)
        ok = (pr[:-1] >= 0) & (pr[1:] >= 0)             # 두 시점 모두 학습 split 인 쌍만
        pairs = np.stack([pr[:-1][ok], pr[1:][ok]], 1)
        for m, W in unmixings(Fu[tr], pairs, D_SLOW).items():
            add(res, m, two_sided(*arg(Fu), W, True))
        print(f"  seed {s} done", flush=True)
    return res



_ZMEMO = {}


def _encode_once(enc, Wt):
    """Encode a model's windows once and reuse it.

    run_real asks for the same encoder's embeddings up to three times in a row
    (train split, test split, and the unmixing fit stream), which on Sleep-EDF at
    patch 10 means re-running 104k windows and moving ~7 GB back from the GPU each
    time -- 13 full encodes per seed where 6 would do. Every call for a given
    encoder is consecutive, so a single-entry memo keyed on the object identity is
    enough; it is cleared on each new encoder so only one array is ever resident.
    Callers only ever slice the result, never mutate it."""
    from probes import encode_all
    k = id(enc)
    if k not in _ZMEMO:
        _ZMEMO.clear()
        _ZMEMO[k] = encode_all(enc, Wt)
    return _ZMEMO[k]


def run_real(npz, n_ax, kw, lk, te, static=False):
    """Real data: same comparison, using the shared multi-position probe pipeline
    so the gated and post-hoc numbers are produced by identical scoring code."""
    from train_real import train_real
    from probes import encode_all, C_MIN
    from model import D_SLOW, D_Z
    # See run_synth. The ungated control has gate=False, which makes gate_sym inert,
    # so it keeps its existing checkpoint and is shared with the asymmetric runs.
    sym = os.environ.get("HGLP_GATE_SYM", "0") == "1"
    if sym:
        print("  gate_sym=ON  (z_slow weighted by 1-g)", flush=True)
    res = {}
    for s in SEEDS:
        gated = {lam: train_real(npz, n_ax=n_ax, seed=s, loss_kind=lk, target_enc=te,
                                 lam=lam, gate_sym=sym, log_every=10 ** 9, **kw)
                 for lam in LAMS}
        ungated = train_real(npz, n_ax=n_ax, seed=s, loss_kind=lk, target_enc=te,
                             gate=False, xcov=False, blocknorm=False,
                             log_every=10 ** 9, **kw)
        g0 = gated[LAMS[0]]
        lab, fast, tr, teI = g0["lab"], g0["fast"], g0["tr"], g0["te"]

        def probe_set(enc, idx):
            """Features actually SCORED: labeled positions (HAPT) or the last
            position of each record (PTB-XL, static label)."""
            Z = _encode_once(enc, g0["Wt"])
            if static:
                return Z[idx, -1], lab[idx, -1], fast[idx, -1]
            ok = (lab >= 1) & (lab <= 6); ok[:, :C_MIN] = False
            m = ok[idx]
            f, y, z = Z[idx][m], lab[idx][m] - 1, fast[idx][m]
            # Cap the rows. probes.multi_position has always subsampled to 60k, but
            # score() -- which this feeds -- did not, so on Sleep-EDF at patch 10
            # every linear probe was fit on 13.3M rows and one seed took hours. A
            # 16-dim linear probe is saturated far below this cap; the subsample
            # moves the estimate by noise, not by substance.
            if len(f) > PROBE_ROWS:
                sel = np.random.default_rng(s).choice(len(f), PROBE_ROWS,
                                                      replace=False)
                f, y, z = f[sel], y[sel], z[sel]
            return f, y, z

        def fit_stream(enc, idx, max_win=1500):
            """Time-ordered embeddings the unmixings are FIT on: every position
            from C_MIN onward, window-major so positions stay contiguous in time.
            Returns (X, pairs) with pairs = time-adjacent rows within a window."""
            Z = _encode_once(enc, g0["Wt"])
            w = idx if len(idx) <= max_win else np.random.default_rng(0).choice(
                idx, max_win, replace=False)
            lo = 0 if static else C_MIN
            Zs = Z[w][:, lo:]                                   # (nw, T, D)
            nw, T, D = Zs.shape
            base = np.arange(nw)[:, None] * T
            pairs = np.stack([(base + np.arange(T - 1)).ravel(),
                              (base + np.arange(1, T)).ravel()], 1)
            return Zs.reshape(-1, D), pairs

        cls = True                          # both datasets are classification here
        coord, randsplit = _split_mats(D_Z, D_SLOW, s)
        for lam in LAMS:                    # NOT `L` -- shadows the window length
            a, ya, za = probe_set(gated[lam]["enc"], tr)
            b, yb, zb = probe_set(gated[lam]["enc"], teI)
            add(res, f"gate@lam{lam:g}", two_sided(a, ya, za, b, yb, zb, coord, cls))
            add(res, f"gate@lam{lam:g}_full", full_score(a, ya, za, b, yb, zb, cls))

        Ftr, ytr, ztr = probe_set(ungated["enc"], tr)
        Fte, yte, zte = probe_set(ungated["enc"], teI)
        arg = (Ftr, ytr, ztr, Fte, yte, zte)
        add(res, "ungated(block)", two_sided(*arg, coord, cls))
        add(res, "random", two_sided(*arg, randsplit, cls))
        add(res, "ungated_full", full_score(*arg, cls))
        Xfit, pairs = fit_stream(ungated["enc"], tr)        # unmixings see the SERIES
        for m, W in unmixings(Xfit, pairs, D_SLOW).items():
            add(res, m, two_sided(*arg, W, cls))
        print(f"  seed {s} done", flush=True)
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "synth"
    stem = sys.argv[2] if len(sys.argv) > 2 else "nce+ema"
    lk, te = stem.split("+")
    from model import D_SLOW as D_SLOW_REPORT
    os.makedirs("../../runs_v2", exist_ok=True)
    print(f"=== post-hoc rotation baseline: {which}, stem {stem}, {len(SEEDS)} seeds ===")
    if which == "synth":
        res = run_synth(lk, te)
    elif which == "ptbxl":
        res = run_real("../../data/ptbxl_v2.npz", 1,
                       dict(tau=16.0, w=8, dmin=8, dmax=48, min_context=8), lk, te, static=True)
    elif which == "hapt":
        res = run_real("../../data/hapt_v2.npz", 3,
                       dict(tau=40.0, w=12, dmin=12, dmax=128, min_context=16), lk, te)
    elif which == "sleepedf":
        # Config is env-overridable so the patch-10 re-run can sweep tau without
        # editing code. Defaults reproduce the original patch-50 setting.
        cfg = dict(npz=os.environ.get("HGLP_NPZ", "../../data/sleepedf_v2.npz"),
                   tau=float(os.environ.get("HGLP_TAU", 24.0)),
                   w=int(os.environ.get("HGLP_W", 12)),
                   dmin=int(os.environ.get("HGLP_DMIN", 12)),
                   dmax=int(os.environ.get("HGLP_DMAX", 128)),
                   steps=int(os.environ.get("HGLP_STEPS", 5000)))
        print(f"  config {cfg}", flush=True)
        res = run_real(cfg["npz"], 3,
                       dict(tau=cfg["tau"], w=cfg["w"], dmin=cfg["dmin"],
                            dmax=cfg["dmax"], min_context=16, steps=cfg["steps"]),
                       lk, te)
    else:
        raise SystemExit(f"unknown dataset {which!r}; "
                         "expected synth | ptbxl | hapt | sleepedf")
    def stat(v, half, factor):
        """[mean, sd, per-seed...]. The per-seed list is what makes a PAIRED test
        possible -- same seed means same data and same init, so pairing removes the
        across-seed variance that swamped the 3-seed means."""
        a = [c[half][factor] for c in v]
        return [float(np.mean(a)), float(np.std(a, ddof=1) if len(a) > 1 else 0.0)] + \
               [float(x) for x in a]

    agg = {}
    for m, v in res.items():
        if not v:
            continue
        halves = ("slow_half",) if v[0][1] is None else ("slow_half", "fast_half")
        agg[m] = {h: dict(slow=stat(v, i, 0), fast=stat(v, i, 1))
                  for i, h in enumerate(halves)}
    agg["_config"] = dict(dataset=which, stem=stem, seeds=SEEDS, lams=LAMS, gap=GAP,
                          ungated_blocknorm=False, d_slow=int(D_SLOW_REPORT))
    json.dump(agg, open(f"../../runs_v2/rotation_{which}_{stem}"
                   f"{'' if GAP == 0.05 or which != 'synth' else f'_gap{GAP:g}'}"
                   f"{os.environ.get('HGLP_LAM_TAG','')}{os.environ.get('HGLP_TAG','')}.json", "w"), indent=2)

    print("\n=== can a label-free post-hoc unmixing of the UNGATED embedding match the gate? ===")
    print("    a real alternative must do BOTH: keep slow / drop fast in its slow half,")
    print("    AND keep fast in its complement. `random` is the null both are judged against.\n")
    print(f"{'method':16s} | {'slow half (16d)':>21} | {'fast half (48d)':>21}")
    print(f"{'':16s} | {'slow F1':>10}{'fast R2':>11} | {'slow F1':>10}{'fast R2':>11}")
    for m, r in agg.items():
        if m == "_config":
            continue
        a = r["slow_half"]; b = r.get("fast_half")
        rhs = (f"{b['slow'][0]:10.3f}{b['fast'][0]:+11.3f}" if b else f"{'(전체 임베딩)':>21}")
        print(f"{m:16s} | {a['slow'][0]:10.3f}{a['fast'][0]:+11.3f} | {rhs}")
    null = agg["random"]["slow_half"]["fast"][0]
    print(f"\n  random-16 null on the fast factor: {null:+.3f}  "
          f"(a slow half above this excluded NOTHING)")
    for m, r in agg.items():
        if m in ("random", "_config") or "fast_half" not in r:
            continue
        print(f"  {m:16s} exclusion vs null {null - r['slow_half']['fast'][0]:+.3f}   "
              f"fast retained in complement {r['fast_half']['fast'][0]:.3f}")
