"""Does the separation survive a probe that is not linear -- for US AND for SFA?

Extends mlp_probe_check.py, which asked the question only of our own blocks. That
was not a fair test: the paper's real-data losses are entirely in the exclusion
term (term_breakdown.py: PTB-XL -0.136/-0.051, HAPT/Reg -0.007, with inclusion
tied everywhere), and SFA's exclusion sits at 0.89-0.99. If SFA's exclusion is
also a linear-readability artefact, the comparison the paper reports is measured
with a ruler that flatters both sides equally and the ranking may not survive.

So every post-hoc unmixing is re-scored with the same MLP head on the same
features, splits and subsample. The unmixings are fit exactly as in rotation.py --
on the TRAIN split only, from a time-ordered stream with adjacent-position pairs --
so nothing about them changes except the probe that reads them.


Every number in the paper comes from a linear readout: logistic regression for the
persistent factor, ridge for the transient proxy. CLAUDE.md 1 already concedes the
consequence -- exclusion holds "at the linear level only" -- but the paper has
never measured what a nonlinear probe would say. Two things ride on it.

  1. Our own claim. Exclusion near 0.99 may mean the transient factor is absent
     from z_slow, or merely that it is there in a form a linear map cannot reach.
     If an MLP recovers it, the claim shrinks to a statement about linear
     readability.

  2. The ceiling. Widening the trunk dropped the transient proxy's linear score
     everywhere in the embedding (z_full R2 0.763 -> 0.206 on HAPT/NCE). That
     could be lost information or a change of representation. Only a nonlinear
     probe tells them apart.

Design. The features, the splits, the subsampling and the block definitions are
identical to probes.block_factor; the ONLY change is the head. Linear and MLP are
computed on the same arrays in the same call, so nothing else can move. Both
report the same four rows (z_slow, z_mix, z_full and a random split of the same
widths) and the same three SEP terms, plus the ceiling.

MLP settings are fixed across every condition and deliberately modest: one hidden
layer of 256, early stopping on a 10% validation split. A probe tuned per cell
would measure our tuning, not the representation.

python3 src/exp/mlp_probe_check.py -> runs_v2/mlp_probe_check.json
"""
import json, pathlib, sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "hglp"))
sys.path.insert(0, str(HERE.parent))
from model import D_SLOW, D_Z, P, L                                   # noqa: E402
from train import DEV                                                 # noqa: E402
from probes import C_MIN, rand_subspace                               # noqa: E402
from rotation import unmixings                                        # noqa: E402

ROOT = HERE.parents[2]
OUT = ROOT / "runs_v2"
MAXN, SEED = 40000, 0
MLP = dict(hidden_layer_sizes=(256,), max_iter=300, early_stopping=True,
           validation_fraction=.1, random_state=0)
DATASETS = {                                    # npz, n_ax, train kwargs, static label
    "synth": (None, None, dict(gap=0.03), False),
    "hapt": (ROOT / "data/hapt_v2.npz", 3,
             dict(tau=40.0, w=12, dmin=12, dmax=128, min_context=16), False),
    "ptbxl": (ROOT / "data/ptbxl_v2.npz", 1,
              dict(tau=16.0, w=8, dmin=8, dmax=48, min_context=8), True),
    # patch-10 config A: the only real-data setting whose tau comes from the rule
    "sleepedf": (ROOT / "data/sleepedf_p10.npz", 3,
                 dict(tau=9.0, w=8, dmin=8, dmax=64, min_context=16, steps=5000), False),
}


def head(kind, mlp):
    """The only thing that differs between the two columns of every table here."""
    if kind == "cls":
        m = MLPClassifier(**MLP) if mlp else LogisticRegression(
            max_iter=1000, class_weight="balanced")
    else:
        m = MLPRegressor(**MLP) if mlp else Ridge()
    return make_pipeline(StandardScaler(), m)


def score(ftr, ytr, ztr, fte, yte, zte, cls, mlp):
    """(persistent score, transient score) for one block, one probe family."""
    if cls:
        pred = head("cls", mlp).fit(ftr, ytr).predict(fte)
        kept = float(f1_score(yte, pred, average="macro", labels=np.unique(ytr),
                              zero_division=0))
    else:
        kept = float(head("reg", mlp).fit(ftr, ytr).score(fte, yte))
    fast = float(head("reg", mlp).fit(ftr, ztr).score(fte, zte))
    return kept, fast


def matrix_basis(Ftr, ytr, ztr, Fte, yte, zte, cls, mlp, W):
    """Score one post-hoc basis (W_slow, W_mix) with the given probe head."""
    rng = np.random.default_rng(SEED)
    pick = lambda F, y, z: (lambda s: (F[s], y[s], z[s]))(
        rng.choice(len(F), min(MAXN, len(F)), replace=False))
    Ftr, ytr, ztr = pick(Ftr, ytr, ztr)
    Fte, yte, zte = pick(Fte, yte, zte)
    Ws, Wf = W
    a = score(Ftr @ Ws, ytr, ztr, Fte @ Ws, yte, zte, cls, mlp)
    b = score(Ftr @ Wf, ytr, ztr, Fte @ Wf, yte, zte, cls, mlp)
    return {"z_slow": a, "z_mix": b, "z_full": (a[0], max(a[1], b[1])),
            "randsplit_slow": a, "randsplit_mix": b}


def matrix(Ftr, ytr, ztr, Fte, yte, zte, cls, mlp, n_rand=3):
    rng = np.random.default_rng(SEED)
    pick = lambda F, y, z: (lambda s: (F[s], y[s], z[s]))(
        rng.choice(len(F), min(MAXN, len(F)), replace=False))
    Ftr, ytr, ztr = pick(Ftr, ytr, ztr)
    Fte, yte, zte = pick(Fte, yte, zte)
    D = Ftr.shape[1]
    out = {k: score(Ftr[:, s], ytr, ztr, Fte[:, s], yte, zte, cls, mlp)
           for k, s in (("z_slow", slice(0, D_SLOW)), ("z_mix", slice(D_SLOW, D)),
                        ("z_full", slice(0, D)))}
    pairs = []
    for i in range(n_rand):                      # the null: an arbitrary split
        Q = rand_subspace(D, D, SEED + 7919 * i)
        pairs.append([score(Ftr @ Q[:, s], ytr, ztr, Fte @ Q[:, s], yte, zte, cls, mlp)
                      for s in (slice(0, D_SLOW), slice(D_SLOW, D))])
    out["randsplit_slow"] = tuple(float(np.mean([p[0][j] for p in pairs])) for j in (0, 1))
    out["randsplit_mix"] = tuple(float(np.mean([p[1][j] for p in pairs])) for j in (0, 1))
    return out


def sep_of(m):
    c = lambda v: float(min(max(v, 0.0), 1.0))
    i, a, e = c(m["z_slow"][0]), c(m["z_mix"][1]), c(1 - m["z_slow"][1])
    return dict(inclusion=i, allocation=a, exclusion=e, sep=i * a * e,
                ceiling=float(m["z_full"][1]),
                null_sep=c(m["randsplit_slow"][0]) * c(m["randsplit_mix"][1])
                * c(1 - m["randsplit_slow"][1]))


def feats_synth(enc, seed=99, n_win=400, pos=tuple(range(64, 240, 12)), gap=0.03):
    from datagen import make_dataset
    d = make_dataset(300_000, seed=seed, gap=gap)
    x, s, u = d["x"], d["s"], d["u"]
    rng = np.random.default_rng(seed)
    st = rng.integers(0, len(x) - L * P - 1, n_win)
    xb = torch.from_numpy(np.stack([x[a:a + L * P].reshape(L, P) for a in st])).to(DEV)
    with torch.no_grad():
        Z = torch.cat([enc(xb[i:i + 128]) for i in range(0, len(xb), 128)]).cpu().numpy()
    F_ = np.concatenate([Z[:, a] for a in pos])
    ys = np.concatenate([s[st + a * P + (P - 1)] for a in pos])
    yu = np.concatenate([u[st + a * P + (P - 1)] for a in pos])
    wid = np.tile(np.arange(n_win), len(pos))
    cut = np.sort(st)[n_win // 2]
    tr = np.flatnonzero(st[wid] + L * P <= cut); te = np.flatnonzero(st[wid] > cut)
    return (F_[tr], ys[tr], yu[tr]), (F_[te], ys[te], yu[te])


def feats_real(r, static):
    with torch.no_grad():
        Z = torch.cat([r["enc"](r["Wt"][i:i + 64])
                       for i in range(0, len(r["Wt"]), 64)]).cpu().numpy()
    lab, fast = r["lab"], r["fast"]
    if static:
        g = lambda i: (Z[i, -1], lab[i, -1], fast[i, -1])
    else:
        ok = (lab >= 1) & (lab <= 6); ok[:, :C_MIN] = False
        g = lambda i: (Z[i][ok[i]], lab[i][ok[i]] - 1, fast[i][ok[i]])
    return g(r["tr"]), g(r["te"])



def fit_stream_for(r, static, max_win=1500):
    """Time-ordered training embeddings + adjacent-position pairs, matching
    rotation.fit_stream so the unmixings see exactly what they see there."""
    with torch.no_grad():
        Z = torch.cat([r["enc"](r["Wt"][i:i + 64])
                       for i in range(0, len(r["Wt"]), 64)]).cpu().numpy()
    idx = r["tr"]
    w = idx if len(idx) <= max_win else np.random.default_rng(0).choice(
        idx, max_win, replace=False)
    lo = 0 if static else C_MIN
    Zs = Z[w][:, lo:]
    nw, T, D = Zs.shape
    base = np.arange(nw)[:, None] * T
    prs = np.stack([(base + np.arange(T - 1)).ravel(),
                    (base + np.arange(1, T)).ravel()], 1)
    return Zs.reshape(-1, D), prs


def main():
    res = {}
    for name, (npz, n_ax, kw, static) in DATASETS.items():
        for stem in ("nce", "l1"):
            key = f"{name}/{stem}"
            print(f"training {key} ...", flush=True)
            if name == "synth":
                from train import train
                r = train(loss_kind=stem, target_enc="ema", seed=SEED, gate=True,
                          xcov=True, lam=4.0, log_every=10 ** 9, **kw)
                a, b = feats_synth(r["enc"], gap=kw["gap"])
            else:
                from train_real import train_real
                r = train_real(str(npz), n_ax=n_ax, seed=SEED, loss_kind=stem,
                               target_enc="ema", gate=True, xcov=True, lam=4.0,
                               log_every=10 ** 9, **kw)
                a, b = feats_real(r, static)
            cls = True                                  # persistent factor is discrete
            cell = {}
            for probe in ("linear", "mlp"):
                m = matrix(*a, *b, cls, probe == "mlp")
                cell[probe] = dict(block_factor={k: dict(persistent=v[0], transient=v[1])
                                                 for k, v in m.items()},
                                   **sep_of(m))

            # Post-hoc unmixings, fit the way rotation.py fits them: on the TRAIN
            # split only, from a time-ordered stream whose adjacent rows are
            # adjacent positions. Then read with BOTH probe heads.
            if name == "synth":
                bases = {}          # feats_synth has no Wt/tr stream; skip unmixings
            else:
                Xfit, pairs = fit_stream_for(r, static)
                bases = unmixings(Xfit, pairs, D_SLOW)
            for mname, W in bases.items():
                for probe in ("linear", "mlp"):
                    mm = matrix_basis(*a, *b, cls, probe == "mlp", W)
                    cell.setdefault(f"{mname}/{probe}", {}).update(
                        dict(block_factor={k: dict(persistent=v[0], transient=v[1])
                                           for k, v in mm.items()},
                             **sep_of(mm)))
            res[key] = cell
            for probe in ("linear", "mlp"):
                s = cell[probe]
                print(f"  {probe:<7} downstream(z_slow) {s['inclusion']:.3f}   "
                      f"SEP {s['sep']:.3f}  [incl {s['inclusion']:.3f} "
                      f"alloc {s['allocation']:.3f} excl {s['exclusion']:.3f}]   "
                      f"ceiling {s['ceiling']:.3f}   null SEP {s['null_sep']:.3f}",
                      flush=True)

    print(f"\n{'조건':<14}{'method':<12}{'probe':<8}{'포함':>8}{'배정':>8}{'배제':>8}{'SEP':>8}")
    for k, cell in res.items():
        for name in sorted(cell):
            if "/" in name:
                meth, probe = name.split("/")
            else:
                meth, probe = "ours", name
            s_ = cell[name]
            print(f"{k:<14}{meth:<12}{probe:<8}{s_['inclusion']:>8.3f}"
                  f"{s_['allocation']:>8.3f}{s_['exclusion']:>8.3f}{s_['sep']:>8.3f}")

    (OUT / "mlp_probe_rotation.json").write_text(json.dumps(dict(
        config=dict(mlp=MLP, max_samples=MAXN, seed=SEED, lam=4.0, gate=True,
                    xcov=True, d_slow=D_SLOW, d_z=D_Z,
                    note="identical features, splits and blocks; only the probe "
                         "head differs. Downstream = z_slow's persistent score, "
                         "which is also SEP's inclusion term."),
        results=res), indent=1))
    print("\nwrote mlp_probe_rotation.json")


if __name__ == "__main__":
    main()
