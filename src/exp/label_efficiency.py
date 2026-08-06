"""Label efficiency (P3-5): does z_slow reach a given activity-F1 with FEWER
labelled windows than the full embedding?

Claim under test: at a fixed target macro-F1, the z_slow probe needs a smaller
label budget than a z_full probe of the SAME model and than the full embedding
of an ungated control. FALSIFIED if the z_slow curve does not sit above z_full
at small budgets — or if it sits above by no more than the dimension-matched
`rand16` arm does, in which case the gain is the statistical advantage of
fitting 16 instead of 64 coefficients on n points and has nothing to do with
slowness. That control is the reason `rand16` is in the default arm set.

Encoder-agnostic by construction: the entry point takes per-position embeddings,
not a training function, so our stems, the ungated control and the SSL baselines
(`src/baselines/common.py` contract: `embed` -> (B, L, D), plus lab/tr/te) are
scored by identical code. `fast` is deliberately NOT consumed — leak is the
block x factor matrix's job (`exp/twosided.py`); this file measures label cost.
The two corrections the baseline adapters ask for are caller-side and need no
change here: PCA-16 the baseline embedding before passing it (D = 320 / 1536 vs
our 64, and this metric is dominated by probe dimension — that is what `rand16`
measures), and pass `c_min=L-1` for a last-position-only comparison, where a
bidirectional encoder no longer sees more of the window than our causal one.

The budget unit is the WINDOW, never the position. HAPT windows are cut with
stride L/2 and carry ~132 labelled positions each past the C1 floor, so a
position-level draw would hand a "5-label" probe ~600 correlated samples, half
of them sharing raw samples with their neighbours. The budget applies to the
TRAIN side only; the eval set is fixed across budgets, draws and arms, and
macro-F1 uses the EVAL set's class list (C4) so a 5-window draw is scored on the
same 6-way problem as the full one.

Usage: python3 label_efficiency.py [steps] [n_seeds] [n_draw]
       python3 label_efficiency.py selfcheck
Writes runs_v2/label_efficiency_hapt.json (…_smoke.json for short runs).
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hglp"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import D_SLOW, D_Z
from probes import C_MIN, encode_all, rand_subspace

ROOT = pathlib.Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/hapt_v2.npz"
HAPT = dict(n_ax=3, tau=40.0, w=12, dmin=12, dmax=128, min_context=16)
SLEEPEDF = dict(n_ax=3, tau=24.0, w=12, dmin=12, dmax=128, min_context=16)
# steps=5000 for sleepedf: 2500 leaves the 48-dim z_fast undertrained (measured).
DSETS = {"hapt": (ROOT / "data/hapt_v2.npz", HAPT, 2500),
         "sleepedf": (ROOT / "data/sleepedf_v2.npz", SLEEPEDF, 5000)}
DATASET = "hapt"          # overridden by argv[1]

BUDGETS = (5, 10, 25, 50, 100, 250, 500, None)   # None = every train window
N_DRAW, STEPS, N_SEED = 5, 2500, 1
STEMS = ("l1+ema", "nce+ema")                    # both stems carry Part 2 (CLAUDE.md §2)
# same train cap as probes.multi_position, so the top of the curve is comparable
# with the headline numbers. It binds above ~450 windows on HAPT: budgets past
# that point differ only in WHICH 60k positions are seen, so the curve flattens
# there for a reason that is not saturation. n_pos is reported to make it visible.
MAX_TRAIN, MAX_EVAL = 60000, 20000
TARGETS = (0.8, 0.9, 0.95)                       # fractions of the ceiling arm's score


def _fit_score(ftr, ytr, fte, yte, labels):
    """macro-F1 over a FIXED label list (C4). A budget-dependent label set would
    score small budgets on an easier problem and make the curve meaningless."""
    if len(np.unique(ytr)) < 2:            # a 5-window draw can be single-class or empty
        pred = np.full(len(yte), ytr[0] if len(ytr) else labels[0])
    else:
        pred = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced")
        ).fit(ftr, ytr).predict(fte)
    return float(f1_score(yte, pred, average="macro", labels=labels, zero_division=0))


def label_efficiency(feats, lab, tr, te, budgets=BUDGETS, n_draw=N_DRAW, seed=0,
                     c_min=C_MIN, label_range=(1, 6), max_train=MAX_TRAIN,
                     max_eval=MAX_EVAL):
    """feats: {arm: (n, L, d)} per-position features over the SAME window array
    — e.g. {"z_slow": Z[..., :16], "z_full": Z, "ts2vec": Zb}. tr/te: window
    index arrays from the C3 group split.

    Returns ({arm: {budget: [f1 per draw]}}, meta). Every arm is scored on the
    SAME window draws, so the arms are paired and draw noise cancels in a
    difference — the comparison is within-draw, not between independent curves.
    """
    lo, hi = label_range
    ok = (lab >= lo) & (lab <= hi)
    ok[:, :c_min] = False                                     # C1 context floor
    n_win, n_L = next(iter(feats.values())).shape[:2]
    assert c_min < n_L, f"context floor {c_min} >= window {n_L}"
    assert all(f.shape[:2] == (n_win, n_L) for f in feats.values()), "arms disagree on shape"
    assert lab.shape == (n_win, n_L), "labels do not match the feature grid"
    assert not set(tr.tolist()) & set(te.tolist()), "C3: train and eval windows overlap"

    def cells(idx, cap, rng):
        """Every labelled position inside windows `idx`, as (window id, position)."""
        w, p = np.nonzero(ok[idx])
        w = idx[w]
        if cap and len(w) > cap:
            s = rng.choice(len(w), cap, replace=False)
            w, p = w[s], p[s]
        return w, p

    we, pe = cells(te, max_eval, np.random.default_rng(seed))   # FIXED across budgets
    yte = lab[we, pe] - lo
    labels = np.unique(yte)
    fe = {a: F[we, pe] for a, F in feats.items()}
    ks = [min(b or len(tr), len(tr)) for b in budgets]
    out = {a: {k: [] for k in ks} for a in feats}
    n_pos = {}
    for k in ks:
        for d in range(1 if k == len(tr) else n_draw):    # the full budget has one draw
            rng = np.random.default_rng((seed, k, d))     # draw is independent of arm order
            wd = rng.choice(tr, k, replace=False)
            # the whole experiment rests on this: a labelled window must never be
            # an evaluated window (see _selfcheck for the sample-level version)
            assert not set(wd.tolist()) & set(te.tolist()), "labelled draw hit the eval set"
            wt, pt = cells(wd, max_train, rng)
            ytr = lab[wt, pt] - lo
            n_pos.setdefault(k, []).append(len(ytr))
            for a, F in feats.items():
                out[a][k].append(_fit_score(F[wt, pt], ytr, fe[a], yte, labels))
    meta = dict(budgets=ks, n_draw=n_draw, n_eval=int(len(yte)), n_eval_windows=len(te),
                n_train_windows=len(tr), classes=labels.tolist(), c_min=int(c_min),
                chance_f1=float(1.0 / len(labels)), max_train=max_train,
                n_pos=[int(np.mean(n_pos[k])) for k in ks])
    return out, meta


def common_budgets(cells):
    """Budgets every cell actually has.

    The `None` ("all labelled windows") budget resolves to len(tr), and the C3
    group split holds out a random third of the SUBJECTS, so len(tr) differs
    between seeds (measured: 1409 vs 1431). Pooling on one cell's budget list
    then raises KeyError on another -- which it did, after the whole run had
    finished and before anything was saved."""
    arm = next(iter(cells[0]))
    keep = set(cells[0][arm])
    for c in cells[1:]:
        keep &= set(c[arm])
    return sorted(keep)


def summarize(cells, budgets):
    """Pool draws across cells (stems x encoder seeds) into mean/sd per budget."""
    return {a: dict(budgets=list(budgets),
                    mean=[float(np.mean([f for c in cells for f in c[a][k]])) for k in budgets],
                    sd=[float(np.std([f for c in cells for f in c[a][k]])) for k in budgets],
                    draws=[[float(f) for c in cells for f in c[a][k]] for k in budgets])
            for a in cells[0]}


def _cross(ks, ms, t):
    """First budget whose MEAN reaches t, interpolated in log(budget). The means
    are not monotone at small budgets, so 'first crossing' is a convention — the
    curve itself is the result, this just makes 'fewer labels' a number."""
    i = next((i for i, m in enumerate(ms) if m >= t), None)
    if i is None:
        return float("nan")
    if i == 0:
        return float(ks[0])
    (x0, y0), (x1, y1) = (np.log(ks[i - 1]), ms[i - 1]), (np.log(ks[i]), ms[i])
    return float(np.exp(x0 + (t - y0) * (x1 - x0) / max(y1 - y0, 1e-9)))


def crossings(agg, targets=TARGETS):
    """Budget needed per arm to reach each fraction of the CEILING — the best
    full-budget score across arms, so every arm is measured against one bar
    (an arm's own ceiling would let a weak arm claim efficiency for free)."""
    ceil = max(v["mean"][-1] for v in agg.values())
    return dict(ceiling=float(ceil),
                targets={f"{t:.2f}": {a: _cross(v["budgets"], v["mean"], t * ceil)
                                      for a, v in agg.items()} for t in targets})


def report(agg, meta, cr):
    arms = list(agg)
    print(f"\n=== label efficiency: macro-F1 vs labelled TRAIN windows "
          f"(eval fixed: {meta['n_eval']} positions in {meta['n_eval_windows']} windows, "
          f"chance {meta['chance_f1']:.3f}) ===")
    print(f"{'windows':>8} {'positions':>9} " + " ".join(f"{a:>17}" for a in arms))
    for i, k in enumerate(meta["budgets"]):
        row = " ".join(f"{agg[a]['mean'][i]:9.3f}±{agg[a]['sd'][i]:.3f}" for a in arms)
        print(f"{k:>8} {meta['n_pos'][i]:>9} {row}")
    print(f"\nlabelled windows needed to reach a fraction of the ceiling "
          f"({cr['ceiling']:.3f}); nan = never inside the swept range")
    print(f"{'target':>8} " + " ".join(f"{a:>17}" for a in arms))
    for t, row in cr["targets"].items():
        print(f"{t:>8} " + " ".join(f"{row[a]:>17.1f}" for a in arms))
    print("\n  z_slow beats z_full only if it also beats rand16 — otherwise the gain "
          "is\n  16-vs-64 parameters, not slowness.")


def build_cell(stem, seed, steps, n_draw):
    """Our encoder, gated and ungated, from one C3 split — all arms share the
    window array, so the same draws apply to every arm."""
    from train_real import train_real
    lk, tgt = stem.split("+")
    kw = dict(seed=seed, steps=steps, loss_kind=lk, target_enc=tgt,
              log_every=10 ** 9, **DSETS[DATASET][1])
    res = train_real(str(DSETS[DATASET][0]), **kw)
    ung = train_real(str(DSETS[DATASET][0]), gate=False, xcov=False, **kw)
    assert np.array_equal(res["tr"], ung["tr"]), "gated/ungated split differs"
    Z, Zu = encode_all(res["enc"], res["Wt"]), encode_all(ung["enc"], ung["Wt"])
    Q = rand_subspace(D_Z, D_SLOW, seed)
    feats = {"z_slow": Z[..., :D_SLOW], "z_full": Z, f"rand{D_SLOW}": Z @ Q,
             "ungated_z_full": Zu, "ungated_first16": Zu[..., :D_SLOW]}
    return label_efficiency(feats, res["lab"], res["tr"], res["te"],
                            seed=seed, n_draw=n_draw)


def _selfcheck():
    """Window-level sampling, checked two ways.

    (a) the disjointness assert fires when a labelled draw can reach eval windows;
    (b) on data whose per-window identity is memorisable, the POSITION-level draw
        this file exists to avoid scores near-perfectly at a 5-window budget while
        the window-level draw stays near chance. Rewrite the sampler to pool
        positions and this gap collapses, failing the assert.
    """
    rng = np.random.default_rng(0)
    n_win, Lw, d, k, bud = 24, 96, 32, 4, 5
    y = rng.integers(1, k + 1, n_win)
    ident = 3.0 * rng.standard_normal((n_win, d))            # per-window fingerprint
    shared = 0.3 * rng.standard_normal((k + 1, d))           # the only transferable signal
    Z = (ident[:, None] + shared[y][:, None]
         + 0.1 * rng.standard_normal((n_win, Lw, d))).astype(np.float32)
    lab = np.repeat(y[:, None], Lw, 1)
    tr, te = np.arange(0, n_win, 2), np.arange(1, n_win, 2)

    r, meta = label_efficiency({"z": Z}, lab, tr, te, budgets=(bud,), n_draw=3,
                               c_min=0, max_train=None, max_eval=None)
    good = float(np.mean(r["z"][bud]))

    # the bug: same number of labelled positions, drawn independently, so train and
    # eval positions come from the same (in real data: overlapping) windows
    wi, pi = np.divmod(rng.permutation(n_win * Lw), Lw)
    n = bud * Lw
    a, b = slice(0, n), slice(n, 2 * n)
    lbl = np.unique(lab[wi[b], pi[b]] - 1)
    bad = _fit_score(Z[wi[a], pi[a]], lab[wi[a], pi[a]] - 1,
                     Z[wi[b], pi[b]], lab[wi[b], pi[b]] - 1, lbl)

    try:
        label_efficiency({"z": Z}, lab, np.arange(n_win), te, budgets=(bud,),
                         n_draw=1, c_min=0)
        raise SystemExit("selfcheck FAILED: overlapping train/eval windows accepted")
    except AssertionError:
        pass
    assert bad > good + 0.3, f"position-level leak not detected: {bad:.3f} vs {good:.3f}"
    assert meta["n_eval"] == len(te) * Lw and meta["budgets"] == [bud]
    print(f"selfcheck OK — window-level {good:.3f} vs position-level {bad:.3f} "
          f"at {bud} labels (chance {1/k:.3f}); overlap assert fires")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selfcheck":
        _selfcheck(); sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] in DSETS:
        DATASET = sys.argv.pop(1)
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else DSETS[DATASET][2]
    n_seed = int(sys.argv[2]) if len(sys.argv) > 2 else N_SEED
    n_draw = int(sys.argv[3]) if len(sys.argv) > 3 else N_DRAW
    _selfcheck()                            # the sampler IS the experiment
    cells, tags, meta = [], [], None
    for stem in STEMS:
        for s in range(n_seed):
            print(f"=== {DATASET} label efficiency: {stem} seed {s} "
                  f"({steps} steps, {n_draw} draws/budget) ===", flush=True)
            c, meta = build_cell(stem, s, steps, n_draw)
            cells.append(c); tags.append(f"{stem}/s{s}")
            ks = meta["budgets"]
            print("  " + "  ".join(f"{a} {c[a][ks[0]][0]:.3f}->{c[a][ks[-1]][0]:.3f}"
                                   for a in c), flush=True)
    budgets = common_budgets(cells)
    dropped = [b for b in meta["budgets"] if b not in budgets]
    if dropped:                       # never silently: say which budget was lost
        print(f"  note: budgets {dropped} are not shared by all cells (len(tr) "
              f"varies with the C3 split) — pooling over {budgets}", flush=True)
    meta["budgets"] = budgets
    agg = summarize(cells, budgets)
    cr = crossings(agg)
    out = dict(config=dict(stems=list(STEMS), n_seeds=n_seed, steps=steps,
                           n_draw=n_draw, dataset=DATASET, **DSETS[DATASET][1]),
               meta=meta, agg=agg, crossings=cr,
               cells={t: {a: {str(k): v for k, v in d.items()} for a, d in c.items()}
                      for t, c in zip(tags, cells)})
    path = ROOT / (f"runs_v2/label_efficiency_{DATASET}.json"
                   if steps == DSETS[DATASET][2]
                   else f"runs_v2/label_efficiency_{DATASET}_smoke.json")
    json.dump(out, open(path, "w"), indent=2)
    report(agg, meta, cr)
    print("saved", path)
