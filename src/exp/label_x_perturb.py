"""Does z_slow help when labels are SCARCE and the fast factor is perturbed?

The two experiments we ran covered the axes separately: label efficiency on clean
data, and perturbation robustness with the full label set. Both were negative,
and the explanation offered was that a supervised linear probe already learns to
ignore the fast coordinates, so removing them in advance buys nothing.

That explanation makes a prediction it has never been tested against: the probe
can only make that selection if it has enough labels. With few labels it should
lean on fast coordinates spuriously, a fast-factor perturbation should then break
z_full, and z_slow -- which has no fast coordinates to lean on -- should survive.
If the crossover does not appear, the explanation is wrong and the failure needs
a different account.

Fitting is always on CLEAN data; only the evaluation windows are perturbed, so
the probe never sees the corruption it is judged on.

Usage: python3 label_x_perturb.py [ptbxl|hapt|synth] [steps] [n_seed] [n_draw]
       HGLP_TAG=_smoke for a short-step trial (do NOT overwrite the real result)
Writes runs_v2/label_x_perturb_<dataset>.json
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hglp"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "baselines"))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import D_SLOW, D_Z, P, L
from probes import rand_subspace

BUDGETS = (10, 25, 50, 100, 250, None)    # None = every train window
# per dataset: synth's fast attack REPLACES a fraction of u, so 1.0 (full
# replacement) is its ceiling -- asking for 2.0 there silently re-ran 1.0
LEVELS = {"ptbxl": (0.0, 0.5, 1.0, 2.0), "hapt": (0.0, 0.5, 1.0, 2.0),
          "synth": (0.0, 0.5, 1.0, 2.0), "synthseries": (0.0, 0.25, 0.5, 1.0)}
# which last-position label values are real classes. PTB-XL's is a binary
# diagnosis where 0 IS a class; HAPT's 0 is unlabeled and 7-12 are postural
# transitions, so only 1-6 are scored (the C1 label_range the probes use).
LABEL_RANGE = {"ptbxl": None, "hapt": (1, 6), "synth": (1, 3), "synthseries": None}
# TS2Vec / PatchTST arms. Part 2 only: the question there is not "does the block
# separate" but "how much does the score drop when the encoder meets a domain it was
# not trained on, as probe labels get scarce". That is a WITHIN-arm difference, so
# the baselines' much larger embedding width (TS2Vec 320, PatchTST C*128) cancels.
BASELINES = os.environ.get("HGLP_BASELINES", "1") == "1"
BASE_ARMS = ("ts2vec", "patchtst")
STRENGTHS = LEVELS["ptbxl"]               # rebound per dataset in run()
STEMS = ["l1+ema", "nce+ema"]
GAP = float(os.environ.get("HGLP_GAP", 0.01))   # synthetic difficulty (regime spacing)
PERT_KIND = os.environ.get("HGLP_PERT", "scale")
# xcov weight per loss kind, for OUR model only -- the no-mechanism baseline has
# xcov off, so lam does not reach it. Needed because lam=4 collapses L1's inclusion
# on +-3% synthetic (0.964 at lam=1 -> 0.524), which would report a downstream cost
# that is a hyperparameter artefact rather than the mechanism's price.
#   HGLP_LAM='{"l1": 1.0}'
LAM = json.loads(os.environ.get("HGLP_LAM", "{}"))
TAG = os.environ.get("HGLP_TAG", "")       # set this for smoke tests; a short-step
# run writing to the real filename destroys the real result and it is NOT in git
# Two fit protocols, same encoder (always trained on CLEAN data), same perturbed
# eval set. They answer different questions and we report both:
#   A  fit on CLEAN train rows      -> zero-shot transfer into the shifted domain
#   B  fit on PERTURBED train rows  -> label efficiency WITHIN the shifted domain
# B is the one the label axis speaks to directly: with few labels the probe cannot
# work out which coordinates to trust in the new domain, and z_slow only offers 16.
# Caveat for B: `scale` is a global gain, which StandardScaler partly absorbs when
# you refit in the same domain -- a null there may mean the attack is weak, not that
# z_slow is useless. Re-run with HGLP_PERT=noise|wander before concluding.


# The question is "what does SEPARATING the embedding buy you", so the reference
# cannot be our own model's z_full -- that shares the encoder and only differs in
# width, which measures overfitting, not separation. The reference is a model
# trained with NO mechanism at all (no gate, no xcov, no per-block LN):
#   z_slow    ours, the 16-dim gated block            <- the claim
#   z_full    ours, all 64                            <- width control, same encoder
#   rand16    ours, a random 16-dim readout           <- is the BLOCK special?
#   base_full no-mechanism model, all 64              <- THE baseline
#   base16    no-mechanism model, its first 16        <- width-matched baseline
ARMS = ("z_slow", "z_full", "rand16", "base_full", "base16") + \
       (BASE_ARMS if BASELINES else ())


def _arm_feats(Z, Z0, Q):
    return Z[:, :D_SLOW], Z, Z @ Q, Z0, Z0[:, :D_SLOW]


def _fit_score(ftr, ytr, fte, yte, labels):
    if len(np.unique(ytr)) < 2:
        return float(f1_score(yte, np.full_like(yte, ytr[0]), average="macro",
                              labels=labels, zero_division=0))
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, class_weight="balanced"))
    return float(f1_score(yte, clf.fit(ftr, ytr).predict(fte), average="macro",
                          labels=labels, zero_division=0))


# ------------------------------------------------------- real data (PTB-XL, HAPT)
def real_case(dataset, seed, steps):
    """Label read at the last position -- the only position a bidirectional baseline
    could also be scored at, so the arms stay comparable. Perturbation is amplitude
    scaling, which targets the fast factor (instantaneous ECG voltage / accmag)."""
    from train_real import train_real
    from common import load_windows
    from baseline_robust import DSET, last, perturb
    npz, n_ax, kw = DSET[dataset]
    Wt, lab, fast, tr, te = load_windows(npz, n_ax, seed=seed)
    y = lab[:, -1]
    if LABEL_RANGE[dataset] is not None:      # HAPT: 0 = unlabeled, 7-12 = transitions
        lo, hi = LABEL_RANGE[dataset]
        ok = (y >= lo) & (y <= hi)
        tr, te = tr[ok[tr]], te[ok[te]]       # drop the windows, keep original indexing
    g = torch.Generator(device=Wt.device).manual_seed(seed)
    # the SSL baselines do not depend on our stem, so fit them once per seed
    base_enc = {}
    for name in (BASE_ARMS if BASELINES else ()):
        mod = __import__(f"{name}_adapter")
        b = mod.fit_encoder(npz, n_ax, train_idx=tr, seed=seed, steps=steps)
        base_enc[name] = b["embed"]
        print(f"  {name}: D={b['D']} n_train={len(tr)}"
              + ("  ⚠ D > n_train: 사후 부분공간이 미결정" if b["D"] > len(tr) else ""),
              flush=True)
    out = {}
    for stem in STEMS:
        lk, tg = stem.split("+")
        common = dict(npz=npz, n_ax=n_ax, seed=seed, loss_kind=lk, target_enc=tg,
                      steps=steps, log_every=10 ** 9, **kw)
        r = train_real(**{**common, **({"lam": LAM[lk]} if lk in LAM else {})})  # ours
        r0 = train_real(gate=False, xcov=False, blocknorm=False, **common)  # no mechanism
        Q = rand_subspace(D_Z, D_SLOW, seed)
        def arms(W):
            d = dict(zip(ARMS, _arm_feats(last(r["enc"], W), last(r0["enc"], W), Q)))
            for name in (BASE_ARMS if BASELINES else ()):
                d[name] = last(base_enc[name], W)
            return d
        # perturb ALL windows, not just test: fit-B needs perturbed TRAIN rows too
        pert = {s: arms(perturb(Wt, PERT_KIND, s, g, n_ax=n_ax)) for s in STRENGTHS}
        out[stem] = (arms(Wt), pert, y, tr, te)
    return out


# -------------------------------------------------------------- synthetic
def synth_case(seed, steps):
    """Regime label per position; the fast attack replaces a fraction of u with an
    independent OU draw, which moves the fast factor and leaves the carrier alone."""
    from train import train, DEV
    from synth_robust import corrupt_series, encode_windows, POS, N_WIN, DATA_SEED
    out = {}
    x, sreg, attacks = corrupt_series(gap=GAP)
    rng = np.random.default_rng(DATA_SEED)
    starts = rng.integers(0, len(x) - L * P - 1, N_WIN)
    y = np.concatenate([sreg[starts + a * P + (P - 1)] for a in POS])
    wid = np.tile(np.arange(N_WIN), len(POS))
    cut = np.sort(starts)[N_WIN // 2]
    tr = np.flatnonzero(starts[wid] + L * P <= cut)
    te = np.flatnonzero(starts[wid] > cut)
    lv = {0.0: None, 0.25: ("fast", 0.25), 0.5: ("fast", 0.5), 1.0: ("fast", 1.0)}
    for stem in STEMS:
        lk, tg = stem.split("+")
        common = dict(loss_kind=lk, target_enc=tg, seed=seed, steps=steps, gap=GAP,
                      log_every=10 ** 9)
        r = train(**common)                                                # our method
        r0 = train(gate=False, xcov=False, blocknorm=False, **common)      # no mechanism
        Q = rand_subspace(D_Z, D_SLOW, seed)
        enc = lambda e, xs: np.concatenate([encode_windows(e, xs, starts)[:, a] for a in POS])
        arms = lambda xs: dict(zip(ARMS, _arm_feats(enc(r["enc"], xs), enc(r0["enc"], xs), Q)))
        pert = {s: arms(x if key is None else attacks[key][0]) for s, key in lv.items()}
        out[stem] = (arms(x), pert, y, tr, te)
    return out


def run(dataset, seed, steps, n_draw):
    global STRENGTHS
    STRENGTHS = LEVELS[dataset]
    case = synth_case(seed, steps) if dataset == "synthseries" else real_case(dataset, seed, steps)
    res = {}
    for stem, (clean, pert, y, tr, te) in case.items():
        # the eval set is FIXED across budgets and strengths so curves are comparable
        yte, labels = y[te], np.unique(y[tr])
        for b in BUDGETS:
            # key on the BUDGET LABEL, not on the resolved count: `None` resolves to
            # len(tr), which differs per seed under the C3 group split, and keying on
            # it silently splits the full-budget row into one row per seed
            k = min(b or len(tr), len(tr))
            for d in range(1 if b is None else n_draw):
                rng = np.random.default_rng((seed, k, d))
                idx = rng.choice(tr, k, replace=False)
                for arm in ARMS:
                    for s in STRENGTHS:
                        for fit, F in (("A", clean[arm]), ("B", pert[s][arm])):
                            v = _fit_score(F[idx], y[idx], pert[s][arm][te], yte, labels)
                            res.setdefault(f"{stem}/{fit}/{arm}/{b}/{s}", []).append(v)
        print(f"  {stem} done", flush=True)
    return res


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "ptbxl"
    gapsuf = f"_gap{GAP:g}" if ds == "synth" else ""
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
    n_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    n_draw = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    os.makedirs("../../runs_v2", exist_ok=True)
    cells = [run(ds, s, steps, n_draw) for s in range(n_seed)]
    keys = sorted(set().union(*[set(c) for c in cells]))
    # keep the PER-SEED means: the paired sd across seeds is the only thing that
    # says whether a +0.03 gap is a result or noise, and averaging it away here is
    # what made the previous run unusable
    per_seed = {k: [float(np.mean(c[k])) for c in cells if k in c] for k in keys}
    agg = {k: float(np.mean(v)) for k, v in per_seed.items()}
    sd = {k: float(np.std(v, ddof=1)) if len(v) > 1 else None for k, v in per_seed.items()}
    json.dump(dict(agg=agg, sd=sd, per_seed=per_seed,
                   config=dict(dataset=ds, steps=steps, n_seed=n_seed, n_draw=n_draw,
                               budgets=[str(b) for b in BUDGETS], strengths=list(STRENGTHS),
                               gap=GAP, pert_kind=PERT_KIND, lam_override=LAM, fits=dict(
                                   A="probe fit on CLEAN train rows (zero-shot transfer)",
                                   B="probe fit on PERTURBED train rows (in-domain label efficiency)"))),
              open(f"../../runs_v2/label_x_perturb_{ds}{gapsuf}{TAG}.json", "w"), indent=2)

    def gap_vs(stem, fit, base, b, s, arm="z_slow"):
        """paired-by-seed arm − base, so the sd is the sd of the DIFFERENCE"""
        u = per_seed.get(f"{stem}/{fit}/{arm}/{b}/{s}"); v = per_seed.get(f"{stem}/{fit}/{base}/{b}/{s}")
        if not u or not v:
            return None, None
        d = np.array(u) - np.array(v)
        return float(d.mean()), (float(d.std(ddof=1)) if len(d) > 1 else None)

    for fit, title in (("A", "깨끗한 데이터로 프로브 적합 → 교란 평가 (제로샷 전이)"),
                       ("B", "교란 데이터로 프로브 적합 → 교란 평가 (도메인 내 라벨 효율성)")):
        for base in ("base_full", "base16", "z_full", "rand16"):
            print(f"\n=== [{fit}] {title} | z_slow − {base} ({PERT_KIND}) ===")
            for stem in STEMS:
                print(f"\n{stem}")
                print(f"{'예산':>8}" + "".join(f"{f'교란 {s}':>16}" for s in STRENGTHS))
                for b in BUDGETS:
                    row = ""
                    for s in STRENGTHS:
                        m, e = gap_vs(stem, fit, base, b, s)
                        row += f"{'—':>16}" if m is None else \
                               f"{m:+.3f}±{e:.3f}".rjust(16) if e is not None else f"{m:+16.3f}"
                    print(f"{str(b):>8}" + row)
    print("\n  양수 = z_slow 우세. ± 는 시드 간 짝지은 차이의 표준편차 "
          "— |평균| < ±면 잡음입니다.")
    print("  예측(N14): 같은 교란 열 안에서 예산이 줄수록 커진다.")
    print(f"saved runs_v2/label_x_perturb_{ds}{gapsuf}{TAG}.json")
