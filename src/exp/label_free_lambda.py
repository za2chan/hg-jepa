"""Can lambda be chosen without labels?

The paper concedes that every knob except tau was set by looking at downstream
labels, and Locatello et al. is cited to place that in context. This asks whether
one of them -- the cross-covariance weight -- can be recovered from quantities the
signal already provides.

The criterion is

    J(lambda) = exclusion(lambda) * longhorizon(lambda)

Both factors are label-free:

  exclusion    1 - R^2 of reading the transient proxy from z_slow. The proxy is
               computed from the raw signal (||acc||, instantaneous voltage, EOG
               envelope), never from a task label. This is already SEP's third term.

  longhorizon  R^2 of linearly predicting z_slow(t+Delta) from z_slow(t) for
               Delta > tau, on held-out groups. This is self-supervised: it asks
               whether the block still carries information about its own state a
               long way ahead, which is exactly what "persistent" was defined to
               mean. No target encoder and no label enter it.

Why the product. Exclusion alone is maximised by a block that holds nothing --
emptiness is free. Long-horizon self-predictability alone is maximised by a block
that is constant. Neither degenerate solution survives multiplication, and a
collapsed block is caught anyway by the RankMe guard reported alongside.

Success is measured against the oracle: for how many of the settings does
argmax_lambda J agree with argmax_lambda SEP, where SEP needed labels?

python3 src/exp/label_free_lambda.py -> runs_v2/label_free_lambda.json
"""
import json
import pathlib
import sys

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "hglp"))
sys.path.insert(0, str(HERE.parent))
ROOT = HERE.parents[2]
RUNS = ROOT / "runs_v2"

from model import D_SLOW                                              # noqa: E402
from probes import C_MIN, encode_all, rankme                          # noqa: E402

LAMS = [0.0, 1.0, 4.0, 16.0, 64.0]
MAXN = 60000

# (label, npz, n_ax, train kwargs, rotation json stem). Kwargs must match the ones
# rotation.py used or the checkpoint cache will miss and this would retrain.
SETTINGS = [
    # npz=None marks the synthetic testbed, which trains through train.py on data
    # generated in-process rather than from a file.
    ("Synthetic/Reg", None, 1, dict(tau=16.0), "l1", "rotation_synth_l1+ema_gap0.03"),
    ("Synthetic/NCE", None, 1, dict(tau=16.0), "nce", "rotation_synth_nce+ema_gap0.03"),
    ("PTB-XL/Reg", "../../data/ptbxl_v2.npz", 1,
     dict(tau=16.0, w=8, dmin=8, dmax=48, min_context=8), "l1", "rotation_ptbxl_l1+ema"),
    ("PTB-XL/NCE", "../../data/ptbxl_v2.npz", 1,
     dict(tau=16.0, w=8, dmin=8, dmax=48, min_context=8), "nce", "rotation_ptbxl_nce+ema"),
    ("HAPT/Reg", "../../data/hapt_v2.npz", 3,
     dict(tau=40.0, w=12, dmin=12, dmax=128, min_context=16), "l1", "rotation_hapt_l1+ema"),
    ("HAPT/NCE", "../../data/hapt_v2.npz", 3,
     dict(tau=40.0, w=12, dmin=12, dmax=128, min_context=16), "nce", "rotation_hapt_nce+ema"),
    ("Sleep-EDF/Reg", "../../data/sleepedf_p10.npz", 3,
     dict(tau=9.0, w=8, dmin=8, dmax=64, min_context=16, steps=5000), "l1",
     "rotation_sleepedf_l1+ema_p10A"),
    ("Sleep-EDF/NCE", "../../data/sleepedf_p10.npz", 3,
     dict(tau=9.0, w=8, dmin=8, dmax=64, min_context=16, steps=5000), "nce",
     "rotation_sleepedf_nce+ema_p10A"),
]
SEEDS = [0, 1, 2]


def r2(Xtr, ytr, Xte, yte):
    m = make_pipeline(StandardScaler(), Ridge()).fit(Xtr, ytr)
    return float(m.score(Xte, yte))


def sub(rng, n, cap):
    return rng.choice(n, cap, replace=False) if n > cap else np.arange(n)


def evaluate(r, tau, static, mc=C_MIN):
    """(exclusion, long-horizon self-R^2, RankMe) for one trained model."""
    Z = encode_all(r["enc"], r["Wt"])
    lab, fast, tr, te = r["lab"], r["fast"], r["tr"], r["te"]
    S = Z[:, :, :D_SLOW]
    rng = np.random.default_rng(0)

    # --- exclusion: read the transient proxy out of z_slow -------------------
    if static:
        etr, ete = (S[tr, -1], fast[tr, -1]), (S[te, -1], fast[te, -1])
    else:
        ok = (lab >= 1) & (lab <= 6); ok[:, :C_MIN] = False
        etr = (S[tr][ok[tr]], fast[tr][ok[tr]])
        ete = (S[te][ok[te]], fast[te][ok[te]])
    a, b = sub(rng, len(etr[0]), MAXN), sub(rng, len(ete[0]), MAXN)
    excl = float(np.clip(1.0 - r2(etr[0][a], etr[1][a], ete[0][b], ete[1][b]), 0, 1))

    # --- long horizon: predict z_slow(t+D) from z_slow(t), D > tau -----------
    # This needs only z_slow at two time positions, NOT a label at each position. An
    # earlier version returned NaN whenever the label was static, which conflated
    # "one label per record" with "one position per record" -- PTB-XL has 100
    # positions in its window and the self-prediction is perfectly well defined there.
    # The floor is the dataset's own min_context, not HAPT's probe floor C_MIN=64,
    # which on PTB-XL's L=100 would have left four usable positions.
    L = S.shape[1]
    D = int(min(max(int(np.ceil(tau)) * 2, int(np.ceil(tau)) + 1), L - mc - 1))
    src = np.arange(mc, L - D)
    Xtr = S[tr][:, src].reshape(-1, D_SLOW); Ytr = S[tr][:, src + D].reshape(-1, D_SLOW)
    Xte = S[te][:, src].reshape(-1, D_SLOW); Yte = S[te][:, src + D].reshape(-1, D_SLOW)
    a, b = sub(rng, len(Xtr), MAXN), sub(rng, len(Xte), MAXN)
    lh = float(np.clip(r2(Xtr[a], Ytr[a], Xte[b], Yte[b]), 0, 1))
    return excl, lh, float(rankme(Xte[b][:5000]))


def run_synth(lk, gap=0.03):
    """Same criterion on the synthetic testbed, where the answer is known.

    Worth doing precisely because the penalty is NOT inert here -- the oracle picks
    lam=1 (Reg) and lam=16 (NCE), never 0 -- so a criterion that only ever says
    "turn it off" cannot pass by accident. The transient factor u is the generator's
    own variable rather than a proxy read off the signal, which removes the one
    place a reader could argue the exclusion term is measuring the wrong thing.
    """
    import torch
    from train import train, DEV
    from datagen import make_dataset
    from model import P, L

    tau = 16.0
    Dh = int(tau) * 2                                   # one horizon beyond the gate
    # The evaluation set is fixed at seed=99 and so is identical for every lam and
    # every training seed; building it once turns 30 generations into one.
    d = make_dataset(300_000, seed=99, gap=gap)          # must match the training gap
    x, u = d["x"], d["u"]
    rng0 = np.random.default_rng(99)
    starts = rng0.integers(0, len(x) - L * P - 1, 400)
    xb = torch.from_numpy(np.stack([x[st:st + L * P].reshape(L, P)
                                    for st in starts])).to(DEV)
    rows = {}
    for lam in LAMS:
        e_, l_, rm = [], [], []
        for s in SEEDS:
            enc = train(loss_kind=lk, target_enc="ema", seed=s, gap=gap, lam=lam,
                        log_every=10 ** 9)["enc"]
            rng = np.random.default_rng(99)
            with torch.no_grad():
                Z = torch.cat([enc(xb[i:i + 128])
                               for i in range(0, len(xb), 128)]).cpu().numpy()
            S = Z[:, :, :D_SLOW]
            cut = np.sort(starts)[len(starts) // 2]      # split on window start, not row
            tr, te = starts + L * P <= cut, starts > cut

            pos = np.arange(C_MIN, L - Dh)
            flat = lambda M, m: M[m][:, pos].reshape(-1, M.shape[-1])
            yu = np.stack([u[starts + a * P + (P - 1)] for a in pos], 1)
            a_, b_ = sub(rng, tr.sum() * len(pos), MAXN), sub(rng, te.sum() * len(pos), MAXN)
            e_.append(np.clip(1.0 - r2(flat(S, tr)[a_], yu[tr].reshape(-1)[a_],
                                       flat(S, te)[b_], yu[te].reshape(-1)[b_]), 0, 1))
            Ytr = S[tr][:, pos + Dh].reshape(-1, D_SLOW)
            Yte = S[te][:, pos + Dh].reshape(-1, D_SLOW)
            l_.append(np.clip(r2(flat(S, tr)[a_], Ytr[a_], flat(S, te)[b_], Yte[b_]), 0, 1))
            rm.append(float(rankme(flat(S, te)[b_][:5000])))
        rows[lam] = (float(np.mean(e_)), float(np.mean(l_)), float(np.mean(rm)))
    return rows


def sep_from_rotation(stem, lam):
    """The oracle: SEP at this lambda, from the stored rotation run."""
    p = RUNS / f"{stem}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    k = f"gate@lam{lam:g}"
    if k not in d or "fast_half" not in d[k]:
        return None
    a, b = d[k]["slow_half"], d[k]["fast_half"]
    c = lambda v: float(np.clip(v, 0, 1))
    return c(a["slow"][0]) * c(b["fast"][0]) * c(1 - a["fast"][0])


def main():
    from train_real import train_real
    out, agree = {}, 0
    print(f"{'setting':16s}{'lam':>5s}{'excl':>7s}{'longH':>7s}{'J':>7s}{'RankMe':>8s}{'SEP*':>7s}")
    for label, npz, n_ax, kw, lk, stem in SETTINGS:
        synth = npz is None
        static = (not synth) and "ptbxl" in npz
        srows = run_synth(lk) if synth else None
        rows = {}
        for lam in LAMS:
            if synth:
                E, LH, RM = srows[lam]
            else:
                e_, l_, rm = [], [], []
                for s in SEEDS:
                    r = train_real(npz, n_ax=n_ax, seed=s, loss_kind=lk,
                                   target_enc="ema", lam=lam, log_every=10 ** 9, **kw)
                    a, b, c = evaluate(r, kw["tau"], static, kw["min_context"])
                    e_.append(a); l_.append(b); rm.append(c)
                E, LH, RM = (float(np.mean(e_)), float(np.nanmean(l_)), float(np.mean(rm)))
            J = E * LH if np.isfinite(LH) else float("nan")
            oracle = sep_from_rotation(stem, lam)
            rows[lam] = dict(exclusion=E, longhorizon=LH, J=J, rankme=RM, sep=oracle)
            print(f"{label:16s}{lam:5g}{E:7.3f}{LH:7.3f}{J:7.3f}{RM:8.1f}"
                  f"{(oracle if oracle is not None else float('nan')):7.3f}")
        valid = {k: v for k, v in rows.items() if np.isfinite(v["J"])}
        if valid and all(v["sep"] is not None for v in valid.values()):
            jbest = max(valid, key=lambda k: valid[k]["J"])
            obest = max(valid, key=lambda k: valid[k]["sep"])
            hit = jbest == obest
            agree += hit
            print(f"  -> J picks lam={jbest:g}, oracle picks lam={obest:g}  "
                  f"{'MATCH' if hit else 'miss'}")
            rows["_pick"] = dict(J=jbest, oracle=obest, match=bool(hit))
        out[label] = rows

    n = sum(1 for v in out.values() if "_pick" in v)
    print(f"\nagreement: {agree}/{n} settings")
    out["_summary"] = dict(agree=agree, n=n, lams=LAMS, seeds=SEEDS)
    (RUNS / "label_free_lambda.json").write_text(json.dumps(out, indent=1))
    print("-> runs_v2/label_free_lambda.json")


if __name__ == "__main__":
    main()
