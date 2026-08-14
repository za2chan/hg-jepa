"""How much does the random-subspace null move between draws?

The exclusion term divides by the fast-factor score of ONE random d-dim subspace
(averaged over n_rand=3 draws, whose spread we never stored). If that spread is
large, calling the null "the measurable range" is not defensible and the whole
exclusion axis inherits the noise -- so this has to be settled before any table
is built, and before deciding whether to keep the random null at all or switch to
the full embedding's fast score as the reference.

Draws 30 random subspaces per (dataset, stem) and reports mean, sd, and the
inter-quartile range. Verdict: sd within 10% of the mean means n_rand=3 is fine.

Usage: python3 null_stability.py [n_draw]
Writes runs_v2/null_stability.json
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hglp"))

import numpy as np

from model import D_SLOW, D_Z
from probes import score, rand_subspace
from twosided import synth_feats, real_feats, DATASETS

STEMS = ["l1+ema", "nce+ema"]
CAP = 20000          # smaller than the C1 cap: we need the VARIANCE, not the exact value


def feats_for(dataset, stem, seed, steps):
    lk, tg = stem.split("+")
    if dataset == "synth":
        from train import train
        r = train(loss_kind=lk, target_enc=tg, seed=seed, steps=steps,
                  gap=0.01, log_every=10 ** 9)      # the hard setting is the one we report
        return synth_feats(r["enc"], gap=0.01)
    from train_real import train_real
    npz, n_ax, kw, static = DATASETS[dataset]
    r = train_real(npz, n_ax=n_ax, seed=seed, loss_kind=lk, target_enc=tg,
                   steps=steps, log_every=10 ** 9, **kw)
    return real_feats(r, static)


def run(dataset, stem, seed, steps, n_draw):
    Ftr, ytr, ztr, Fte, yte, zte = feats_for(dataset, stem, seed, steps)
    rng = np.random.default_rng(0)
    a = rng.choice(len(Ftr), min(CAP, len(Ftr)), replace=False)
    b = rng.choice(len(Fte), min(CAP, len(Fte)), replace=False)
    Ftr, ytr, ztr, Fte, yte, zte = Ftr[a], ytr[a], ztr[a], Fte[b], yte[b], zte[b]
    fast, slow = [], []
    for i in range(n_draw):
        Q = rand_subspace(D_Z, D_SLOW, 7000 + i)
        s, f = score(Ftr @ Q, ytr, ztr, Fte @ Q, yte, zte)
        slow.append(s); fast.append(f)
    full = score(Ftr, ytr, ztr, Fte, yte, zte)
    return dict(fast_mean=float(np.mean(fast)), fast_sd=float(np.std(fast)),
                fast_q=[float(np.percentile(fast, q)) for q in (25, 50, 75)],
                fast_min=float(np.min(fast)), fast_max=float(np.max(fast)),
                slow_mean=float(np.mean(slow)), slow_sd=float(np.std(slow)),
                z_full_slow=float(full[0]), z_full_fast=float(full[1]),
                n_draw=n_draw, draws_fast=[float(x) for x in fast])


if __name__ == "__main__":
    n_draw = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
    os.makedirs("../../runs_v2", exist_ok=True)
    out = {}
    for ds in ("synth", "ptbxl", "hapt"):
        for stem in STEMS:
            k = f"{ds}/{stem}"
            out[k] = run(ds, stem, 0, steps, n_draw)
            r = out[k]
            print(f"  {k:18s} null fast {r['fast_mean']:.3f} ± {r['fast_sd']:.3f} "
                  f"(sd/mean {r['fast_sd']/max(r['fast_mean'],1e-9)*100:5.1f}%) "
                  f"| range [{r['fast_min']:.3f}, {r['fast_max']:.3f}] "
                  f"| z_full fast {r['z_full_fast']:.3f}", flush=True)
    json.dump(out, open("../../runs_v2/null_stability.json", "w"), indent=2)

    print(f"\n{'데이터셋/스템':20}{'변동계수':>10}{'판정':>12}")
    for k, r in out.items():
        cv = r["fast_sd"] / max(r["fast_mean"], 1e-9)
        print(f"{k:20}{cv*100:9.1f}%{'안정' if cv < 0.10 else '불안정':>12}")
    print("\n  변동계수 = sd/mean. 10% 미만이면 n_rand=3 유지, 그 이상이면 "
          "기준선을 z_full 로 교체하거나 n_rand 를 늘려야 합니다.")
    print("saved runs_v2/null_stability.json")
