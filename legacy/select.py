"""Retrospective validation of LABEL-FREE model selection (reviewer #3 /
Locatello: hyperparameter tuning smuggles labels in).

For every run with a saved embedding (runs/emb_nepa_*.npz, time-ordered
final-patch embeddings + labels), compute unsupervised health metrics:
  rankme_slow / rankme_fast : effective rank per block (collapse detector)
  cross_corr                : mean |corr| between blocks (duplication)
  slowness gap              : fast-block temporal roughness minus slow-block's
                              (separation should make z_slow the slow one)
and a label-free criterion:
  valid  = rankme_slow >= 3
  UNSUP  = (rough_fast - rough_slow) - cross_corr      (higher = better)
Then unseal the labels: LABELED = regime_acc(z_slow) - mean(fast leak, clipped)
and ask: had we picked the config by UNSUP alone, how close to the labeled
best would we have landed? Reports per-config table, Spearman, top-1 regret.

Usage: python3 select.py
"""
import glob
import json
import re

import numpy as np
from scipy.stats import spearmanr

D_Z = 64


def rankme(B):
    s = np.linalg.svd(B - B.mean(0), compute_uv=False)
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def roughness(B):
    # mean squared step / (2*var): ~1 for white noise, ->0 for slow drift
    return float((np.diff(B, axis=0) ** 2).mean() / (2 * B.var(0).mean() + 1e-12))


def main():
    rows = []
    for f in sorted(glob.glob("runs/emb_nepa_*.npz")):
        tag = f.split("emb_")[1][:-4]
        d_slow = int(re.search(r"_ds(\d+)_", tag).group(1))
        j = json.load(open(f"runs/{tag}.json"))
        d = np.load(f)
        Zs, Zf = d["Z"][:, :d_slow], d["Z"][:, d_slow:]
        C = np.corrcoef(Zs.T, Zf.T)[:d_slow, d_slow:]
        unsup = dict(rankme_slow=rankme(Zs), rankme_fast=rankme(Zf),
                     cross_corr=float(np.abs(C).mean()),
                     rough_slow=roughness(Zs), rough_fast=roughness(Zf))
        unsup["UNSUP"] = ((unsup["rough_fast"] - unsup["rough_slow"])
                          - unsup["cross_corr"]) if unsup["rankme_slow"] >= 3 else -9
        labeled = (j["z_slow->regime_acc"]
                   - 0.5 * (max(j["z_slow->u_r2"], 0) + max(j["z_slow->phase_r2"], 0)))
        cfg = re.sub(r"_s\d+_", "_", tag)          # aggregate over seeds
        rows.append((cfg, unsup, labeled))

    # aggregate seeds per config
    cfgs = {}
    for cfg, u, lab in rows:
        cfgs.setdefault(cfg, []).append((u, lab))
    print(f"{'config':44s} {'UNSUP':>7s} {'LABELED':>8s} "
          f"{'rkS':>5s} {'xcorr':>6s} {'roughS/F':>10s}")
    agg = []
    for cfg, rs in sorted(cfgs.items()):
        u = {k: np.mean([r[0][k] for r in rs]) for k in rs[0][0]}
        lab = np.mean([r[1] for r in rs])
        agg.append((cfg, u["UNSUP"], lab))
        print(f"{cfg:44s} {u['UNSUP']:7.3f} {lab:8.3f} {u['rankme_slow']:5.1f} "
              f"{u['cross_corr']:6.3f} {u['rough_slow']:.2f}/{u['rough_fast']:.2f}")

    us = [a[1] for a in agg]; ls = [a[2] for a in agg]
    rho = spearmanr(us, ls)
    pick = max(agg, key=lambda a: a[1])
    best = max(agg, key=lambda a: a[2])
    print(f"\nSpearman(UNSUP, LABELED) = {rho.statistic:.3f} (p={rho.pvalue:.4f})")
    print(f"unsup pick : {pick[0]}  labeled={pick[2]:.3f}")
    print(f"label best : {best[0]}  labeled={best[2]:.3f}")
    print(f"top-1 regret = {best[2] - pick[2]:.3f}")


if __name__ == "__main__":
    main()
