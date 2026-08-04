"""Three-point stem comparison on synthetic under the v2 design (bounded target,
Δ-first anchors, per-block LN, continuous Δ):

    reg+ema     HGLP-Reg   (headline stem, D4)
    nce+ema     control    (v1's 'cpc' cell — stabilized target, contrastive loss)
    nce+online  HGLP-NCE   (D2 pure form: both-sided grads, no EMA, no stop-grad)

v1 found: reg+ema 0.810 / nce+ema 0.810 both worked, nce+online 0.456 failed
(vacuous low leak, z_fast carried u at R2~0) -> D4 kept Reg. This re-tests that
under v2, where the target no longer re-contains the anchor's past.
Writes runs_v2/stems_synth.json.
"""
import json
import os

import numpy as np
import torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import D_Z, D_SLOW
from train import train, rankme, DEV
from datagen import make_dataset
from model import P, L

STEMS = [("reg", "ema"), ("nce", "ema"), ("nce", "online")]


@torch.no_grad()
def probe(enc, seed=99, n_win=400, positions=range(24, 240, 12)):
    """Multi-position probe (C1) on held-out synthetic windows, per-position
    ground truth at the end sample of each patch."""
    d = make_dataset(300_000, seed=seed)
    x, s, u = d["x"], d["s"], d["u"]
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(x) - L * P - 1, n_win)
    xb = torch.from_numpy(np.stack([x[st:st + L * P].reshape(L, P)
                                    for st in starts])).to(DEV)
    Z = torch.cat([enc(xb[i:i + 128]) for i in range(0, len(xb), 128)]).cpu().numpy()
    out = {}
    for name, sl in [("z_slow", slice(0, D_SLOW)), ("z_fast", slice(D_SLOW, D_Z)),
                     ("z_full", slice(0, D_Z))]:
        F_, ys, yu = [], [], []
        for a in positions:
            idx = starts + a * P + (P - 1)
            F_.append(Z[:, a, sl]); ys.append(s[idx]); yu.append(u[idx])
        F_ = np.concatenate(F_); ys = np.concatenate(ys); yu = np.concatenate(yu)
        m = len(F_) // 2
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        acc = clf.fit(F_[:m], ys[:m]).score(F_[m:], ys[m:])
        rg = make_pipeline(StandardScaler(), Ridge())
        leak = rg.fit(F_[:m], yu[:m]).score(F_[m:], yu[m:])
        out_std = float(F_.std())
        out[name] = dict(regime_acc=float(acc), leak_u_r2=float(leak),
                         rankme=float(rankme(F_[:5000])), feat_std=out_std)
    return out


if __name__ == "__main__":
    os.makedirs("../runs_v2", exist_ok=True)
    res = {}
    for lk, te in STEMS:
        tag = f"{lk}+{te}"
        print(f"\n=== {tag} ===")
        r = train(loss_kind=lk, target_enc=te, seed=0)
        pr = probe(r["enc"])
        pr["loss_final"] = float(np.mean(r["losses"][-100:]))
        res[tag] = pr
        for b in ("z_slow", "z_fast", "z_full"):
            print(f"  {b:7s} regime {pr[b]['regime_acc']:.3f} | leak(u R2) "
                  f"{pr[b]['leak_u_r2']:+.3f} | RankMe {pr[b]['rankme']:.1f}")
    json.dump(res, open("../runs_v2/stems_synth.json", "w"), indent=2)

    print("\n=== SUMMARY (slow-kept / leak from z_slow; z_fast should CARRY u) ===")
    for tag, pr in res.items():
        vac = "VACUOUS" if pr["z_fast"]["leak_u_r2"] < 0.2 else "ok"
        print(f"  {tag:11s} slow-kept {pr['z_slow']['regime_acc']:.3f} | "
              f"leak {pr['z_slow']['leak_u_r2']:+.3f} | "
              f"z_fast carries u {pr['z_fast']['leak_u_r2']:+.3f} ({vac})")
    print("saved runs_v2/stems_synth.json")
