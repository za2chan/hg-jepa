"""WHERE in the embedding does the fast factor live? — the variance question.

On HAPT a plain PCA of the ungated embedding excludes the fast proxy about as
well as our gate, while on PTB-XL it does not. The proposed explanation was that
HAPT's fast proxy sits in LOW-VARIANCE directions, so any variance-ranked
selection drops it for free and the gate has nothing left to do. That was an
inference from the leak numbers, not a measurement. This measures it.

Method: take the ungated (g0_x0) embedding, PCA it, and for every component
record (i) its share of embedding variance and (ii) the univariate held-out R^2
of the fast proxy from that component alone. If the fast factor really hides in
the tail, the top-d variance components carry little of it.

Why the two datasets should differ:
  HAPT   fast = ||acc(t)||, dominated by a CONSTANT 1 g of gravity; the informative
         part is a small residual (measured: 99.4% of its variance is within-activity,
         static-activity std 0.013-0.017 around a mean of 1.02)
  PTB-XL fast = the instantaneous ECG voltage, which swings through the full dynamic
         range every beat and has no such constant pedestal

Usage: python3 fastaxis.py
Writes runs_v2/fastaxis.json, runs_v2/fig_fastaxis.png
"""
import json
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hglp"))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import D_SLOW, D_Z
from twosided import real_feats, DATASETS


def spectrum(Ftr, ztr, Fte, zte, d=D_SLOW):
    """Per-principal-component variance share and univariate fast-proxy R^2."""
    p = PCA(n_components=Ftr.shape[1]).fit(Ftr)
    Atr, Ate = p.transform(Ftr), p.transform(Fte)
    var = p.explained_variance_ratio_
    r2 = np.array([make_pipeline(StandardScaler(), Ridge())
                   .fit(Atr[:, [i]], ztr).score(Ate[:, [i]], zte) for i in range(Atr.shape[1])])
    r2 = np.maximum(r2, 0.0)
    joint = lambda cols: float(make_pipeline(StandardScaler(), Ridge())
                               .fit(Atr[:, cols], ztr).score(Ate[:, cols], zte))
    top, tail = list(range(d)), list(range(d, Atr.shape[1]))
    return dict(var_share=var.tolist(), fast_r2_per_pc=r2.tolist(),
                var_share_top=float(var[:d].sum()),
                fast_r2_top=joint(top), fast_r2_tail=joint(tail),
                fast_r2_all=joint(list(range(Atr.shape[1]))),
                fast_r2_share_top=float(r2[:d].sum() / max(r2.sum(), 1e-9)))


def main():
    from train_real import train_real
    out = {}
    for name in ("hapt", "ptbxl"):
        npz, n_ax, kw, static = DATASETS[name]
        r = train_real(npz, n_ax=n_ax, seed=0, gate=False, xcov=False,
                       loss_kind="nce", target_enc="ema", log_every=10 ** 9, **kw)
        Ftr, ytr, ztr, Fte, yte, zte = real_feats(r, static)
        out[name] = spectrum(Ftr, ztr, Fte, zte)
        o = out[name]
        print(f"\n=== {name} (ungated embedding, {len(Ftr)} train rows) ===")
        print(f"  top-{D_SLOW} PCs hold {o['var_share_top']*100:5.1f}% of embedding variance")
        print(f"  fast proxy R2 from top-{D_SLOW} PCs : {o['fast_r2_top']:.3f}")
        print(f"  fast proxy R2 from the tail 48 PCs : {o['fast_r2_tail']:.3f}")
        print(f"  fast proxy R2 from all 64 PCs      : {o['fast_r2_all']:.3f}")
        print(f"  -> share of per-PC fast R2 sitting in the top-{D_SLOW}: "
              f"{o['fast_r2_share_top']*100:.1f}%")
    json.dump(out, open("../../runs_v2/fastaxis.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, name in enumerate(("hapt", "ptbxl")):
        o = out[name]
        v, r2 = np.array(o["var_share"]), np.array(o["fast_r2_per_pc"])
        a = ax[j]
        a.bar(np.arange(len(v)), v / v.max(), color="0.75", label="variance share (normalised)")
        a.plot(np.arange(len(r2)), r2 / max(r2.max(), 1e-9), "o-", ms=3, color="C3",
               label="fast-proxy $R^2$ from that PC (normalised)")
        a.axvline(D_SLOW - 0.5, color="C0", ls="--",
                  label=f"PCA would keep the first {D_SLOW}")
        a.set_title(f"{name}: top-{D_SLOW} PCs hold {o['var_share_top']*100:.0f}% of variance,\n"
                    f"fast $R^2$ {o['fast_r2_top']:.2f} (top) vs {o['fast_r2_tail']:.2f} (tail)",
                    fontsize=10)
        a.set_xlabel("principal component (variance-ranked)"); a.legend(fontsize=7)
    fig.suptitle("Does the fast factor hide in low-variance directions? "
                 "(ungated g0_x0 embedding)", fontsize=11)
    fig.tight_layout()
    fig.savefig("../../runs_v2/fig_fastaxis.png", dpi=140)
    print("\nsaved runs_v2/fastaxis.json, runs_v2/fig_fastaxis.png")


if __name__ == "__main__":
    os.makedirs("../../runs_v2", exist_ok=True)
    main()
