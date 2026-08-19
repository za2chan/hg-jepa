"""B5 pilot (protocol E4). Trains HGLP-Reg with the v1 cumulative target and the
v2 RF-bounded target, then judges B5 on two axes:

  (1) STABILITY   — loss convergence, target RankMe, probe slow-kept / leak.
  (2) HARMLESSNESS(Δ) = R2(s,u -> z_bar(t+Δ)) - R2(s -> z_bar(t+Δ)).
      Prediction: cumulative stays > 0 at large Δ (target re-contains the
      anchor's past); bounded decays to ~0.

Writes runs_v2/pilot_b5.json + runs_v2/pilot_b5.png. Run from src/.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hglp"))

import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression

from datagen import make_dataset
from model import P, L, D_Z, D_SLOW
from train import train, rankme, DEV

DELTAS = [8, 12, 16, 24, 32, 48, 64, 96, 128]
W_SLICE = 8


def eval_windows(seed=99, n_win=400):
    d = make_dataset(300_000, seed=seed)
    x, s, u = d["x"], d["s"], d["u"]
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(x) - L * P - 1, n_win)
    xb = np.stack([x[st:st + L * P].reshape(L, P) for st in starts])
    return torch.from_numpy(xb).to(DEV), starts, s, u


def factor_at(starts, s, u, patch):
    """Ground-truth (regime, u) at the END sample of `patch` for each window."""
    idx = starts + patch * P + (P - 1)
    return s[idx], u[idx]


@torch.no_grad()
def harmlessness(tgt, mode, xb, starts, s, u, anchors):
    """Per Δ: R2(s,u->z_bar) - R2(s->z_bar), anchor factors vs target embedding."""
    n = len(xb)
    Zfull = None
    if mode == "cumulative":
        Zfull = torch.cat([tgt(xb[i:i + 128]) for i in range(0, n, 128)])  # (n,L,D_Z)
    curve = {}
    for D in DELTAS:
        tgt_pos = anchors + D                                  # (n_anchor,) per window
        # build (window, anchor) grid
        wi = np.repeat(np.arange(n), len(anchors))
        ap = np.tile(anchors, n)
        tpos = ap + D
        if mode == "cumulative":
            Z = Zfull[torch.from_numpy(wi).to(DEV), torch.from_numpy(tpos).to(DEV)].cpu().numpy()
        else:
            ar = np.arange(W_SLICE)
            pos = (tpos - W_SLICE)[:, None] + ar               # (M, w)
            sl = xb[torch.from_numpy(wi).to(DEV)[:, None], torch.from_numpy(pos).to(DEV)]
            Z = tgt(sl)[:, -1].cpu().numpy()
        sa, ua = factor_at(starts[wi], s, u, ap)               # anchor factors
        Xs = np.eye(3)[sa]                                     # slow only
        Xsu = np.concatenate([Xs, ua[:, None]], 1)             # slow + fast
        m = len(Z) // 2
        r2 = lambda X: Ridge().fit(X[:m], Z[:m]).score(X[m:], Z[m:])
        curve[D] = float(r2(Xsu) - r2(Xs))
    return curve


@torch.no_grad()
def probe(enc, xb, starts, s, u, positions):
    """slow-kept (regime acc) and leak (u R2) from z_slow, per-position labels."""
    Z = torch.cat([enc(xb[i:i + 128]) for i in range(0, len(xb), 128)])
    Zs = Z[:, :, :D_SLOW].cpu().numpy()                        # (n, L, D_SLOW)
    feats, reg, uu = [], [], []
    for a in positions:
        feats.append(Zs[:, a]); sa, ua = factor_at(starts, s, u, a)
        reg.append(sa); uu.append(ua)
    F_ = np.concatenate(feats); reg = np.concatenate(reg); uu = np.concatenate(uu)
    m = len(F_) // 2
    acc = LogisticRegression(max_iter=1000).fit(F_[:m], reg[:m]).score(F_[m:], reg[m:])
    leak = Ridge().fit(F_[:m], uu[:m]).score(F_[m:], uu[m:])
    rm = rankme(F_)
    return dict(slow_kept=float(acc), leak_u=float(leak), rankme=float(rm))


def main():
    os.makedirs("../../runs_v2", exist_ok=True)
    xb, starts, s, u = eval_windows()
    anchors = np.arange(16, 120, 12)                           # a + Δ_max < L
    probe_pos = list(range(16, 240, 16))
    out = {}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for mode, col in [("cumulative", "#e34948"), ("bounded", "#2a78d6")]:
        r = train(target_mode=mode, seed=0)
        st = probe(r["enc"], xb, starts, s, u, probe_pos)
        hc = harmlessness(r["tgt"], mode, xb, starts, s, u, anchors)
        losses = r["losses"]
        st["loss_final"] = float(np.mean(losses[-100:]))
        st["loss_slope_last500"] = float(np.polyfit(range(500), losses[-500:], 1)[0])
        out[mode] = dict(stability=st, harmlessness=hc)
        ax[0].plot(losses, color=col, lw=0.8, label=mode)
        ax[1].plot(DELTAS, [hc[D] for D in DELTAS], "o-", color=col, label=mode)
        print(f"\n=== {mode} ===")
        print("  stability:", {k: round(v, 3) for k, v in st.items()})
        print("  harmlessness(Δ):", {D: round(hc[D], 3) for D in DELTAS})
    ax[0].set_title("loss"); ax[0].set_xlabel("step"); ax[0].legend(fontsize=8)
    ax[1].axhline(0, color="k", lw=0.5, ls=":"); ax[1].set_xscale("log")
    ax[1].set_title("harmlessness(Δ) = R²(s,u) − R²(s)\nlower→0 = fast info useless at long Δ")
    ax[1].set_xlabel("Δ (patches, log)"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("../../runs_v2/pilot_b5.png", dpi=120); plt.close()
    json.dump(out, open("../../runs_v2/pilot_b5.json", "w"), indent=2)

    # ---- verdict ----
    cb, cc = out["bounded"], out["cumulative"]
    far = [96, 128]
    b_far = np.mean([cb["harmlessness"][D] for D in far])
    c_far = np.mean([cc["harmlessness"][D] for D in far])
    stable = (cb["stability"]["rankme"] > 3 and cb["stability"]["slow_kept"] > 0.5
              and cb["stability"]["loss_slope_last500"] < 1e-3)
    print("\n=== VERDICT ===")
    print(f"  bounded stable: {stable} (RankMe {cb['stability']['rankme']:.1f}, "
          f"slow-kept {cb['stability']['slow_kept']:.2f}, "
          f"loss slope {cb['stability']['loss_slope_last500']:.1e})")
    print(f"  harmlessness at far Δ: bounded {b_far:.3f} vs cumulative {c_far:.3f} "
          f"(want bounded << cumulative, bounded→0)")
    print("  saved runs_v2/pilot_b5.{json,png}")


if __name__ == "__main__":
    main()
