"""Does the post-hoc baseline produce the same staircase? Yes -- but not where
its own ordering points.

An earlier version of this script drew only SFA's SLOWEST direction, found noise,
and concluded the staircase was the gate's doing. That was wrong, and the mistake
is worth keeping in the record: the regime staircase sits in a middle-ranked SFA
component, where it is at least as clean as ours. Over the full 16-dim subspace
the two are level.

What survives is narrower and matches the framing the handoff already fixed. Our
staircase is in the FIRST principal component of a block named before training.
SFA's is in a component its own label-free ordering does not point to, so picking
it out needs the labels we are trying to do without. `--seeds` checks that this
is a property of the method and not an accident of one run: if the component that
holds the regime moved around, no fixed rule ("always take the third") can stand
in for the labels. That is an operational difference, not a separation-quality
one, and the figure must not be captioned as if it were.

Design, deliberately generous to SFA:
  * the ungated control is the same g0_x0 cell the 2x2 ablation and rotation.py
    use (no gate, no xcov, no block LayerNorm), same data, same steps;
  * SFA is fitted on the SAME trajectory it is then drawn on -- rotation.py fits
    on a train split only, so if SFA falls short here it is not for want of data;
  * time-adjacent pairs are consecutive sliding windows.

Scores. `accuracy` is a one-dimensional logistic readout of the regime from a
single component, fitted on the first half of the series and tested on the second
(chance 1/3). `eta2` is the share of that component's variance regime explains,
which assumes nothing about the regimes lying in a monotone order along the axis.

python3 src/exp/fig_traj_sfa.py            -> runs_v2/fig_traj_sfa.{png,json}
python3 src/exp/fig_traj_sfa.py --seeds 5  -> runs_v2/fig_sfa_seeds.{png,json}
"""
import argparse, json, pathlib, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "hglp"))
sys.path.insert(0, str(HERE.parent))
from train import train, DEV                                          # noqa: E402
from model import Encoder, D_SLOW, P, L                               # noqa: E402
from datagen import make_dataset                                      # noqa: E402
from rotation import sfa_basis                                        # noqa: E402

OUT = HERE.parents[2] / "runs_v2"
GAP, STEPS = 0.03, 2500
C_OURS, C_SFA1, C_SFA2, MUTE = "#1a5fb4", "#c0392b", "#7d3ac1", "#9aa0a6"


def encoder(cache, seed, **kw):
    if cache.exists():
        enc = Encoder().to(DEV)
        enc.load_state_dict(torch.load(cache, map_location=DEV))
        enc.eval()
        return enc
    print(f"  training {cache.stem} ...")
    r = train(seed=seed, gap=GAP, steps=STEPS, log_every=100000, **kw)
    torch.save(r["enc"].state_dict(), cache)
    return r["enc"]


def pair(seed):
    """The gated model and its ungated control, for one training seed."""
    g = encoder(OUT / f"model_embedfig_nce_lam4.pt" if seed == 0 else
                OUT / f"model_sfaseed_gated_s{seed}.pt", seed,
                loss_kind="nce", target_enc="ema", gate=True, xcov=True, lam=4.0)
    u = encoder(OUT / "model_embedfig_ungated.pt" if seed == 0 else
                OUT / f"model_sfaseed_ungated_s{seed}.pt", seed,
                loss_kind="nce", target_enc="ema", gate=False, xcov=False,
                blocknorm=False, lam=0.0)
    return g, u


def trajectory(enc, n_samp=90_000, stride_patches=4, seed=7):
    """Sliding windows over one held-out series -> (embeddings, regime, time)."""
    d = make_dataset(n_samp, seed=seed, gap=GAP)
    x, s = d["x"], d["s"]
    starts = np.arange(0, len(x) - L * P - 1, stride_patches * P)
    xb = torch.from_numpy(np.stack([x[st:st + L * P].reshape(L, P)
                                    for st in starts])).to(DEV)
    with torch.no_grad():
        Z = torch.cat([enc(xb[i:i + 128]) for i in range(0, len(xb), 128)]).cpu().numpy()
    return Z[:, -1], s[starts + L * P - 1], starts + L * P - 1


def eta2(v, reg):
    g = [v[reg == k] for k in np.unique(reg)]
    ss_b = sum(len(x) * (x.mean() - v.mean()) ** 2 for x in g)
    return float(ss_b / ((v - v.mean()) ** 2).sum())


def readout(X, reg):
    n = len(X) // 2
    X = X.reshape(len(X), -1)
    return float(LogisticRegression(max_iter=3000)
                 .fit(X[:n], reg[:n]).score(X[n:], reg[n:]))


def analyse(gated, ungated):
    """One seed -> our PC1, the SFA components, and where the regime landed."""
    Zg, reg, t = trajectory(gated)
    Zu, reg_u, _ = trajectory(ungated)
    assert np.array_equal(reg, reg_u), "the two runs must see the same series"
    ours = PCA(n_components=1, random_state=0).fit_transform(Zg[:, :D_SLOW])[:, 0]
    pr = np.stack([np.arange(len(Zu) - 1), np.arange(1, len(Zu))], 1)
    S = (Zu - Zu.mean(0)) @ sfa_basis(Zu, pr)
    e = np.array([eta2(S[:, i], reg) for i in range(D_SLOW)])
    return dict(ours=ours, S=S, reg=reg, t=t, eta=e, best=int(e.argmax()),
                acc_ours=readout(ours, reg), eta_ours=eta2(ours, reg),
                sub_ours=readout(Zg[:, :D_SLOW], reg),
                sub_sfa=readout(S[:, :D_SLOW], reg))


def panel(ax, t, v, switch, colour, tag, note, acc):
    ax.plot(t, (v - v.mean()) / v.std(), lw=1.0, color=colour)
    ax.set_ylabel(tag, fontsize=10, color=colour, labelpad=8)
    # Descriptive line and score go INSIDE the axes: two-line y-labels collided
    # with each other in the left margin and were unreadable.
    ax.text(.006, .95, note, transform=ax.transAxes, fontsize=8.5, va="top",
            color=colour,
            bbox=dict(boxstyle="round,pad=.25", fc="white", ec="none", alpha=.85))
    ax.text(.994, .06, f"regime accuracy {acc:.2f}", transform=ax.transAxes,
            fontsize=8.5, ha="right", va="bottom", color=colour,
            bbox=dict(boxstyle="round,pad=.25", fc="white", ec="none", alpha=.85))
    for sw in switch:
        ax.axvline(sw, color="#d64550", lw=.8, ls=":", alpha=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.margins(y=.18)


def fig_single(a):
    r = analyse(*pair(0))
    reg, t, S, best = r["reg"], r["t"], r["S"], r["best"]
    switch = t[1:][reg[1:] != reg[:-1]]
    print(f"seed 0 — ours {r['acc_ours']:.3f}  SFA#1 {readout(S[:, 0], reg):.3f}  "
          f"SFA#{best+1} {readout(S[:, best], reg):.3f}   "
          f"16-dim: ours {r['sub_ours']:.3f} SFA {r['sub_sfa']:.3f}")

    fig, ax = plt.subplots(4, 1, figsize=(9.6, 7.2), sharex=True,
                           gridspec_kw=dict(height_ratios=[.55, 1.5, 1.5, 1.5],
                                            hspace=.18))
    ax[0].step(t, reg, where="post", lw=1.6, color=MUTE)
    ax[0].set_yticks([0, 1, 2]); ax[0].set_ylabel("regime", fontsize=10, labelpad=8)
    ax[0].tick_params(labelsize=8); ax[0].spines[["top", "right"]].set_visible(False)
    for sw in switch:
        ax[0].axvline(sw, color="#d64550", lw=.8, ls=":", alpha=.6)
    panel(ax[1], t, r["ours"], switch, C_OURS, "ours",
          "PC1 of $z_{per}$, a block named before training", r["acc_ours"])
    panel(ax[2], t, S[:, 0], switch, C_SFA1, "SFA #1",
          "the component SFA's own ordering points to", readout(S[:, 0], reg))
    panel(ax[3], t, S[:, best], switch, C_SFA2, f"SFA #{best+1}",
          "the component the regime is actually in", readout(S[:, best], reg))
    ax[3].set_xlabel("time (samples); each point is a window's last position",
                     fontsize=9)
    fig.suptitle("SFA recovers the regime too, but not in the component its own "
                 "ordering points to (synthetic $\\pm$3%, one seed)",
                 fontsize=10.5, y=.995)
    fig.savefig(OUT / "fig_traj_sfa.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    (OUT / "fig_traj_sfa.json").write_text(json.dumps(dict(
        config=dict(gap=GAP, seed=0, steps=STEPS, stem="nce", lam_gated=4.0,
                    sfa_fitted_on="the plotted series itself (generous to SFA)"),
        accuracy=dict(ours=r["acc_ours"], sfa_slowest=readout(S[:, 0], reg),
                      sfa_best=readout(S[:, best], reg), best_component=best + 1,
                      chance=1 / 3),
        subspace_16d=dict(ours=r["sub_ours"], sfa=r["sub_sfa"]),
        sfa_eta2_by_component=[float(v) for v in r["eta"]]), indent=1))
    print("wrote fig_traj_sfa.png  fig_traj_sfa.json")


def fig_seeds(n):
    """Which SFA component holds the regime, across training seeds. If the answer
    moves, no fixed rule can replace the labels."""
    rows, curves = [], []
    for s in range(n):
        print(f"seed {s}")
        r = analyse(*pair(s))
        rows.append(dict(seed=s, best_component=r["best"] + 1,
                         eta2_best=float(r["eta"][r["best"]]),
                         eta2_slowest=float(r["eta"][0]),
                         acc_ours=r["acc_ours"], eta2_ours=r["eta_ours"],
                         acc_sub_ours=r["sub_ours"], acc_sub_sfa=r["sub_sfa"]))
        curves.append(r["eta"])
        print(f"  regime lands in SFA #{r['best']+1} (eta2 {r['eta'][r['best']]:.3f}); "
              f"slowest {r['eta'][0]:.3f}; ours PC1 eta2 {r['eta_ours']:.3f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = np.arange(1, D_SLOW + 1)
    for s, e in enumerate(curves):
        ax.plot(xs, e, lw=1.2, alpha=.75, marker="o", ms=3.5,
                label=f"seed {s}")
        ax.plot(int(e.argmax()) + 1, e.max(), marker="*", ms=13, color="#d64550",
                zorder=5, lw=0)
    mean_ours = float(np.mean([r["eta2_ours"] for r in rows]))
    ax.axhline(mean_ours, color=C_OURS, lw=1.6, ls="--")
    ax.text(D_SLOW, mean_ours, f" ours: PC1 of $z_{{slow}}$ ({mean_ours:.2f})",
            color=C_OURS, fontsize=9, va="bottom", ha="right")
    ax.set_xlabel("SFA component, ordered slowest first", fontsize=9.5)
    ax.set_ylabel("$\\eta^2$: share of the component's\nvariance regime explains",
                  fontsize=9.5)
    ax.set_xticks(xs[::2]); ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("The regime is not in SFA's slowest component, and which one it is\n"
                 "moves with the seed (stars mark the peak)", fontsize=10, pad=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sfa_seeds.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    best = [r["best_component"] for r in rows]
    (OUT / "fig_sfa_seeds.json").write_text(json.dumps(dict(
        config=dict(gap=GAP, steps=STEPS, stem="nce", lam_gated=4.0, n_seeds=n),
        per_seed=rows, best_components=best,
        best_component_distinct=sorted(set(best)),
        note="ours is always PC1 of a block fixed before training; SFA's index "
             "varies, and its own slowness ordering does not point to it"),
        indent=1))
    print(f"\nregime landed in SFA components {best} across {n} seeds")
    print("wrote fig_sfa_seeds.png  fig_sfa_seeds.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    fig_seeds(a.seeds) if a.seeds else fig_single(a)
