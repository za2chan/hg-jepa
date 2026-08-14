"""The z_slow trajectory on real data (HAPT), for both stems.

The synthetic version of this figure has ground-truth regimes and a generator we
wrote. HAPT does not: the persistent factor is the activity a person is actually
doing, the transitions are where they actually changed activity, and nothing here
was designed to make the block's job easy.

Reconstructing a continuous trajectory. The v2 windows are 256 patches with a
128-patch stride, i.e. 50% overlap (verified: window i's last 128 patches equal
window i+1's first 128 for 98.6% of consecutive pairs within a subject). Taking
positions [128:256] of every window therefore tiles the recording exactly once,
and every embedding kept this way has at least 128 patches of causal context --
twice the probe's context floor (probes.C_MIN = 64).

Labels. HAPT codes 1-6 as the six activities, 7-12 as postural transitions and 0
as unlabelled. The probe scores 1-6 only, so the figure shades everything else
rather than pretending the block should track it.

The subject is drawn from the HELD-OUT group split, so the encoder never saw it.

python3 src/exp/fig_traj_hapt.py -> runs_v2/fig_traj_hapt.{png,json}
"""
import json, pathlib, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "hglp"))
from train_real import train_real                                     # noqa: E402
from train import DEV                                                 # noqa: E402
from model import Encoder, D_SLOW                                     # noqa: E402

OUT = HERE.parents[2] / "runs_v2"
NPZ = HERE.parents[2] / "data" / "hapt_v2.npz"
STRIDE, HZ, PATCH = 128, 50.0, 4          # patches, Hz, samples per patch
ACTS = {1: "walking", 2: "walking upstairs", 3: "walking downstairs",
        4: "sitting", 5: "standing", 6: "lying"}
COL = {"nce": "#1a5fb4", "l1": "#7d3ac1"}
MUTE = "#9aa0a6"


def fit(stem, seed=0):
    """train_real also returns the normalised tensors and the group split, so the
    run dict is rebuilt even when the weights are cached; only the optimisation is
    skipped."""
    cache = OUT / f"model_hapt_{stem}.pt"
    r = train_real(str(NPZ), seed=seed, loss_kind=stem, target_enc="ema",
                   gate=True, xcov=True, lam=4.0, log_every=1000,
                   steps=(0 if cache.exists() else 2500))
    if cache.exists():
        r["enc"].load_state_dict(torch.load(cache, map_location=DEV))
        r["enc"].eval()
        print(f"  cached: {cache.name}")
    else:
        torch.save(r["enc"].state_dict(), cache)
    return r


def tile(r, subj_id, subj):
    """Held-out subject -> (z at every tiled position, label, time in seconds)."""
    idx = np.flatnonzero((subj == subj_id) & r["is_test"])
    idx.sort()
    with torch.no_grad():
        Z = torch.cat([r["enc"](r["Wt"][idx][i:i + 64])
                       for i in range(0, len(idx), 64)]).cpu().numpy()
    z = Z[:, STRIDE:].reshape(-1, Z.shape[-1])            # new content only
    lab = r["lab"][idx][:, STRIDE:].reshape(-1)
    t = np.arange(len(z)) * PATCH / HZ                    # seconds
    return z, lab, t


def readout(f, y):
    """Activity macro-F1 from a feature block, on labelled positions only, split
    in time so the halves do not share windows."""
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = (y >= 1) & (y <= 6)
    f, y = f[m].reshape(m.sum(), -1), y[m]
    n = len(y) // 2
    if len(np.unique(y[:n])) < 2:
        return float("nan")
    p = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
    return float(f1_score(y[n:], p.fit(f[:n], y[:n]).predict(f[n:]),
                          average="macro", labels=np.unique(y[:n]), zero_division=0))


def main():
    subj = np.load(NPZ)["subj"]
    runs, res = {}, {}
    for stem in ("nce", "l1"):
        print(f"training HAPT {stem}+ema, gate+xcov ...", flush=True)
        runs[stem] = fit(stem)

    # A held-out subject with plenty of activity changes, chosen once and shared
    # by both stems so the two panels are the same stretch of time.
    r0 = runs["nce"]
    cand = [s for s in np.unique(subj[r0["is_test"]])]
    best = max(cand, key=lambda s: (np.diff(tile(r0, s, subj)[1]) != 0).sum())
    print(f"held-out subjects {cand}; drawing subject {best}")

    # Five rows: the activity strip, then PC1 and PC2 for each stem. One component
    # was not enough here -- six activities do not fit on a line, and PC1 carries
    # only about a third of z_slow's variance (against 94% on synthetic).
    fig, axes = plt.subplots(5, 1, figsize=(10.0, 9.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[.8, 1.2, 1.2, 1.2, 1.2],
                                              hspace=.18))
    for k, stem in enumerate(("nce", "l1")):
        z, lab, t = tile(runs[stem], best, subj)
        pca = PCA(n_components=2, random_state=0).fit(z[:, :D_SLOW])
        pcs = pca.transform(z[:, :D_SLOW])
        var = pca.explained_variance_ratio_
        f1 = readout(z[:, :D_SLOW], lab)
        f1_2d = readout(pcs, lab)
        res[stem] = dict(macro_f1_from_z_slow=f1, macro_f1_from_pc1_pc2=f1_2d,
                         pc1_var=float(var[0]), pc2_var=float(var[1]))
        name = "HGLP-NCE" if stem == "nce" else "HGLP-Reg"
        for j in (0, 1):
            ax = axes[1 + 2 * k + j]
            ax.plot(t, (pcs[:, j] - pcs[:, j].mean()) / pcs[:, j].std(), lw=1.0,
                    color=COL[stem], alpha=1.0 if j == 0 else .75)
            ax.set_ylabel(f"{name}\nPC{j+1} ({var[j]*100:.0f}% var)",
                          fontsize=9, color=COL[stem])
            if j == 1:
                ax.text(.994, .05,
                        f"activity macro-F1 — from PC1+PC2 {f1_2d:.2f}, "
                        f"from all 16 dims {f1:.2f}",
                        transform=ax.transAxes, fontsize=8.5, ha="right",
                        va="bottom", color=COL[stem],
                        bbox=dict(boxstyle="round,pad=.25", fc="white", ec="none",
                                  alpha=.85))

        if k == 0:                                        # activity strip, drawn once
            a0 = axes[0]
            a0.plot(t, np.where((lab >= 1) & (lab <= 6), lab, np.nan), lw=2.2,
                    color=MUTE, solid_capstyle="butt")
            unl = ~((lab >= 1) & (lab <= 6))
            a0.fill_between(t, 0.5, 6.5, where=unl, color="#e9ecef", lw=0)
            a0.set_yticks(range(1, 7))
            a0.set_yticklabels([ACTS[i] for i in range(1, 7)], fontsize=7.5)
            a0.set_ylim(.5, 6.5)
            a0.set_ylabel("activity", fontsize=9.5)
            switch = t[1:][(lab[1:] != lab[:-1])]
            res["n_switches"] = int(len(switch))
            res["duration_s"] = float(t[-1])
            res["unlabelled_frac"] = float(unl.mean())

    for ax in axes:
        for sw in switch:
            ax.axvline(sw, color="#d64550", lw=.8, ls=":", alpha=.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    for ax in axes[1:]:
        ax.fill_between(t, *ax.get_ylim(), where=unl, color="#f2f2f2", lw=0, zorder=0)
    axes[-1].set_xlabel("time within the recording (seconds)", fontsize=9.5)
    axes[0].legend(handles=[Patch(fc="#e9ecef", ec="none",
                                  label="unlabelled or postural transition")],
                   fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(f"$z_{{slow}}$ on held-out HAPT subject {best}: it steps where the "
                 f"activity changes (dotted lines)", fontsize=10.5, y=.995)
    fig.savefig(OUT / "fig_traj_hapt.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    res["config"] = dict(dataset="HAPT v2", subject=int(best), stems=["nce", "l1"],
                         target="ema", gate=True, xcov=True, lam=4.0, tau=40.0,
                         seed=0, stride_patches=STRIDE, hz=HZ, patch=PATCH,
                         note="subject is in the held-out group split")
    (OUT / "fig_traj_hapt.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "config"},
                     indent=1, default=float))
    print("wrote fig_traj_hapt.png  fig_traj_hapt.json")


if __name__ == "__main__":
    main()
