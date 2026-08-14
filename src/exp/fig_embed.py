"""Two qualitative figures for the separation claim, on synthetic gap=0.03.

Both are read off ONE trained model (the main 2x2 cell: nce+ema, gate on, xcov on,
lam=4), so the SEP written to the JSON is the SEP of exactly the model drawn here.
Nothing is hand-transcribed (CLAUDE.md 4-3).

  EMBED  a t-SNE grid of the block x factor matrix, made visible. `--compare` draws
         the paper's figure: two rows (with and without the mechanism) x three
         columns, the columns being exactly the three terms SEP multiplies. Without
         `--compare` it draws the older 2x2 grid for one model, which additionally
         shows z_mix coloured by the persistent factor.

  TRAJ   z_per's leading principal components against time, with the regime
         switches marked. If the block tracks a persistent factor it should sit
         still between switches and move at them.

HONEST READING (do not oversell the grid). z_mix is NOT the mirror image of z_per.
The gate only constrains when z_mix is available to the predictor, not what it
contains, and near-horizon prediction needs the current persistent state as much as
the transient one -- so z_mix carries BOTH factors by design (CLAUDE.md 2). That is
why the paper's figure omits the z_mix/persistent panel: it is not a term of SEP and
showing it invited the reader to look for a failure that was never claimed away.

python3 src/exp/fig_embed.py --compare   -> runs_v2/fig_embed_compare.png  (paper Fig. 3)
python3 src/exp/fig_embed.py            -> runs_v2/fig_embed_<stem>.{png,json}
                                           runs_v2/fig_traj_<stem>.png
                                           runs_v2/model_embedfig_<stem>.pt  (cache)
python3 src/exp/fig_embed.py --stem l1   for the regression stem
"""
import argparse, json, pathlib, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "hglp"))
from train import train, DEV                                          # noqa: E402
from model import D_Z, D_SLOW, P, L                                   # noqa: E402
from datagen import make_dataset, DWELL, U_TAU                        # noqa: E402
from probes import block_factor, sep_index                            # noqa: E402

OUT = HERE.parents[2] / "runs_v2"
GAP, LAM, SEED, STEPS = 0.03, 4.0, 0, 2500
# Same palette as fig_persistence.py so a reader who learned the colours in Fig. 1
# does not have to relearn them here.
C_PERS, C_TRAN, INK, MUTE = "#1a5fb4", "#e8a33d", "#222222", "#9aa0a6"


def get_model(stem, lam, cache, mech=True):
    """Train one cell and cache it; the figures are qualitative but the SEP quoted
    in the caption must come from this exact model. mech=False is the g0_x0 control
    the 2x2 ablation and rotation.py use -- no gate, no xcov, and no per-block
    LayerNorm, since per-block LN privileges the coordinate split on its own."""
    r = train(loss_kind=stem, target_enc="ema", seed=SEED,
              gate=mech, xcov=mech, blocknorm=mech,
              gap=GAP, lam=(lam if mech else 0.0), steps=STEPS, log_every=1000)
    torch.save(r["enc"].state_dict(), cache)
    return r["enc"]


def held_out(enc, seed=99, n_win=400, positions=tuple(range(64, 240, 12))):
    """Held-out windows -> full embeddings, regime label, transient factor.
    Mirrors twosided.synth_feats, including its window-level (not position-level)
    train/test cut: window starts are random, so neighbouring windows overlap and
    a position-level split would leak."""
    d = make_dataset(300_000, seed=seed, gap=GAP)
    x, s, u = d["x"], d["s"], d["u"]
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(x) - L * P - 1, n_win)
    xb = torch.from_numpy(np.stack([x[st:st + L * P].reshape(L, P)
                                    for st in starts])).to(DEV)
    with torch.no_grad():
        Z = torch.cat([enc(xb[i:i + 128]) for i in range(0, len(xb), 128)]).cpu().numpy()
    F_ = np.concatenate([Z[:, a] for a in positions])
    ys = np.concatenate([s[starts + a * P + (P - 1)] for a in positions])
    yu = np.concatenate([u[starts + a * P + (P - 1)] for a in positions])
    wid = np.tile(np.arange(n_win), len(positions))
    cut = np.sort(starts)[n_win // 2]
    tr = np.flatnonzero(starts[wid] + L * P <= cut)
    te = np.flatnonzero(starts[wid] > cut)
    assert not (set(wid[tr]) & set(wid[te])), "window leaked across split"
    return (F_[tr], ys[tr], yu[tr]), (F_[te], ys[te], yu[te])


def stem_of(tag):
    # `nomech_nce` does not start with "nce", so match on the token, not the prefix
    return "HGLP-NCE" if "nce" in tag.split("_") else "HGLP-Reg"


def fig_grid(F, ys, yu, sep, tag, lam, n_show=3000, seed=0):
    """Rows: which block. Columns: which factor the colour encodes."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(F), min(n_show, len(F)), replace=False)
    F, ys, yu = F[idx], ys[idx], yu[idx]
    blocks = [("$z_{per}$  (16 dims)", F[:, :D_SLOW]),
              ("$z_{mix}$  (48 dims)", F[:, D_SLOW:])]
    emb = {name: TSNE(n_components=2, perplexity=30, init="pca",
                      random_state=seed).fit_transform(B) for name, B in blocks}

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.0))
    for i, (name, _) in enumerate(blocks):
        E = emb[name]
        for j, (fac, col) in enumerate([("persistent factor $s$", "s"),
                                        ("transient factor $u$", "u")]):
            ax = axes[i, j]
            if col == "s":
                for k, c in enumerate(["#1a5fb4", "#7d3ac1", "#d64550"]):
                    m = ys == k
                    ax.scatter(E[m, 0], E[m, 1], s=3, c=c, alpha=.55, linewidths=0,
                               label=f"regime {k}")
                if i == 0:
                    ax.legend(fontsize=7.5, frameon=False, markerscale=3,
                              loc="upper right", handletextpad=.1)
            else:
                sc = ax.scatter(E[:, 0], E[:, 1], s=3, c=yu, cmap="cividis",
                                alpha=.65, linewidths=0)
                if i == 0:
                    cb = fig.colorbar(sc, ax=ax, fraction=.045, pad=.02)
                    cb.set_label("$u$", fontsize=8); cb.ax.tick_params(labelsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#cccccc")
            if i == 0:
                ax.set_title(f"coloured by {fac}", fontsize=10, pad=8)
            if j == 0:
                ax.set_ylabel(name, fontsize=11)

    # One line per panel saying what it is evidence for. The reader should not have
    # to hold SEP's definition in their head to read the grid.
    notes = ([("inclusion", MUTE), ("exclusion", MUTE),
              (None, None), ("allocation", MUTE)] if "nomech" in tag else
             [("inclusion", C_PERS), ("exclusion", C_PERS),
              (None, None), ("allocation", C_TRAN)])
    for ax, (note, c) in zip(axes.ravel(), notes):      # not `tag`: that is the filename
        if note:
            ax.text(.02, .02, note, transform=ax.transAxes, fontsize=9,
                    weight="bold", color=c, va="bottom")
        else:
            ax.text(.02, .02,
                    "arbitrary block" if "nomech" in tag else "unconstrained",
                    transform=ax.transAxes, fontsize=9, color=MUTE, va="bottom")
    head = (f"no mechanism: no gate, no xcov, no block LayerNorm  "
            f"(synthetic $\\pm$3%, {stem_of(tag)}, SEP {sep['sep']:.3f})"
            if "nomech" in tag else
            f"t-SNE of each block, coloured by each factor  "
            f"(synthetic $\\pm$3%, {stem_of(tag)}, $\\lambda$={lam:g}, "
            f"SEP {sep['sep']:.3f})")
    fig.suptitle(head, fontsize=11, y=.98)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(OUT / f"fig_embed_{tag}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {name: emb[name].shape for name in emb}


def fig_compare(panels, n_show=3000, seed=0):
    """One figure, two rows x three columns: our model over the mechanism-free one,
    the columns being exactly the three terms SEP multiplies.

    The 2x2 grid of `fig_grid` spent a quarter of its area on z_mix coloured by the
    persistent factor, which is not a SEP term and is unconstrained by design -- the
    panel invited the reader to look for a failure that was never claimed away. Drop
    it and the freed width pays for the mechanism-free row, which is the comparison
    that actually shows what the gate and the penalty do.

    `panels` is [(row label, F, ys, yu, sep), ...] with the gated model first.
    """
    rng = np.random.default_rng(seed)
    COLS = [("$z_{per}$ / persistent $s$", "per", "s", "inclusion", C_PERS),
            ("$z_{per}$ / transient $u$", "per", "u", "exclusion", C_PERS),
            ("$z_{mix}$ / transient $u$", "mix", "u", "allocation", C_TRAN)]

    fig, axes = plt.subplots(len(panels), 3, figsize=(11.4, 7.2))
    for i, (row_label, F, ys, yu, sep) in enumerate(panels):
        idx = rng.choice(len(F), min(n_show, len(F)), replace=False)
        Fi, si, ui = F[idx], ys[idx], yu[idx]
        emb = {b: TSNE(n_components=2, perplexity=30, init="pca",
                       random_state=seed).fit_transform(
                   Fi[:, :D_SLOW] if b == "per" else Fi[:, D_SLOW:])
               for b in ("per", "mix")}
        for j, (title, block, colour, term, c) in enumerate(COLS):
            ax, E = axes[i, j], emb[block]
            if colour == "s":
                for k, cc in enumerate(["#1a5fb4", "#7d3ac1", "#d64550"]):
                    m = si == k
                    ax.scatter(E[m, 0], E[m, 1], s=3, c=cc, alpha=.55, linewidths=0,
                               label=f"regime {k}")
                if i == 0:
                    ax.legend(fontsize=7.5, frameon=False, markerscale=3,
                              loc="upper right", handletextpad=.1)
            else:
                sc = ax.scatter(E[:, 0], E[:, 1], s=3, c=ui, cmap="cividis",
                                alpha=.65, linewidths=0)
                if i == 0 and j == 2:
                    cb = fig.colorbar(sc, ax=ax, fraction=.045, pad=.02)
                    cb.set_label("$u$", fontsize=8); cb.ax.tick_params(labelsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#cccccc")
            if i == 0:
                ax.set_title(title, fontsize=10, pad=8)
            if j == 0:
                ax.set_ylabel(f"{row_label}\nSEP {sep['sep']:.3f}", fontsize=10)
            # The term label is the claim. Print its VALUE too: two of the three
            # columns are near-identical across the rows, and asking a reader to
            # eyeball which scatter is "less structured" would be asking them to
            # see a difference that is only in the middle column.
            ax.text(.02, .02, f"{term}  {sep[term]:.3f}", transform=ax.transAxes,
                    fontsize=9, weight="bold", color=(c if i == 0 else MUTE),
                    va="bottom")
    fig.suptitle("The three terms of SEP, with and without the mechanism "
                 "(synthetic $\\pm$3%, HGLP-NCE)", fontsize=11, y=.98)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(OUT / "fig_embed_compare.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("  runs_v2/fig_embed_compare.png")


def fig_traj(enc, tag, lam, n_samp=90_000, stride_patches=4, seed=7):
    """z_slow's leading PCs against time on one held-out series, with the regime
    switches marked. Windows slide by `stride_patches`; each point is z_slow read
    at the LAST position of its window, so the x axis is the window's right edge."""
    d = make_dataset(n_samp, seed=seed, gap=GAP)
    x, s = d["x"], d["s"]
    step = stride_patches * P
    starts = np.arange(0, len(x) - L * P - 1, step)
    xb = torch.from_numpy(np.stack([x[st:st + L * P].reshape(L, P)
                                    for st in starts])).to(DEV)
    with torch.no_grad():
        Z = torch.cat([enc(xb[i:i + 128]) for i in range(0, len(xb), 128)]).cpu().numpy()
    zs = Z[:, -1, :D_SLOW]                       # last position, persistent block
    right = starts + L * P - 1                   # sample index of that position
    pcs = PCA(n_components=2, random_state=0).fit_transform(zs)
    reg = s[right]
    switch = right[1:][reg[1:] != reg[:-1]]

    # One row per component. Overlaying PC1 and PC2 hid both: they step at the same
    # instants, so the traces cross constantly and neither staircase is readable.
    var = PCA(n_components=2).fit(zs).explained_variance_ratio_
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 5.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[.8, 1.6, 1.6], hspace=.14))
    axes[0].step(right, reg, where="post", lw=1.6, color=MUTE)
    axes[0].set_yticks([0, 1, 2]); axes[0].set_ylabel("regime\n($s$)", fontsize=9)
    for k, (lab, col) in enumerate([("PC1", C_PERS), ("PC2", "#7d3ac1")]):
        ax = axes[k + 1]
        ax.plot(right, pcs[:, k], lw=1.0, color=col)
        ax.set_ylabel(f"$z_{{per}}$ {lab}\n({var[k]*100:.0f}% of variance)",
                      fontsize=9, color=col)
    axes[2].set_xlabel("time (samples); each point is the window's last position",
                       fontsize=9)
    for ax in axes:
        for sw in switch:
            ax.axvline(sw, color="#d64550", lw=.9, ls=":", alpha=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.suptitle(f"$z_{{per}}$ moves at regime switches and sits still between them "
                 f"(synthetic $\\pm$3%, {tag}, $\\lambda$={lam:g}; dotted lines are switches)",
                 fontsize=10.5, y=.99)
    fig.savefig(OUT / f"fig_traj_{tag}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return dict(n_windows=int(len(starts)), stride_patches=stride_patches,
                n_switches=int(len(switch)), dwell=int(DWELL), u_tau=int(U_TAU),
                pc_var=[float(v) for v in
                        PCA(n_components=2).fit(zs).explained_variance_ratio_])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="nce", choices=["nce", "l1"])
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--nomech", action="store_true",
                    help="the g0_x0 control: no gate, no xcov, no block LayerNorm")
    ap.add_argument("--compare", action="store_true",
                    help="one 2x3 figure: gated over mechanism-free, three SEP terms. "
                         "Reuses both cached encoders; trains neither if both exist.")
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()

    if a.compare:
        rows = []
        for nomech, label in ((False, "gate + xcov"), (True, "no mechanism")):
            tg = f"nomech_{a.stem}" if nomech else f"{a.stem}_lam{a.lam:g}"
            ck = OUT / f"model_embedfig_{tg}.pt"
            if ck.exists() and not a.retrain:
                from model import Encoder
                e = Encoder(blocknorm=not nomech).to(DEV)
                e.load_state_dict(torch.load(ck, map_location=DEV)); e.eval()
                print(f"cached encoder: {ck.name}")
            else:
                e = get_model(a.stem, a.lam, ck, mech=not nomech)
            (Ftr, ytr, utr), (Fte, yte, ute) = held_out(e)
            sp = sep_index(block_factor(Ftr, ytr, utr, Fte, yte, ute, seed=0))
            rows.append((label, Fte, yte, ute, sp))
            print(f"  {label}: SEP {sp['sep']:.3f}")
        fig_compare(rows)
        return
    OUT.mkdir(exist_ok=True)
    tag = f"nomech_{a.stem}" if a.nomech else f"{a.stem}_lam{a.lam:g}"
    cache = OUT / f"model_embedfig_{tag}.pt"

    if cache.exists() and not a.retrain:
        from model import Encoder
        enc = Encoder(blocknorm=not a.nomech).to(DEV)
        enc.load_state_dict(torch.load(cache, map_location=DEV))
        enc.eval()
        print(f"cached encoder: {cache.name}")
    else:
        what = "NO mechanism" if a.nomech else f"gate+xcov, lam={a.lam}"
        print(f"training {a.stem}+ema, {what}, gap={GAP} ...")
        enc = get_model(a.stem, a.lam, cache, mech=not a.nomech)

    (Ftr, ytr, utr), (Fte, yte, ute) = held_out(enc)
    bf = block_factor(Ftr, ytr, utr, Fte, yte, ute, seed=0)
    sep = sep_index(bf)
    print(f"SEP {sep['sep']:.3f}  incl {sep['inclusion']:.3f}  "
          f"alloc {sep['allocation']:.3f}  excl {sep['exclusion']:.3f}")

    shapes = fig_grid(Fte, yte, ute, sep, tag, a.lam)
    traj = fig_traj(enc, tag, a.lam)

    res = dict(config=dict(stem=a.stem, target="ema", gate=not a.nomech,
                           xcov=not a.nomech, blocknorm=not a.nomech,
                           gap=GAP, lam=(0.0 if a.nomech else a.lam),
                           seed=SEED, steps=STEPS,
                           d_slow=D_SLOW, d_z=D_Z),
               sep=sep, block_factor=bf, tsne=dict(perplexity=30, init="pca",
                                                   n_points=3000),
               trajectory=traj)
    (OUT / f"fig_embed_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote fig_embed_{tag}.png  fig_traj_{tag}.png  fig_embed_{tag}.json")


if __name__ == "__main__":
    main()
