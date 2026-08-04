"""Round-2 explanatory figures for the PROTOCOL Q&A.

explain_targets.png  : receptive field of the TARGET in CPC / HEPA / ours(v1) /
                       proposed. Shows that only ours overlaps the anchor's past.
explain_blocknorm.png: where LayerNorm sits in v1 vs per-block vs pre-split.
English labels (no Korean font available); prose explains in Korean.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mp

BLUE, ORANGE, RED, GREEN, GREY = "#2a78d6", "#e9963a", "#e34948", "#2a9d8f", "#9aa0a6"


def fig_targets():
    L, t, dlt = 100.0, 40.0, 35.0          # window, anchor, horizon
    tgt = t + dlt
    rows = [
        ("Ours (v1)", (0, tgt), True,
         "target = causal enc at t+D  ->  RF = [0, t+D]  CONTAINS the anchor's past"),
        ("CPC (2018)", (tgt - 4, tgt), False,
         "target = z_{t+k}, LOCAL encoder (small RF); context c_t only on input side"),
        ("HEPA", (t, tgt), False,
         "target = bidirectional summary of (t, t+D] with attention pooling"),
        ("Proposed (RF-bounded point)", (tgt - 8, tgt), False,
         "target = enc at t+D but fed only x[t+D-w : t+D]  ->  RF excludes [0,t]"),
    ]
    fig, ax = plt.subplots(figsize=(14, 6.2))
    for i, (name, (a, b), overlap, note) in enumerate(rows):
        y = len(rows) - i
        # window baseline
        ax.plot([0, L], [y, y], color="0.85", lw=9, solid_capstyle="butt", zorder=1)
        # anchor receptive field (always [0, t])
        ax.add_patch(mp.Rectangle((0, y - 0.17), t, 0.34, color=BLUE, alpha=0.35, zorder=2))
        # target receptive field
        ax.add_patch(mp.Rectangle((a, y - 0.17), b - a, 0.34, color=ORANGE,
                                  alpha=0.75, zorder=3))
        if overlap:                                     # the problematic region
            ax.add_patch(mp.Rectangle((0, y - 0.17), t, 0.34, facecolor="none",
                                      edgecolor=RED, hatch="////", lw=2, zorder=4))
        ax.plot([t], [y], marker="o", color=BLUE, ms=9, zorder=5)
        ax.plot([tgt], [y], marker="*", color=ORANGE, ms=15, mec="k", mew=.6, zorder=5)
        ax.text(-3, y, name, ha="right", va="center", fontsize=10, weight="bold")
        ax.text(L + 2, y, note, ha="left", va="center", fontsize=8.2)
    ax.axvline(t, color=BLUE, ls=":", lw=1)
    ax.axvline(tgt, color=ORANGE, ls=":", lw=1)
    ax.text(t, 4.62, "anchor t", color=BLUE, ha="center", fontsize=9)
    ax.text(tgt, 4.62, "t + D", color=ORANGE, ha="center", fontsize=9)
    handles = [mp.Patch(color=BLUE, alpha=.35, label="anchor receptive field  [0, t]"),
               mp.Patch(color=ORANGE, alpha=.75, label="TARGET receptive field"),
               mp.Patch(facecolor="none", edgecolor=RED, hatch="////",
                        label="overlap: target re-contains the anchor's past")]
    ax.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5,
              bbox_to_anchor=(0.5, -0.16))
    ax.set_xlim(-26, L + 78); ax.set_ylim(0.3, 4.9); ax.set_yticks([])
    ax.set_xlabel("time within the window")
    ax.set_title("What the prediction TARGET is allowed to see\n"
                 "CPC and HEPA both exclude the anchor's past by construction; v1 does not",
                 fontsize=11)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig("docs/eda/figs/explain_targets.png", dpi=115)
    plt.close()


def fig_blocknorm():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    titles = ["v1 (current): LN over all 64  -> COUPLED",
              "(i) per-block LN  -> decoupled",
              "(iii) LN before split  -> no coupling site"]
    for k, a in enumerate(ax):
        a.set_xlim(0, 10); a.set_ylim(0, 10); a.axis("off")
        a.set_title(titles[k], fontsize=10)

        def box(x, y, w, h, label, color, fs=8.5):
            a.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                          fc=color, ec="0.3", lw=1.1))
            a.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs)

        def arrow(x, y0, y1):
            a.annotate("", xy=(x, y1), xytext=(x, y0),
                       arrowprops=dict(arrowstyle="->", lw=1.3, color="0.35"))

        box(2.5, 8.3, 5, 1.0, "transformer trunk  h  (D_MODEL = 96)", "#dbe7f6")
        arrow(5, 8.25, 7.5)
        if k == 2:
            box(2.5, 6.5, 5, 1.0, "LayerNorm  (before split)", "#ffe6a8")
            arrow(5, 6.45, 5.8)
            box(2.5, 4.9, 5, 0.9, "Linear 96 -> 64   =>  z", "#dbe7f6")
            arrow(5, 4.85, 4.2)
        else:
            box(2.5, 6.6, 5, 0.9, "Linear 96 -> 64   =>  z (D_Z = 64)", "#dbe7f6")
            arrow(5, 6.55, 5.9)
        if k == 0:
            box(2.5, 5.0, 5, 0.9, "LayerNorm over ALL 64 dims", "#f6c6c6")
            a.text(5, 4.55, "one mean/std shared by both blocks", ha="center",
                   fontsize=7.6, color=RED)
            arrow(5, 4.3, 3.6)
        y0 = 2.4 if k == 0 else (2.6 if k == 1 else 2.6)
        box(1.2, y0, 3.2, 1.0, "z_slow  (16 dims)", "#cfe8dd")
        box(5.6, y0, 3.2, 1.0, "z_fast  (48 dims)", "#cfe8dd")
        if k == 1:
            a.annotate("", xy=(2.8, y0), xytext=(2.8, y0 - 0.9),
                       arrowprops=dict(arrowstyle="<-", lw=1.2, color="0.35"))
            a.annotate("", xy=(7.2, y0), xytext=(7.2, y0 - 0.9),
                       arrowprops=dict(arrowstyle="<-", lw=1.2, color="0.35"))
            box(1.2, y0 - 2.0, 3.2, 0.9, "LayerNorm(16)", "#ffe6a8")
            box(5.6, y0 - 2.0, 3.2, 0.9, "LayerNorm(48)", "#ffe6a8")
            a.text(5, 0.15, "independent statistics per block", ha="center",
                   fontsize=7.6, color=GREEN)
        if k == 0:
            a.text(5, 1.9, "z_fast variance up  =>  z_slow rescaled down",
                   ha="center", fontsize=7.8, color=RED)
        if k == 2:
            a.text(5, 1.9, "blocks share upstream stats, but no post-split LN",
                   ha="center", fontsize=7.8, color=GREY)
    plt.tight_layout(); plt.savefig("docs/eda/figs/explain_blocknorm.png", dpi=115)
    plt.close()


if __name__ == "__main__":
    fig_targets(); fig_blocknorm()
    print("saved explain_targets.png, explain_blocknorm.png")
