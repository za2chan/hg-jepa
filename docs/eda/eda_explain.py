"""Two explanatory figures for the PROTOCOL Q&A (A2 window policy, B4 sampling).
English labels (matplotlib has no Korean font here); prose explains in Korean.
Saves to docs/eda/figs/. Observation only."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp

FS = 50.0


# ---- Figure 1: A2 window-vs-segment policy ----
def fig_window_policy():
    # a toy HAPT-like label timeline (seconds)
    segs = [(0, 3, "VOID", "0.85"), (3, 20, "SIT", "#8ecae6"),
            (20, 23, "STAND_TO_WALK\n(transition 7-12)", "#ffb703"),
            (23, 45, "WALK", "#90be6d"), (45, 48, "VOID", "0.85")]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    for s, e, name, c in segs:
        ax.axvspan(s, e, ymin=0.55, ymax=1.0, color=c, alpha=0.9)
        ax.text((s + e) / 2, 0.9, name, ha="center", va="center", fontsize=8)
    ax.text(-1.5, 0.9, "labels", ha="right", va="center", fontsize=9, weight="bold")

    WIN = 10.24  # HAPT L=128 window in seconds
    cands = [(6.5, "A  fully inside SIT", "#2a9d8f", "end-label SIT  -> CLEAN"),
             (16.5, "B  SIT->transition->WALK", "#e34948", "end-label WALK  -> CONTAMINATED"),
             (0.5, "C  starts in VOID", "#e34948", "end-label SIT  -> CONTAMINATED")]
    ys = [0.4, 0.22, 0.04]
    for (s0, tag, col, note), y in zip(cands, ys):
        ax.add_patch(mp.FancyBboxPatch((s0, y), WIN, 0.12, boxstyle="round,pad=0.0",
                     ec=col, fc="none", lw=2))
        ax.plot([s0 + WIN], [y + 0.06], marker=">", color=col)   # end-point marker
        ax.text(s0 + WIN + 0.4, y + 0.06, note, va="center", fontsize=8, color=col)
        ax.text(s0, y + 0.06, tag + "  ", ha="right", va="center", fontsize=8)
    ax.set_xlim(-6, 52); ax.set_ylim(0, 1.05); ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title("A2: a 10.24 s window vs the label timeline\n"
                 "recommended policy = keep ONLY windows fully inside one activity segment (like A)")
    plt.tight_layout(); plt.savefig("docs/eda/figs/explain_window_policy.png", dpi=115)
    plt.close()


# ---- Figure 2: B4 anchor/Delta sampling density ----
def fig_sampling():
    L = 256
    OFFS = [1, 4, 16, 64, 128]
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6), sharey=True)
    rng = np.random.default_rng(0)

    def draw(a, anchors, per_anchor_deltas, title):
        a.axhspan(-0.5, len(anchors) - 0.5, xmin=0, xmax=1, color="0.96")
        # window bar
        a.plot([0, L], [len(anchors) + 0.5] * 2, color="k", lw=6, solid_capstyle="butt")
        a.text(L / 2, len(anchors) + 1.1, f"one window: positions 0..{L}", ha="center", fontsize=8)
        for i, p in enumerate(anchors):
            a.plot(p, i, "o", color="#2a78d6", ms=7)                # anchor
            for d in per_anchor_deltas[i]:
                tgt = min(p + d, L - 1)
                a.annotate("", xy=(tgt, i), xytext=(p, i),
                           arrowprops=dict(arrowstyle="->", color="#e9963a", lw=1.1, alpha=0.8))
                a.plot(tgt, i, "x", color="#e9963a", ms=5)
        a.set_title(title); a.set_xlabel("position in window (patches)")
        a.set_xlim(-6, L + 6)

    # v1: 8 anchors in [64,128), ONE random Delta each
    an1 = rng.integers(64, 128, 8)
    d1 = [[OFFS[rng.integers(0, 5)]] for _ in an1]
    n1 = sum(len(d) for d in d1)
    draw(ax[0], sorted(an1), [d1[i] for i in np.argsort(an1)],
         f"v1 (sparse): 8 anchors in [64,128), 1 random Delta each\n= {n1} (anchor, Delta) pairs / forward pass")
    ax[0].axvspan(64, 128, color="#2a78d6", alpha=0.08)
    ax[0].text(96, -1.3, "anchor range [64,128) only", ha="center", color="#2a78d6", fontsize=8)

    # v2: more anchors over wider range, ALL 5 Deltas each
    an2 = np.linspace(20, 120, 8).astype(int)
    d2 = [OFFS for _ in an2]
    n2 = sum(len(d) for d in d2)
    draw(ax[1], an2, d2,
         f"v2 (dense): more anchors, ALL 5 Deltas each\n= {n2} pairs / forward pass (same compute)")
    ax[1].set_ylabel("")
    for a in ax:
        a.set_yticks([])
    plt.tight_layout(); plt.savefig("docs/eda/figs/explain_sampling.png", dpi=115)
    plt.close()


if __name__ == "__main__":
    fig_window_policy(); fig_sampling()
    print("saved explain_window_policy.png, explain_sampling.png")
