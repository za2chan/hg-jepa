"""Hand-drawn schematics, redrawn: Fig A (Introduction) and Fig B1/B2 (Method).

A   what separation means, as a colour metaphor on the embedding coordinates
B1  where the prediction TARGET comes from, versus other latent-prediction methods
B2  how the horizon gate works, and that it is soft rather than on/off

No measured quantities here -- these are diagrams. Every number that appears
(d_slow=16, D_Z=64, W_gate) is read from model.py / train.py so the picture cannot
drift from the code.

MAIN  the paper's main figure: B0 as panel (a) plus B2's gate panels as (b)-(d),
      with B2's duplicated encoder/predictor/target boxes dropped. B0 and B2 are
      still written standalone so the pieces stay inspectable.

python3 src/exp/fig_schematics.py -> runs_v2/fig_concept_A.png
                                     runs_v2/fig_targets_B1.png
                                     runs_v2/fig_gate_B2.png
                                     runs_v2/fig_arch_B0.png
                                     runs_v2/fig_main.png
"""
import pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "hglp"))
from model import D_Z, D_SLOW                                        # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[2] / "runs_v2"
PERSIST, TRANSIENT, MIXED, MIXED_B = "#C0392B", "#2E5FA3", "#7D5BA6", "#5C63B0"
INK, MUTE = "#222222", "#9aa0a6"
BOXK = dict(boxstyle="round,pad=0.30", linewidth=1.1, edgecolor=INK, facecolor="white")


def box(ax, x, y, w, h, text, fc="white", fs=9, ec=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                linewidth=1.1, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, x0, y0, x1, y1, style="-|>", color=INK, lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=11, lw=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


# ------------------------------------------------------- Figure A, narrow ----
def fig_a_narrow():
    """The same content as fig_a, stacked instead of side-annotated.

    The paper wraps this figure into a fraction of the text column, and the wide
    version does not survive that: its annotations sit to the RIGHT of the bars, so
    shrinking to fit a column shrinks the type past legibility. Here every
    annotation sits under its bar, which trades height for width -- exactly the
    trade a wrapped figure wants."""
    from matplotlib.colors import LinearSegmentedColormap
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    X0, W, H = 3, 94, 9.0

    def bar(y, spans, title, notes):
        for (a, b, c) in spans:
            ax.add_patch(Rectangle((X0 + W * a / D_Z, y), W * (b - a) / D_Z, H,
                                   facecolor=c, edgecolor="white", lw=0.4))
        ax.add_patch(Rectangle((X0, y), W, H, fill=False, edgecolor=INK, lw=1.3))
        ax.text(X0, y + H + 2.5, title, fontsize=11, color=INK, va="bottom")
        for d in (0, D_Z):
            ax.text(X0 + W * d / D_Z, y - 1.6, str(d), fontsize=8.5,
                    ha="center", va="top", color=MUTE)
        dy = 0
        for txt, size, col in notes:
            ax.text(X0, y - 7.0 - dy, txt, fontsize=size, color=col, va="top")
            dy += 5.0 * (1 + txt.count("\n"))

    bar(84, [(0, D_Z, MIXED)], "Ordinary embedding",
        [("every coordinate mixes both factors", 9.5, INK)])

    bar(50, [(0, D_SLOW, PERSIST), (D_SLOW, D_Z, MIXED_B)], "What we ask for",
        [("a nameable block holds the persistent\nfactor and little else", 9.5, INK),
         ("the rest is unconstrained — it keeps both", 9, MUTE)])
    ax.text(X0 + W * D_SLOW / D_Z, 48.4, str(D_SLOW), fontsize=8.5,
            ha="center", va="top", color=MUTE)
    ax.text(X0 + W * D_SLOW / (2 * D_Z), 54.5, "$z_{per}$", fontsize=10,
            ha="center", color="white", weight="bold")
    ax.text(X0 + W * (D_SLOW + D_Z) / (2 * D_Z), 54.5, "$z_{mix}$", fontsize=10,
            ha="center", color="white", weight="bold")

    cm = LinearSegmentedColormap.from_list("pt", [PERSIST, MIXED, TRANSIENT])
    kx, ky, kw = X0, 14, W
    for i in range(160):
        ax.add_patch(Rectangle((kx + kw * i / 160, ky), kw / 160 + 0.05, 3.2,
                               facecolor=cm(i / 159), edgecolor="none"))
    ax.add_patch(Rectangle((kx, ky), kw, 3.2, fill=False, edgecolor=MUTE, lw=0.8))
    ax.text(kx, ky + 4.4, "what a coordinate carries", fontsize=9, color=INK)
    for x, ha, txt in ((kx, "left", "persistent\nfactor only"),
                       (kx + kw / 2, "center", "both"),
                       (kx + kw, "right", "transient\nfactor only")):
        ax.text(x, ky - 1.4, txt, fontsize=8, ha=ha, va="top", color=MUTE)

    fig.savefig(OUT / "fig_concept_A_narrow.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure A ----
def fig_a():
    fig, ax = plt.subplots(figsize=(9.2, 3.5))
    ax.set_xlim(0, 100); ax.set_ylim(-5, 42); ax.axis("off")
    X0, W = 8, 62

    def bar(y, spans, label):
        for (a, b, c) in spans:
            ax.add_patch(Rectangle((X0 + W * a / D_Z, y), W * (b - a) / D_Z, 7.0,
                                   facecolor=c, edgecolor="white", lw=0.4))
        ax.add_patch(Rectangle((X0, y), W, 7.0, fill=False, edgecolor=INK, lw=1.3))
        ax.text(X0, y + 9.0, label, fontsize=10.5, color=INK, va="bottom")
        for d in (0, D_Z):
            ax.text(X0 + W * d / D_Z, y - 2.2, str(d), fontsize=8, ha="center", color=MUTE)

    bar(30, [(0, D_Z, MIXED)], "Ordinary embedding")
    ax.text(X0 + W + 3, 33.5, "every coordinate mixes\nboth factors",
            fontsize=9, va="center", color=INK)

    bar(11, [(0, D_SLOW, PERSIST), (D_SLOW, D_Z, MIXED_B)], "What we ask for")
    ax.text(X0 + W * D_SLOW / D_Z, 9.0, str(D_SLOW), fontsize=8, ha="center", color=MUTE)
    ax.text(X0 + W * D_SLOW / (2 * D_Z), 14.5, "$z_{per}$", fontsize=10,
            ha="center", color="white", weight="bold")
    ax.text(X0 + W * (D_SLOW + D_Z) / (2 * D_Z), 14.5, "$z_{mix}$", fontsize=10,
            ha="center", color="white", weight="bold")
    ax.text(X0 + W + 3, 16.5, "a nameable block holds the\npersistent factor and little else",
            fontsize=9, va="center", color=INK)
    ax.text(X0 + W + 3, 11.0, "the rest is unconstrained —\nit keeps both",
            fontsize=8.5, va="center", color=MUTE)

    # colour key: a continuum, because z_mix is neither pure colour
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("pt", [PERSIST, MIXED, TRANSIENT])
    kx, ky, kw = X0, 0.6, 34
    for i in range(120):
        ax.add_patch(Rectangle((kx + kw * i / 120, ky), kw / 120 + 0.05, 2.0,
                               facecolor=cm(i / 119), edgecolor="none"))
    ax.add_patch(Rectangle((kx, ky), kw, 2.0, fill=False, edgecolor=MUTE, lw=0.8))
    ax.text(kx, ky - 0.6, "persistent\nfactor only", fontsize=7.5, ha="left",
            va="top", color=MUTE)
    ax.text(kx + kw / 2, ky - 0.6, "both", fontsize=7.5, ha="center", va="top", color=MUTE)
    ax.text(kx + kw, ky - 0.6, "transient\nfactor only", fontsize=7.5, ha="right",
            va="top", color=MUTE)
    ax.text(kx + kw + 3, ky + 1.0, "what a coordinate carries", fontsize=8.5,
            va="center", color=INK)
    fig.savefig(OUT / "fig_concept_A.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- Figure B1 ----
def fig_b1():
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 5.4))
    X0, W = 6, 62
    # TS-JEPA (Ennadir et al., arXiv:2509.25449): uniform 70% patch masking, a
    # standard (non-causal) transformer over the VISIBLE patches, and an EMA encoder
    # applied to the MASKED patches as the target. No horizon variable.
    rng = np.random.default_rng(3)
    npatch = 14
    masked = np.zeros(npatch, bool)
    masked[rng.permutation(npatch)[:int(round(0.70 * npatch))]] = True
    ctx_p = [(i / npatch, (i + 1) / npatch, "") for i in range(npatch) if not masked[i]]
    tgt_p = [(i / npatch, (i + 1) / npatch, "") for i in range(npatch) if masked[i]]
    rows = [
        ("TS-JEPA", "uniform 70% patch masking, no time order and no horizon;\n"
                    "target = the masked patches, encoded by an EMA copy",
         ctx_p, tgt_p, "tsjepa"),
        ("HEPA", "target = the WHOLE future interval, attention-pooled",
         [(0.02, 0.42, "context $x_{0:t}$")], [(0.42, 0.82, "pooled $(t,\\,t+\\Delta]$")], "hepa"),
        ("CPC lineage — ours", "target = ONE position, receptive field bounded to $w$",
         [(0.02, 0.42, "context $x_{0:t}$")], [(0.70, 0.82, "$w$")], "ours"),
    ]
    for ax, (name, sub, ctxs, tgts, kind) in zip(axes, rows):
        ax.set_xlim(0, 100); ax.set_ylim(-1.6, 5.2); ax.axis("off")
        ax.add_patch(Rectangle((X0, 0), W, 2.2, fill=False, edgecolor=INK, lw=1.2))
        for a, b, lab in ctxs:
            ax.add_patch(Rectangle((X0 + W * a, 0), W * (b - a), 2.2,
                                   facecolor="#d7dce3", edgecolor="none"))
            ax.text(X0 + W * (a + b) / 2, 1.1, lab, fontsize=8, ha="center", va="center")
        for a, b, lab in tgts:
            ax.add_patch(Rectangle((X0 + W * a, 0), W * (b - a), 2.2,
                                   facecolor=PERSIST, alpha=.30, edgecolor=PERSIST, lw=1.2))
            ax.text(X0 + W * (a + b) / 2, 1.1, lab, fontsize=8, ha="center",
                    va="center", color="#7a1f16")
        if kind in ("hepa", "ours"):
            ax.plot([X0 + W * 0.42] * 2, [-0.5, 2.9], color=MUTE, lw=1, ls=":")
            ax.text(X0 + W * 0.42, -1.2, "$t$", fontsize=9, ha="center")
            ax.plot([X0 + W * 0.82] * 2, [-0.5, 2.9], color=MUTE, lw=1, ls=":")
            ax.text(X0 + W * 0.82, -1.2, "$t+\\Delta$", fontsize=9, ha="center")
            arrow(ax, X0 + W * 0.42, 3.6, X0 + W * 0.82, 3.6, style="<|-|>", color=MUTE)
            ax.text(X0 + W * 0.62, 4.1, "$\\Delta$", fontsize=9, ha="center", color=MUTE)
        if kind == "ours":
            ax.text(X0 + W * 0.76, 3.0, "read the LAST position", fontsize=7.5,
                    ha="center", color="#7a1f16")
        if kind == "tsjepa":
            ax.text(X0 + W * 0.5, 3.0,
                    "grey = visible (encoder)      red = masked (target)",
                    fontsize=7.5, ha="center", color=MUTE)
            ax.text(X0 + W + 3, -0.9, "encoder is NON-causal here; ours is causal",
                    fontsize=7.5, color=MUTE, va="center")
        ax.text(X0, 4.3, name, fontsize=10.5, weight="bold", color=INK)
        ax.text(X0 + W + 3, 1.1, sub, fontsize=8.5, va="center", color=INK)
    axes[2].text(X0 + W + 3, -0.6,
                 "$w_{eff}=\\min(w,\\Delta)$ keeps the target's input inside $(t,\\,t+\\Delta]$,\n"
                 "so it never re-contains the anchor's own past",
                 fontsize=7.5, color=MUTE, va="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig_targets_B1.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- Figure B2 / main pieces ----
def _pipeline(ax, closed, title, boxes=True):
    """One horizon case: timeline + the embedding column.

    boxes=False drops the encoder/predictor/target chain, which panel (a) of
    fig_main already draws in full. Colours, hatching, timeline and the
    g-label are identical either way.
    """
    ax.set_xlim(0, 100 if boxes else 82); ax.set_ylim(0, 42); ax.axis("off")
    ax.text(0, 39, title, fontsize=10.5, weight="bold", color=INK)
    # --- the timeline this case refers to ------------------------------------
    sx, sw, sy, sh = 2, 76, 28, 3.4
    d_frac = 0.20 if not closed else 0.62          # where t+Delta lands
    t_f, tau_f = 0.22, 0.42                        # t and the tau boundary
    ax.add_patch(Rectangle((sx, sy), sw, sh, fill=False, edgecolor=INK, lw=1.1))
    ax.add_patch(Rectangle((sx, sy), sw * t_f, sh,
                           facecolor="#d7dce3", edgecolor="none"))
    ax.text(sx + sw * t_f / 2, sy + sh / 2, "context", fontsize=7, ha="center",
            va="center")
    wf = 0.07
    a_ = sx + sw * (t_f + d_frac) - sw * wf
    ax.add_patch(Rectangle((a_, sy), sw * wf, sh, facecolor=PERSIST, alpha=.30,
                           edgecolor=PERSIST, lw=1.0))
    ax.text(a_ + sw * wf / 2, sy + sh + 1.0, "$w$", fontsize=7, ha="center",
            color="#7a1f16")
    for f, lab in [(t_f, "$t$"), (t_f + d_frac, "$t+\\Delta$")]:
        ax.plot([sx + sw * f] * 2, [sy - 1.2, sy + sh + 0.6], color=MUTE, lw=0.9, ls=":")
        ax.text(sx + sw * f, sy - 3.4, lab, fontsize=8, ha="center")
    ax.plot([sx + sw * (t_f + tau_f)] * 2, [sy - 1.2, sy + sh + 3.2],
            color=PERSIST, lw=1.2, ls="--")
    ax.text(sx + sw * (t_f + tau_f), sy + sh + 4.0, "$t+\\tau$", fontsize=8,
            ha="center", color=PERSIST)
    # --- the embedding column, split into the two blocks ---------------------
    cx = 24 if boxes else 36
    cy, cw, ch = 5, 7, 15
    if boxes:
        box(ax, 2, 9, 17, 7, "causal\nencoder", fs=8.5)
        arrow(ax, 19, 12.5, cx, 12.5)
    n_slow = int(ch * D_SLOW / D_Z)
    ax.add_patch(Rectangle((cx, cy + ch - n_slow), cw, n_slow,
                           facecolor=PERSIST, edgecolor=INK, lw=1.0))
    ax.add_patch(Rectangle((cx, cy), cw, ch - n_slow,
                           facecolor=("#e9ecef" if closed else MIXED_B),
                           edgecolor=INK, lw=1.0,
                           hatch=("//" if closed else None)))
    ax.text(cx + cw / 2, cy + ch - n_slow / 2, "$z_{per}$", fontsize=7.5,
            ha="center", va="center", color="white", weight="bold")
    ax.text(cx + cw / 2, cy + (ch - n_slow) / 2, "$z_{mix}$", fontsize=7.5,
            ha="center", va="center",
            color=(MUTE if closed else "white"), weight="bold")
    ax.text(cx + cw / 2, cy - 2.6, "$g(\\Delta)\\!\\approx\\!0$" if closed
            else "$g(\\Delta)\\!\\approx\\!1$", fontsize=8.5, ha="center",
            color=(MUTE if closed else INK))
    if boxes:
        arrow(ax, cx + cw, 12.5, 40, 12.5)
        box(ax, 40, 9, 15, 7, "predictor\n(MLP)", fs=8.5)
        arrow(ax, 55, 12.5, 61, 12.5)
        ax.text(58, 14.4, "$\\Delta$", fontsize=8, ha="center", color=MUTE)
        box(ax, 61, 9, 17, 7, "latent target\nat $t+\\Delta$", fs=8.5)
        ax.text(69.5, 6.6, "L1 or InfoNCE", fontsize=7.5, ha="center", color=MUTE)
    msg = ("only the persistent block reaches the predictor" if closed
           else "the whole embedding reaches the predictor")
    if boxes:
        ax.text(24, 21.5, msg, fontsize=8.5, color=(PERSIST if closed else INK))
    else:
        ax.text(40, 21.5, msg, fontsize=8.5, ha="center",
                color=(PERSIST if closed else INK))


def _gate_curve(axg, tau=16.0, Wg=4.0):
    """The gate as a function of horizon. Note kappa, not W (outline III)."""
    d = np.linspace(0, 4 * tau, 400)
    g = 1 / (1 + np.exp(-(tau - d) / Wg))
    axg.plot(d, g, lw=2.2, color=MIXED)
    axg.axvline(tau, color=MUTE, ls=":", lw=1.2)
    axg.text(tau * 1.06, 0.93, "$\\tau$", fontsize=11, color=INK)
    axg.fill_between(d, 0, g, where=d < tau, color=MIXED_B, alpha=.14)
    axg.set_xticks([0, tau, 2 * tau, 3 * tau])
    axg.set_xticklabels(["0", "$\\tau$", "$2\\tau$", "$3\\tau$"], fontsize=9)
    axg.set_xlabel("horizon $\\Delta$", fontsize=9)
    axg.set_ylabel("$g(\\Delta)$  — how much of $z_{mix}$ passes", fontsize=8.5)
    axg.text(0.97, 0.72, "$g(\\Delta)=\\sigma\\!\\left(\\dfrac{\\tau-\\Delta}{\\kappa}\\right)$",
             transform=axg.transAxes, fontsize=10, ha="right",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc"))
    axg.set_ylim(-0.03, 1.08)
    axg.grid(alpha=.25, lw=.6); axg.set_axisbelow(True)
    axg.spines[["top", "right"]].set_visible(False)


def fig_b2(tau=16.0, Wg=4.0):
    fig = plt.figure(figsize=(10.4, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.5, 1.05], height_ratios=[1, 1],
                          hspace=0.45, wspace=0.30)
    _pipeline(fig.add_subplot(gs[0, 0]), False, "(a)  short horizon,  $\\Delta < \\tau$")
    _pipeline(fig.add_subplot(gs[1, 0]), True, "(b)  long horizon,  $\\Delta > \\tau$")
    axg = fig.add_subplot(gs[0, 1])
    _gate_curve(axg, tau, Wg)
    axg.set_title("the gate is soft, not on/off", fontsize=9.5, pad=8)
    fig.text(0.80, 0.28,
             "$\\tau$ comes from the transient\nfactor's lifetime; $\\Delta_{max}$ is set\n"
             "separately, so the axis is drawn\nin units of $\\tau$, not absolute\nhorizons.",
             fontsize=8, color=MUTE, va="center", ha="center")
    fig.suptitle("$z_{per}$ is never gated; only its complement is, and only at long horizons",
                 fontsize=10, y=0.99, color=INK)
    fig.savefig(OUT / "fig_gate_B2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)





# ------------------------------------------------------- Figure B0: overview --
def fig_arch(ax=None):
    """Standalone when ax is None; otherwise draws into the caller's axes
    (fig_main panel a) and leaves saving to the caller."""
    """One-page architecture summary, in the visual idiom of the HEPA figure.

    NOTATION. The code calls the gate's transition width `W_gate` and the target
    receptive field `w`. They are different quantities and `W` / `w` collide on
    the page, so the PAPER writes the gate width as kappa:
        g(Delta) = sigmoid((tau - Delta) / kappa)
    """
    from matplotlib.patches import Polygon
    standalone = ax is None
    fig, ax = (plt.subplots(figsize=(14.6, 5.8)) if standalone
               else (ax.figure, ax))
    ax.set_xlim(0, 146); ax.set_ylim(-4, 56); ax.axis("off")
    TRAIN, FROZEN = "#F08C5A", "#b9bfc7"
    hot, cold = "#FDE7DA", "#eceff2"

    def trap(x, y, w_, h, label, fc, ec):
        ax.add_patch(Polygon([[x, y], [x + w_, y + h * 0.16], [x + w_, y + h * 0.84],
                              [x, y + h]], closed=True, facecolor=fc, edgecolor=ec, lw=1.6))
        ax.text(x + w_ / 2, y + h / 2, label, ha="center", va="center", fontsize=12)

    def badge(x, y, txt, fc, ec):
        ax.text(x, y, txt, ha="center", va="center", fontsize=7.5, color=ec,
                bbox=dict(boxstyle="round,pad=0.25", fc=fc, ec=ec, lw=1.0))

    def circ(x, y, label, fc, ec, r=2.7):
        ax.add_patch(plt.Circle((x, y), r, facecolor=fc, edgecolor=ec, lw=1.5))
        ax.text(x, y, label, ha="center", va="center", fontsize=9.5)

    # ---- input strip -------------------------------------------------------
    sx, sw, sy, sh = 3, 32, 24, 15
    tf, df, wf = 0.44, 0.90, 0.14
    ax.add_patch(Rectangle((sx, sy), sw * tf, sh, facecolor=hot, edgecolor="none"))
    ax.add_patch(Rectangle((sx, sy), sw, sh, fill=False, edgecolor=INK, lw=1.3))
    rp = np.random.default_rng(1)
    for k in range(3):
        yy = sy + sh * (0.22 + 0.28 * k)
        xs = np.linspace(sx, sx + sw, 500)
        ax.plot(xs, yy + 1.3 * np.sin(xs * 1.5 + k) + 0.35 * rp.standard_normal(500),
                lw=0.6, color="#8a4a22", alpha=.75)
    for f in np.arange(0.1, 1.0, 0.1):
        ax.plot([sx + sw * f] * 2, [sy, sy + sh], color="white", lw=0.7, alpha=.7)
    # the target's receptive field: solid, outlined, and bracketed so it reads
    wl, wr = sx + sw * (df - wf), sx + sw * df
    ax.add_patch(Rectangle((wl, sy), wr - wl, sh, facecolor="#9fb3c8", alpha=.75,
                           edgecolor="#3c5a75", lw=1.8, zorder=3))
    arrow(ax, wl, sy - 2.4, wr, sy - 2.4, style="<|-|>", color="#3c5a75", lw=1.4)
    ax.text((wl + wr) / 2, sy - 5.6, r"target window $w$", fontsize=9.5,
            ha="center", color="#3c5a75")
    ax.text((wl + wr) / 2, sy - 8.4, r"($\Delta\geq w$ enforced)", fontsize=7.5,
            ha="center", color=MUTE)
    ax.plot([sx + sw * tf] * 2, [sy - 1.0, sy + sh + 2], color=INK, lw=1.7)
    ax.text(sx + sw * tf, sy + sh + 3.0, r"$t$", fontsize=10.5, ha="center")
    ax.plot([wr] * 2, [sy - 1.0, sy + sh + 2], color=MUTE, lw=1.1, ls=":")
    ax.text(wr, sy + sh + 3.0, r"$t+\Delta$", fontsize=10.5, ha="center")
    ax.text(sx + sw * tf / 2, sy - 2.8, r"context  $x_{\leq t}$", fontsize=9.5,
            ha="center", color="#8a4a22")
    # the horizon itself, and where it goes
    arrow(ax, sx + sw * tf, sy + sh + 7.0, wr, sy + sh + 7.0, style="<|-|>", color=TRAIN,
          lw=1.5)
    ax.text((sx + sw * tf + wr) / 2, sy + sh + 8.4, r"horizon $\Delta$", fontsize=10,
            ha="center", color=TRAIN)

    # ---- online branch -----------------------------------------------------
    arrow(ax, sx + sw * tf, 31.5, 41, 31.5, color=TRAIN, lw=1.8)
    trap(41, 24, 9, 15, r"$f_\theta$", hot, TRAIN)
    badge(45.5, 42.0, "trained", hot, TRAIN)
    arrow(ax, 50, 31.5, 55, 31.5)

    bx, by, bw, bh = 55, 24, 6, 15
    ns = bh * D_SLOW / D_Z
    ax.add_patch(Rectangle((bx, by + bh - ns), bw, ns, facecolor=PERSIST,
                           edgecolor=INK, lw=1.2))
    ax.add_patch(Rectangle((bx, by), bw, bh - ns, facecolor=MIXED_B,
                           edgecolor=INK, lw=1.2))
    ax.text(bx + bw / 2, by + bh - ns / 2, r"$z_{per}$", fontsize=7, ha="center",
            va="center", color="white", weight="bold", rotation=90)
    ax.text(bx + bw / 2, by + (bh - ns) / 2, r"$z_{mix}$", fontsize=7, ha="center",
            va="center", color="white", weight="bold", rotation=90)
    ax.text(bx + bw / 2, by + bh + 2.0, r"$\mathbf{z}_t$", fontsize=10.5, ha="center")

    # ---- THE GATE, named ---------------------------------------------------
    gy = by + (bh - ns) / 2
    gx0, gw, gh = 66, 16, 8
    ax.add_patch(FancyBboxPatch((gx0, gy - gh / 2), gw, gh,
                                boxstyle="round,pad=0.02", linewidth=2.4,
                                edgecolor=PERSIST, facecolor="#FBE3E0"))
    ax.text(gx0 + gw / 2, gy + 1.2, "Horizon Gate", fontsize=10.5, ha="center",
            va="center", color="#7a1f16", weight="bold")
    ax.text(gx0 + gw / 2, gy - 2.0,
            r"$\times\ g(\Delta)=\sigma\!\left(\frac{\tau-\Delta}{\kappa}\right)$",
            fontsize=9, ha="center", va="center", color="#7a1f16")
    arrow(ax, bx + bw, gy, gx0, gy)
    ax.text(63.5, gy + 2.2, r"$z_{mix}$", fontsize=7.5, ha="center", color=MIXED_B)
    # z_slow bypasses the gate entirely -- the point of the figure
    arrow(ax, bx + bw, by + bh - ns / 2, 88, by + bh - ns / 2, color=PERSIST, lw=1.8)
    ax.text(74, by + bh - ns / 2 + 1.6, r"$z_{per}$ never gated", fontsize=8,
            ha="center", color=PERSIST)
    arrow(ax, gx0 + gw, gy, 88, gy)

    box(ax, 88, 26.5, 18, 10, r"Predictor $g_\phi$", fc=hot, ec=TRAIN, fs=11)
    badge(97, 39.4, "trained", hot, TRAIN)

    # ---- horizon conditioning: routed FROM the horizon on the strip ---------
    hy = 49.5
    ax.plot([(sx + sw * tf + wr) / 2, (sx + sw * tf + wr) / 2], [sy + sh + 7.6, hy],
            color=TRAIN, lw=1.2, ls="--")
    ax.plot([(sx + sw * tf + wr) / 2, 92], [hy, hy], color=TRAIN, lw=1.2, ls="--")
    box(ax, 92, hy - 3.0, 14, 6, r"$\log_2\Delta$", fc="white", ec=TRAIN, fs=9.5)
    ax.text(99, hy + 4.2, "how far ahead the target is", fontsize=7.5, ha="center",
            color=TRAIN)
    arrow(ax, 99, hy - 3.0, 99, 36.5, color=TRAIN)

    arrow(ax, 106, 31.5, 111, 31.5)
    circ(114, 31.5, r"$\hat{\mathbf{z}}$", hot, TRAIN)

    # ---- target branch -----------------------------------------------------
    tcx = (wl + wr) / 2
    arrow(ax, tcx, sy - 9.8, tcx, 10, color="#3c5a75")
    arrow(ax, tcx, 10, 41, 10, color=MUTE)
    trap(41, 3, 9, 14, r"$f_{\bar\theta}$", cold, FROZEN)
    badge(45.5, 0.0, "EMA copy · stop-grad", cold, "#6b727a")
    arrow(ax, 50, 10, 66, 10, color=MUTE)
    ax.text(58, 11.6, "read last position", fontsize=8, ha="center", color=MUTE)
    circ(69, 10, r"$\mathbf{z}^{*}$", cold, FROZEN)
    arrow(ax, 71.7, 10, 122, 10, color=MUTE)

    # ---- losses ------------------------------------------------------------
    box(ax, 118, 26.5, 26, 10, r"$\mathcal{L}_{1}$   or   $\mathcal{L}_{\mathrm{NCE}}$",
        fc="white", ec=INK, fs=11)
    arrow(ax, 116.7, 31.5, 118, 31.5)
    arrow(ax, 122, 10, 122, 26.5, color=MUTE)
    box(ax, 66, 15.5, 16, 6.5, r"$\lambda\,\mathcal{L}_{\mathrm{xcov}}$",
        fc="white", ec=INK, fs=10)
    ax.plot([bx + bw / 2, bx + bw / 2], [by, 18.75], ls=":", color=MUTE, lw=1.1)
    arrow(ax, bx + bw / 2, 18.75, 66, 18.75, ls=":", color=MUTE)
    ax.text(74, 13.2, "between the two blocks of $\\mathbf{z}_t$", fontsize=7.5,
            ha="center", color=MUTE)

    ax.text(3, 54.0, "(a)  Horizon-Gated Latent Prediction", fontsize=13, weight="bold")
    if not standalone:
        return
    fig.savefig(OUT / "fig_arch_B0.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------ Figure MAIN: B0 + B2 in one -------
def fig_main(tau=16.0, Wg=4.0):
    """The paper's main figure. Panel (a) is the architecture (was B0);
    (b)-(d) are what the gate does (was B2), with B2's duplicated
    encoder/predictor/target boxes dropped — (a) already shows them."""
    fig = plt.figure(figsize=(14.6, 9.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[5.8, 3.3], hspace=0.10)

    fig_arch(fig.add_subplot(gs[0]))

    bot = gs[1].subgridspec(1, 3, width_ratios=[1, 1, 0.82], wspace=0.22)
    _pipeline(fig.add_subplot(bot[0]), False,
              "(b)  short horizon,  $\\Delta < \\tau$", boxes=False)
    _pipeline(fig.add_subplot(bot[1]), True,
              "(c)  long horizon,  $\\Delta > \\tau$", boxes=False)
    axg = fig.add_subplot(bot[2])
    _gate_curve(axg, tau, Wg)
    axg.set_title("(d)  the gate is soft, not on/off", fontsize=10.5,
                  weight="bold", loc="left", pad=8)

    fig.savefig(OUT / "fig_main.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_a(); fig_b1(); fig_b2(); fig_arch(); fig_main()
    print("wrote fig_concept_A  fig_targets_B1  fig_gate_B2  fig_arch_B0  fig_main")
