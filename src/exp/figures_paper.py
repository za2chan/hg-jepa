"""Paper figures. Every number regenerates from runs_v2/*.json (CLAUDE.md rule 3).

Rewritten 2026-08-07 for the final result set: main22 cells, SEP with no
denominators, the lambda curve against post-hoc points, and the random SPLIT
control (block_factor's rand16/rand48 were independent draws and never formed a
usable null).

    python3 figures_paper.py           -> runs_v2/fig_*.png
    python3 figures_paper.py <name>    -> just that one

Figures, in the order the paper uses them:
  gate        the gate g(Delta) and where it sits (Data and methods)
  difficulty  training-free FFT accuracy vs regime gap -- why +-3% (Results 1)
  ablation    main 2x2 SEP, 3 datasets x 2 stems, random-split line (Results 2)
  rotation    inclusion-exclusion plane: our lambda CURVE vs post-hoc POINTS (Results 3)
  cost        downstream cost against the mechanism-free model (Results 4)
  sensitivity lambda / tau / d_slow (appendix)
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hglp"))
from probes import sep_index

R = pathlib.Path(__file__).resolve().parents[2] / "runs_v2"
load = lambda n: json.loads((R / n).read_text())
save = lambda f, n: (f.savefig(R / f"fig_{n}.png", dpi=200, bbox_inches="tight"),
                     plt.close(f), print(f"  runs_v2/fig_{n}.png"))

STEM = {"l1+ema": "HGLP-Reg", "nce+ema": "HGLP-NCE"}
DSET = [("Synthetic $\\pm$3%", "synth_gap0.03"), ("PTB-XL", "ptbxl"), ("HAPT", "hapt")]
# colourblind-safe; only "gate + xcov" is saturated so it reads first
COL = {"g0_x0_noBN": "#9aa0a6", "g1_x0_noBN": "#7ba7d7",
       "g0_x1": "#e8a33d", "g1_x1": "#1a5fb4"}
LBL = {"g0_x0_noBN": "neither", "g1_x0_noBN": "gate only",
       "g0_x1": "xcov only", "g1_x1": "gate + xcov"}
LAMS = [0, 1, 4, 16, 64]

pair = lambda b, k: (b[k]["slow"], b[k]["fast"])
sep_of = lambda a, b: sep_index({"z_slow": a, "z_fast": b})


def gate():
    """g(Delta) = sigmoid((tau - Delta)/W): what the predictor may read at each horizon."""
    fig, ax = plt.subplots(figsize=(5, 2.6))
    d = np.linspace(1, 128, 500)
    for tau, ls in ((16.0, "-"), (40.0, "--")):
        ax.plot(d, 1 / (1 + np.exp((d - tau) / 4.0)), ls, lw=2,
                label=f"$\\tau$ = {tau:g}")
    ax.axhline(1, color="k", lw=.5, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("horizon $\\Delta$ (patches)")
    ax.set_ylabel("gate $g(\\Delta)$")
    ax.set_title("$z_{\\mathrm{mix}}$ is multiplied by $g(\\Delta)$; "
                 "$z_{\\mathrm{slow}}$ always passes", fontsize=9)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.3)
    save(fig, "gate")


def difficulty():
    """A training-free FFT classifier calibrates how hard each synthetic setting is."""
    d = load("difficulty.json")["fft"]
    gaps = [0.05, 0.03, 0.02, 0.01, 0.005]
    acc = [d[f"gap{g:g}_c64"]["best"] for g in gaps]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot([g * 100 for g in gaps], acc, "o-", color="#1a5fb4", lw=2)
    ax.axhline(1 / 3, color="k", ls=":", lw=1)
    ax.text(4.6, 1 / 3 + .012, "chance", fontsize=8)
    i = gaps.index(0.03)
    ax.annotate("reported setting", (3, acc[i]), (3.4, acc[i] + .10), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=.8))
    ax.set_xlabel("regime gap (%)"); ax.set_ylabel("regime accuracy")
    ax.set_title("Training-free FFT classifier: task difficulty", fontsize=9)
    ax.invert_xaxis(); ax.grid(alpha=.3)
    save(fig, "difficulty")


def ablation():
    """Main 2x2 only. NO random-split line: in the mechanism-free cell the coordinate
    split and a random split agree to within 0.015 in all six settings, so `neither`
    already IS the free baseline here and a second line would just overlap it. The
    random split earns its place in fig_rotation, where the post-hoc unmixings
    re-basis one fixed embedding and an arbitrary split of THAT embedding is the
    reference they must beat."""
    fig, axes = plt.subplots(1, 6, figsize=(15, 3.1), sharey=True)
    i = 0
    for dl, ds in DSET:
        d = load(f"twosided_main22_{ds}.json")
        for stem in ("l1+ema", "nce+ema"):
            ax = axes[i]; i += 1
            cells = ["g0_x0_noBN", "g1_x0_noBN", "g0_x1", "g1_x1"]
            v = [sep_of(pair(d[f"{stem}/{c}"], "z_slow"),
                        pair(d[f"{stem}/{c}"], "z_fast"))["sep"] for c in cells]
            ax.bar(range(4), v, color=[COL[c] for c in cells])
            ax.set_xticks(range(4))
            ax.set_xticklabels([LBL[c] for c in cells], rotation=40, ha="right", fontsize=7)
            ax.set_title(f"{dl}\n{STEM[stem]}", fontsize=8)
            ax.grid(alpha=.3, axis="y")
    axes[0].set_ylabel("SEP")
    fig.suptitle("Separation index by mechanism (5 seeds, $\\lambda$=4)", fontsize=10, y=1.04)
    save(fig, "ablation_2x2")


def rotation():
    """Inclusion-exclusion plane. We trace a CURVE in lambda; post-hoc methods are POINTS.

    Axes are per-panel adaptive (NOT a shared [0,1]): with a fixed [0,1] the real-data
    panels compress every point into a corner and the lambda curve reads as a short
    stub, hiding both the curve shape and whether we sit up-and-right of the
    baselines. Each panel zooms to its own data range (padded), so the curve looks
    like a curve and the win/loss vs post-hoc points is visible. Cross-panel absolute
    comparison is carried by the SEP table (Results C), not by a shared axis here."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.0))
    M = {"SFA": ("s", "#d62728"), "ICA-slow": ("^", "#e8a33d"), "PCA": ("v", "#9aa0a6")}
    for c, (dl, ds) in enumerate(DSET):
        suf = "_gap0.03" if ds.startswith("synth") else ""
        base = "synth" if ds.startswith("synth") else ds
        for r, stem in enumerate(("l1+ema", "nce+ema")):
            ax = axes[r, c]
            d = load(f"rotation_{base}_{stem}{suf}.json")
            f = lambda k: sep_of((d[k]["slow_half"]["slow"][0], d[k]["slow_half"]["fast"][0]),
                                 (d[k]["fast_half"]["slow"][0], d[k]["fast_half"]["fast"][0]))
            pts = [f(f"gate@lam{l}") for l in LAMS]
            base_pts = {m: f(m) for m in M} | {"random": f("random")}
            xs = [p["inclusion"] for p in pts] + [p["inclusion"] for p in base_pts.values()]
            ys = [p["exclusion"] for p in pts] + [p["exclusion"] for p in base_pts.values()]
            ax.plot([p["inclusion"] for p in pts], [p["exclusion"] for p in pts],
                    "o-", color="#1a5fb4", lw=2, ms=5, label="ours ($\\lambda$ curve)")
            for l, p in zip(LAMS, pts):
                ax.annotate(f"{l:g}", (p["inclusion"], p["exclusion"]), fontsize=7,
                            xytext=(3, -9), textcoords="offset points", color="#1a5fb4")
            for m, (mk, col) in M.items():
                p = base_pts[m]
                ax.scatter([p["inclusion"]], [p["exclusion"]], marker=mk, s=60,
                           color=col, zorder=3, label=m)
            p = base_pts["random"]
            ax.scatter([p["inclusion"]], [p["exclusion"]], marker="x", s=50,
                       color="k", zorder=3, label="random split")
            # per-panel adaptive limits: pad the data range by 8%, floor the span so
            # a tight cluster (HAPT) does not zoom to a misleading micro-scale
            def lim(v):
                lo, hi = min(v), max(v)
                span = max(hi - lo, 0.12)
                pad = 0.08 * span + (span - (hi - lo)) / 2
                return lo - pad, hi + pad
            ax.set_xlim(*lim(xs)); ax.set_ylim(*lim(ys))
            ax.set_title(f"{dl} — {STEM[stem]}", fontsize=9)
            ax.set_xlabel("inclusion"); ax.set_ylabel("exclusion")
            ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle("Up and to the right is better. Our knob traces a curve; "
                 "post-hoc unmixings have none. (per-panel axes; see Table for SEP)",
                 fontsize=10, y=.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "rotation")


def rotation_sep():
    """SEP vs lambda: the verdict figure. Our SEP is a CURVE over lambda; each
    post-hoc unmixing is a horizontal line (no knob). Win = our curve pokes above
    the SFA line. Unlike the inclusion-exclusion plane, this shows all three SEP
    terms combined, so it reveals the wins the plane cannot -- HAPT-NCE wins on
    the allocation term, invisible on the plane but visible here."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharex=True)
    BL = {"SFA": "#d62728", "ICA-slow": "#e8a33d", "PCA": "#9aa0a6", "random split": "k"}
    key = {"SFA": "SFA", "ICA-slow": "ICA-slow", "PCA": "PCA", "random split": "random"}
    x = list(range(len(LAMS)))
    for c, (dl, ds) in enumerate(DSET):
        suf = "_gap0.03" if ds.startswith("synth") else ""
        base = "synth" if ds.startswith("synth") else ds
        for r, stem in enumerate(("l1+ema", "nce+ema")):
            ax = axes[r, c]
            d = load(f"rotation_{base}_{stem}{suf}.json")
            sep = lambda k: sep_of((d[k]["slow_half"]["slow"][0], d[k]["slow_half"]["fast"][0]),
                                   (d[k]["fast_half"]["slow"][0], d[k]["fast_half"]["fast"][0]))["sep"]
            ours = [sep(f"gate@lam{l}") for l in LAMS]
            ax.plot(x, ours, "o-", color="#1a5fb4", lw=2, ms=5, zorder=4,
                    label="ours ($\\lambda$ curve)")
            best = max(range(len(ours)), key=lambda i: ours[i])
            ax.scatter([x[best]], [ours[best]], s=140, facecolors="none",
                       edgecolors="#1a5fb4", lw=1.8, zorder=5)  # ring the best lambda
            for name, col in BL.items():
                ax.axhline(sep(key[name]), color=col, ls="--", lw=1.4, label=name)
            ax.set_title(f"{dl} — {STEM[stem]}", fontsize=9)
            ax.grid(alpha=.3, axis="y")
    for ax in axes[1]:
        ax.set_xlabel("$\\lambda$ (xcov weight)")
        ax.set_xticks(x); ax.set_xticklabels([f"{l:g}" for l in LAMS])
    for ax in axes[:, 0]:
        ax.set_ylabel("SEP")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle("SEP vs $\\lambda$. Our knob traces a curve (best ringed); "
                 "post-hoc unmixings are flat lines. Above the line = win.",
                 fontsize=10, y=.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "rotation_sep")


def rotation_synth():
    """Synthetic-only rotation plane (single-column figure). The full 6-panel
    rotation() is illegible on real data at fixed [0,1] axes (points cluster in a
    corner); the synthetic panels are where the curve-vs-point story is visible.
    Real-data verdicts are carried by the SEP table (Results C), not this figure."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), sharex=True, sharey=True)
    M = {"SFA": ("s", "#d62728"), "ICA-slow": ("^", "#e8a33d"), "PCA": ("v", "#9aa0a6")}
    ds = "synth_gap0.03"; suf = "_gap0.03"; base = "synth"
    for r, stem in enumerate(("l1+ema", "nce+ema")):
        ax = axes[r]
        d = load(f"rotation_{base}_{stem}{suf}.json")
        f = lambda k: sep_of((d[k]["slow_half"]["slow"][0], d[k]["slow_half"]["fast"][0]),
                             (d[k]["fast_half"]["slow"][0], d[k]["fast_half"]["fast"][0]))
        pts = [f(f"gate@lam{l}") for l in LAMS]
        ax.plot([p["inclusion"] for p in pts], [p["exclusion"] for p in pts],
                "o-", color="#1a5fb4", lw=2, ms=5, label="ours ($\\lambda$ curve)")
        for l, p in zip(LAMS, pts):
            ax.annotate(f"{l:g}", (p["inclusion"], p["exclusion"]), fontsize=7,
                        xytext=(3, -8), textcoords="offset points", color="#1a5fb4")
        for m, (mk, col) in M.items():
            p = f(m)
            ax.scatter([p["inclusion"]], [p["exclusion"]], marker=mk, s=60,
                       color=col, zorder=3, label=m)
        p = f("random")
        ax.scatter([p["inclusion"]], [p["exclusion"]], marker="x", s=50,
                   color="k", zorder=3, label="random split")
        ax.set_title(f"Synthetic $\\pm$3% — {STEM[stem]}", fontsize=9)
        ax.grid(alpha=.3); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.set_xlabel("inclusion")
    axes[0].set_ylabel("exclusion")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Up and to the right is better. Our knob traces a curve; "
                 "post-hoc unmixings have none.", fontsize=9, y=1.0)
    save(fig, "rotation_synth")


def cost():
    """Downstream cost against the MECHANISM-FREE model (not our own z_full)."""
    files = [("Synthetic $\\pm$3%", "label_x_perturb_synth_gap0.03_l1lam1",
              "label_x_perturb_synth_gap0.03"),
             ("PTB-XL", "label_x_perturb_ptbxl", "label_x_perturb_ptbxl"),
             ("HAPT", "label_x_perturb_hapt", "label_x_perturb_hapt")]
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(3); w = .36
    for j, (stem, src) in enumerate((("l1+ema", 1), ("nce+ema", 2))):
        m, e = [], []
        for lbl, f1, f2 in files:
            d = load(f"{(f1 if src == 1 else f2)}.json")
            s = d["config"]["strengths"][0]
            v = (np.array(d["per_seed"][f"{stem}/B/z_slow/None/{s}"])
                 - np.array(d["per_seed"][f"{stem}/B/base_full/None/{s}"]))
            m.append(v.mean()); e.append(v.std(ddof=1))
        ax.bar(x + (j - .5) * w, m, w, yerr=e, capsize=3,
               color=["#1a5fb4", "#e8a33d"][j], label=STEM[stem])
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels([f[0] for f in files])
    ax.set_ylabel("macro-F1 vs mechanism-free")
    ax.set_title("Cost of separation (full label budget, 5 seeds)", fontsize=9)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.3, axis="y")
    save(fig, "cost")


def sensitivity():
    """lambda / tau / d_slow. Real-data sweeps are 3 seeds -- stated in the caption."""
    AX = [("lam", "$\\lambda$ (xcov weight)", ["1.0", "4.0", "16.0", "64.0"]),
          ("tau", "$\\tau$ multiplier", ["0.5", "1.0", "2.0", "4.0"]),
          ("dslow", "$d_{\\mathrm{slow}}$", ["8", "16", "32"])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.1))
    for ax, (axis, xl, grid) in zip(axes, AX):
        for dl, ds in DSET:
            f = f"sweep_{axis}_{'synth_gap0.03' if ds.startswith('synth') else ds}.json"
            d = load(f)["result"]
            for stem, ls in (("l1+ema", "-"), ("nce+ema", "--")):
                y = [sep_of(pair(d[f"{stem}/{g}"], "z_slow"),
                            pair(d[f"{stem}/{g}"], "z_fast"))["sep"] for g in grid]
                ax.plot(range(len(grid)), y, ls, marker="o", ms=4,
                        label=f"{dl} {STEM[stem]}" if axis == "lam" else None)
        ax.set_xticks(range(len(grid))); ax.set_xticklabels(grid)
        ax.set_xlabel(xl); ax.grid(alpha=.3)
    axes[0].set_ylabel("SEP")
    axes[0].legend(fontsize=6, frameon=False, ncol=2)
    fig.suptitle("Hyper-parameter sensitivity (synthetic 5 seeds, real data 3 seeds)",
                 fontsize=10, y=1.03)
    save(fig, "sensitivity")


ALL = dict(gate=gate, difficulty=difficulty, ablation=ablation,
           rotation=rotation, rotation_synth=rotation_synth,
           rotation_sep=rotation_sep, cost=cost, sensitivity=sensitivity)

if __name__ == "__main__":
    want = sys.argv[1:] or list(ALL)
    for n in want:
        ALL[n]()
