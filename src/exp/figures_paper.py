"""Paper figures/tables. Every number regenerates from runs_v2/*.json (CLAUDE.md rule 3).

python3 src/exp/figures_paper.py   ->  runs_v2/fig_*.png, runs_v2/tab_*.tex
"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parents[2] / "runs_v2"
load = lambda n: json.loads((R / n).read_text())

STEMS = ["l1+ema", "nce+ema"]
STEM_LBL = {"l1+ema": "HGLP-Reg", "nce+ema": "HGLP-NCE"}
CELLS = ["g0_x0", "g1_x0", "g0_x1", "g1_x1"]
CELL_LBL = {"g0_x0": "neither", "g1_x0": "gate only",
            "g0_x1": "xcov only", "g1_x1": "gate + xcov"}
# (dataset, ablation-file stem, slow-metric key, its display name)
DSETS = [("Synthetic", "synth", "regime_acc", "regime acc"),
         ("HAPT", "hapt", "slow_kept_f1", "activity macro-F1")]
LEAK = {"synth": "leak_u_r2", "hapt": "leak_r2"}

# colorblind-safe; the "both" cell is the only saturated one so it reads first
COL = {"g0_x0": "#9aa0a6", "g1_x0": "#7ba7d7", "g0_x1": "#e8a33d", "g1_x1": "#1a5fb4"}


def ablation():
    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5))
    i = 0
    for dname, tag, mkey, mlbl in DSETS:
        for stem in STEMS:
            d = load(f"ablation_{tag}_{stem}.json")
            ax = axes[i]; i += 1
            for c in CELLS:
                r = d[c]["z_slow"]
                x, xe = r[LEAK[tag]]
                y, ye = r[mkey]
                ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", ms=9,
                            color=COL[c], capsize=3, lw=1.5,
                            label=CELL_LBL[c], zorder=3 if c == "g1_x1" else 2)
                rows.append(dict(dataset=dname, stem=stem, cell=c,
                                 kept=y, kept_sd=ye, leak=x, leak_sd=xe,
                                 rankme=r["rankme"][0]))
            ax.set_title(f"{dname} · {STEM_LBL[stem]}", fontsize=10)
            ax.set_xlabel("fast-factor leak into $z_{slow}$  (R²) → worse")
            if i == 1:
                ax.set_ylabel("slow factor kept ↑ better")
            ax.set_xlim(-0.05, 0.85)
            ax.grid(alpha=.25, lw=.6)
            ax.set_axisbelow(True)
            ax.annotate("target", xy=(0.03, 0.96), xytext=(0.30, 0.80),
                        xycoords="axes fraction", textcoords="axes fraction",
                        fontsize=8, color="#999", va="center",
                        arrowprops=dict(arrowstyle="->", color="#bbb", lw=1))
    axes[0].legend(fontsize=8, loc="lower left", framealpha=.9)
    fig.suptitle("The gate ASSIGNS, the xcov penalty EXCLUDES — neither alone works "
                 "(3 seeds, mean ± sd)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(R / "fig_ablation_2x2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # LaTeX
    L = [r"\begin{tabular}{ll" + "cc" * len(DSETS) + "}", r"\toprule",
         " & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{d[0]}}}" for d in DSETS) + r" \\",
         "gate & xcov & " + " & ".join(f"{d[3]} $\\uparrow$ & leak $\\downarrow$" for d in DSETS) + r" \\",
         r"\midrule"]
    for stem in STEMS:
        L.append(rf"\multicolumn{{{2+2*len(DSETS)}}}{{l}}{{\emph{{{STEM_LBL[stem]}}}}} \\")
        for c in CELLS:
            g = r"\checkmark" if c[1] == "1" else "--"
            x = r"\checkmark" if c[4] == "1" else "--"
            cs = []
            for dname, tag, mkey, _ in DSETS:
                r = load(f"ablation_{tag}_{stem}.json")[c]["z_slow"]
                bf = (lambda s: rf"\textbf{{{s}}}") if c == "g1_x1" else (lambda s: s)
                cs += [bf(f"{r[mkey][0]:.3f}\\,\\tiny{{$\\pm${r[mkey][1]:.3f}}}"),
                       bf(f"{r[LEAK[tag]][0]:.3f}\\,\\tiny{{$\\pm${r[LEAK[tag]][1]:.3f}}}")]
            L.append(f"{g} & {x} & " + " & ".join(cs) + r" \\")
        L.append(r"\midrule" if stem != STEMS[-1] else r"\bottomrule")
    L.append(r"\end{tabular}")
    (R / "tab_ablation_2x2.tex").write_text("\n".join(L) + "\n")
    return rows


ROT_ORDER = ["gate(z_slow)", "PCA", "ICA-slow", "SFA", "ungated(z_slow)"]
ROT_LBL = {"gate(z_slow)": r"\textbf{HGLP gate}", "PCA": "PCA",
           "ICA-slow": "ICA (slowness-ranked)", "SFA": "SFA",
           "ungated(z_slow)": "ungated, arbitrary block"}


def rotation():
    dsets = ["synth", "ptbxl", "hapt"]
    names = {"synth": "Synthetic", "ptbxl": "PTB-XL", "hapt": "HAPT"}
    L = [r"\begin{tabular}{l" + "cc" * len(dsets) + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{names[d]}}}" for d in dsets) + r" \\",
         "$d_{slow}$=16 subspace by & " + " & ".join("kept $\\uparrow$ & leak $\\downarrow$" for _ in dsets) + r" \\",
         r"\midrule"]
    for stem in STEMS:
        L.append(rf"\multicolumn{{{1+2*len(dsets)}}}{{l}}{{\emph{{{STEM_LBL[stem]}}}}} \\")
        for m in ROT_ORDER:
            cs = []
            for ds in dsets:
                r = load(f"rotation_{ds}_{stem}.json")[m]
                cs += [f"{r['slow_kept'][0]:.3f}", f"{r['leak'][0]:.3f}"]
            L.append(f"{ROT_LBL[m]} & " + " & ".join(cs) + r" \\")
        L.append(r"\midrule" if stem != STEMS[-1] else r"\bottomrule")
    L.append(r"\end{tabular}")
    (R / "tab_rotation.tex").write_text("\n".join(L) + "\n")

    # slow-kept is near-tied across methods; leak is what separates them -> bars of leak,
    # with slow-kept annotated on each bar so the trade-off stays visible.
    bars = ["gate(z_slow)", "PCA", "SFA", "ungated(z_slow)"]   # ICA == PCA on all three
    blbl = {"gate(z_slow)": "HGLP gate", "PCA": "PCA / ICA", "SFA": "SFA",
            "ungated(z_slow)": "ungated block"}
    mcol = {"gate(z_slow)": "#1a5fb4", "PCA": "#e8a33d",
            "SFA": "#3a9e6e", "ungated(z_slow)": "#9aa0a6"}
    stem = "nce+ema"
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    for ax, ds in zip(axes, dsets):
        y = range(len(bars))
        v = [load(f"rotation_{ds}_{stem}.json")[m] for m in bars]
        ax.barh(list(y), [r["leak"][0] for r in v],
                xerr=[r["leak"][1] for r in v], height=.62,
                color=[mcol[m] for m in bars], error_kw=dict(lw=1, capsize=3))
        for i, r in enumerate(v):
            ax.text(r["leak"][0] + r["leak"][1] + .02, i,
                    f"kept {r['slow_kept'][0]:.3f}", va="center", fontsize=8, color="#444")
        ax.set_yticks(list(y)); ax.set_yticklabels([blbl[m] for m in bars], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, max(r["leak"][0] for r in v) * 1.75)
        ax.set_xlabel("fast-factor leak $\\downarrow$ better")
        ax.set_title(names[ds], fontsize=10)
        ax.grid(alpha=.25, lw=.6, axis="x"); ax.set_axisbelow(True)
    fig.suptitle("Post-hoc rotation keeps the slow factor but does not exclude the fast one "
                 f"({STEM_LBL[stem]}, 3 seeds)", fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(R / "fig_rotation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def shift():
    d = load("ptbxl_v2_seed0.json")["shift"]
    kinds = [k for k in d]
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.5 * len(kinds), 3.2), sharey=True)
    for ax, k in zip(axes, kinds):
        s = d[k]["strengths"]
        ax.plot(s, d[k]["z_slow"], "o-", color="#1a5fb4", lw=2, label="$z_{slow}$ (16d)")
        ax.plot(s, d[k]["z_full"], "s--", color="#9aa0a6", lw=2, label="$z_{full}$ (64d)")
        ax.set_title(f"{k}", fontsize=10)
        ax.set_xlabel("perturbation strength")
        ax.grid(alpha=.25, lw=.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("diagnosis macro-F1")
    axes[0].legend(fontsize=8)
    fig.suptitle("PTB-XL: $z_{slow}$ degrades more gracefully under amplitude (scale) shift "
                 "— single seed, preliminary", fontsize=10.5, y=1.03)
    fig.tight_layout()
    fig.savefig(R / "fig_shift_ptbxl.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rows = ablation()
    rotation()
    shift()
    hdr = f"{'dataset':10}{'stem':9}{'cell':12}{'kept':>16}{'leak':>16}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['dataset']:10}{r['stem']:9}{CELL_LBL[r['cell']]:12}"
              f"{r['kept']:>10.3f}±{r['kept_sd']:.3f}{r['leak']:>10.3f}±{r['leak_sd']:.3f}")
    print("\nwrote fig_ablation_2x2.png fig_rotation.png fig_shift_ptbxl.png "
          "tab_ablation_2x2.tex tab_rotation.tex")
