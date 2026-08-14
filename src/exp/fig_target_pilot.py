"""The harmlessness curve, drawn from runs_v2/pilot_b5.json.

    harmlessness(D) = R2(s,u -> zbar(t+D)) - R2(s -> zbar(t+D))

How much the transient factor adds, BEYOND what the persistent factor already
explains, to a linear reading of the target embedding D ahead. Zero means the
transient factor is redundant at that horizon, so gating it away costs nothing --
the premise the horizon gate rests on. The point of the figure is the CONTRAST
between the two target designs: the receptive-field-bounded target reaches zero
and stays there, the cumulative one never does at any horizon.

No training and no GPU: the curve is already in the pilot's JSON.

    python3 src/exp/fig_target_pilot.py   -> runs_v2/fig_target_pilot.png
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parents[2] / "runs_v2"
# same palette as the other paper figures
C_BOUND, C_CUM, MUTE = "#1a5fb4", "#d64550", "#9aa0a6"
TAU = 16.0                      # the threshold used on synthetic data (train.py default)


def main():
    d = json.loads((OUT / "pilot_b5.json").read_text())
    deltas = sorted((int(k) for k in d["bounded"]["harmlessness"]))

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.axhline(0, color="k", lw=.7, ls=":")
    ax.axvline(TAU, color=MUTE, lw=1.0, ls="--")
    ax.text(TAU * 1.06, .092, r"$\tau$", color=MUTE, fontsize=10, va="top")

    for mode, label, col in (("bounded", "bounded target (ours)", C_BOUND),
                             ("cumulative", "cumulative target (v1)", C_CUM)):
        y = [d[mode]["harmlessness"][str(x)] for x in deltas]
        ax.plot(deltas, y, "o-", color=col, lw=1.6, ms=4.5, label=label)

    ax.set_xscale("log")
    ax.set_xticks(deltas)
    ax.set_xticklabels([str(x) for x in deltas], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel(r"horizon $\Delta$ (patches, log scale)", fontsize=9.5)
    ax.set_ylabel("extra $R^2$ from the\ntransient factor", fontsize=9.5)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax.set_title("Zero means gating the transient block away costs nothing",
                 fontsize=10, pad=6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_target_pilot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  runs_v2/fig_target_pilot.png  "
          f"(bounded {d['bounded']['harmlessness']['16']:+.3f} at D=16; "
          f"cumulative {d['cumulative']['harmlessness']['128']:+.3f} at D=128)")


if __name__ == "__main__":
    main()
