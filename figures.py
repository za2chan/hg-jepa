"""Report figures from hard-regime 3-seed stats."""
import json

import matplotlib.pyplot as plt
import numpy as np

C = {"blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100",
     "ink": "#1f2937", "muted": "#6b7280", "grid": "#e5e7eb"}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": C["muted"],
                     "axes.linewidth": 0.6, "figure.dpi": 200})

import glob
TAGS = {"NEPA gate+dcor (ours)": "nepa_g1_d1", "NEPA gate only": "nepa_g1_d0",
        "NEPA nogate": "nepa_g0_d0", "AR + gate": "ar_g1_d0", "AR no gate": "ar_g0_d0"}
S = {}
for name, pre in TAGS.items():
    rs = [json.load(open(f)) for f in glob.glob(f"runs/{pre}_s*_tau16_ds16_lam4.json")]
    S[name] = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
               for k in rs[0] if k != "tag"}
FULL = {"regime": "z_full->regime_acc", "u": "z_full->u_r2", "phase": "z_full->phase_r2"}
SLOW = {"regime": "z_slow->regime_acc", "u": "z_slow->u_r2", "phase": "z_slow->phase_r2"}


def norm(v, metric, chance=0.0):
    m, s = S[v][SLOW[metric]]
    f = S[v][FULL[metric]][0]
    return max(m - chance, 0) / (f - chance), s / (f - chance)


fig = plt.figure(figsize=(9.8, 2.4))
gs = fig.add_gridspec(1, 4, width_ratios=[1.55, 0.62, 0.62, 1.0], wspace=0.42)

# ---- Panel A: ABSOLUTE probe scores from z_slow, per variant (leak-free) ----
ax = fig.add_subplot(gs[0])
variants = ["NEPA gate+dcor (ours)", "NEPA gate only", "NEPA nogate", "AR + gate", "AR no gate"]
labels = ["Ours", "Gate\nonly", "No\ngate", "AR\n+gate", "AR\nno gate"]
metrics = [("regime", "slow kept: regime acc", C["blue"]),
           ("u", "fast leak: u R2", C["yellow"]),
           ("phase", "fast leak: phase R2", C["aqua"])]
xs = np.arange(len(variants))
for i, (met, name, col) in enumerate(metrics):
    m = [max(S[v][SLOW[met]][0], 0) for v in variants]
    e = [S[v][SLOW[met]][1] for v in variants]
    b = ax.bar(xs + (i - 1) * 0.27, m, 0.24, color=col, label=name,
               yerr=e, error_kw=dict(lw=0.7, capsize=1.5, ecolor=C["ink"]))
    for r, mv, ev in zip(b, m, e):
        ax.text(r.get_x() + r.get_width() / 2, mv + ev + 0.02, f"{mv:.2f}",
                ha="center", fontsize=5.8, color=C["ink"])
ax.set_xticks(xs, labels, fontsize=6.8)
ax.set_ylabel("$z_{slow}$ probe score (absolute)", fontsize=7)
ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.5, 1.0])
ax.legend(fontsize=6.0, frameon=False, ncols=3, loc="lower center",
          bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=1.1)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=C["grid"], lw=0.5); ax.set_axisbelow(True)
ax.set_title("A  What ends up in the slow block (leak-free)", fontsize=8, loc="left", pad=16)

# ---- Panel B: block x factor matrices, ours vs no gate ----
mats, titles = [], ["Ours (gate+dcor)", "No gate"]
for v in ["NEPA gate+dcor (ours)", "NEPA nogate"]:
    rows = []
    for blk in ["z_slow", "z_fast"]:
        r = [max(S[v][f"{blk}->{key}"][0], 0)
             for key in ("regime_acc", "u_r2", "phase_r2")]   # absolute
        rows.append(r)
    mats.append(np.array(rows))
for j, (M, t) in enumerate(zip(mats, titles)):
    ax = fig.add_subplot(gs[1 + j])
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    for a in range(2):
        for b in range(3):
            ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M[a, b] > 0.6 else C["ink"])
    ax.set_xticks(range(3), ["regime\n(slow)", "u\n(fast)", "phase\n(fast)"], fontsize=6.2)
    ax.set_yticks(range(2), ["$z_{slow}$", "$z_{fast}$"], fontsize=7)
    ax.set_title(("B  " if j == 0 else "") + t, fontsize=8, loc="left")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


# ---- Panel C: real data (HAPT) ----
H = {}
for name, pre in [("Gate", "hapt_nepa_g1_d0"), ("No gate", "hapt_nepa_g0_d0"),
                  ("AR+gate", "hapt_ar_g1_d0")]:
    rs = [json.load(open(f)) for f in glob.glob(f"runs_hapt/{pre}_s*.json")]
    H[name] = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
               for k in rs[0] if k != "tag"}
ax = fig.add_subplot(gs[3])
xs = np.arange(3)                                    # absolute z_slow scores
for i, (key_s, name, col) in enumerate([
        ("z_slow->activity_f1", "slow kept (activity F1)", C["blue"]),
        ("z_slow->accmag_r2", "fast leak (acc R2)", C["yellow"])]):
    m = [max(H[v][key_s][0], 0) for v in H]
    e = [H[v][key_s][1] for v in H]
    b = ax.bar(xs + (i - 0.5) * 0.32, m, 0.28, color=col, label=name,
               yerr=e, error_kw=dict(lw=0.7, capsize=1.5, ecolor=C["ink"]))
    for r, mv, ev in zip(b, m, e):
        ax.text(r.get_x() + r.get_width() / 2, mv + ev + 0.03, f"{mv:.2f}",
                ha="center", fontsize=5.8, color=C["ink"])
ax.set_xticks(xs, list(H), fontsize=6.8)
ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.5, 1.0])
ax.legend(fontsize=6.0, frameon=False, ncols=1, loc="upper right",
          bbox_to_anchor=(1.0, 1.02), handlelength=1.1)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=C["grid"], lw=0.5); ax.set_axisbelow(True)
ax.set_title("C  Real data (HAPT, subject-split)", fontsize=8, loc="left", pad=16)
fig.savefig("fig_main.pdf", bbox_inches="tight")
fig.savefig("fig_main.png", bbox_inches="tight")
print("saved 3-panel fig")
