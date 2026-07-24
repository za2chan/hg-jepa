"""Paper figure set (column-width, vertical layouts).

fig_method.pdf : architecture + gate schedule (the missing method figure)
fig_mech.pdf   : mechanism results on synthetic (variants incl. CPC, block
                 matrices, width sweep + linear-vs-MLP honesty)
fig_real.pdf   : real data (HAPT, PTB-XL) + XJTU negative (life probe)

Colors follow the metric identity everywhere:
  blue = slow info kept, yellow = fast leak (u/primary), aqua = fast leak (phase).
"""
import glob
import json

import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np

BLUE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"
INK, MUTED, GRID, SURF = "#1f2937", "#6b7280", "#e5e7eb", "#f4f4f2"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.6, "figure.dpi": 200})
W = 3.4  # single column width (in)


def stats(pattern, drop=("tag",)):
    rs = [json.load(open(f)) for f in glob.glob(pattern)]
    return {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
            for k in rs[0] if k not in drop and isinstance(rs[0][k], float)}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def bars(ax, groups, series, gap=0.26, w=0.23, fs=6.0):
    """groups: x labels; series: list of (name, color, means, errs)."""
    xs = np.arange(len(groups))
    n = len(series)
    for i, (name, col, m, e) in enumerate(series):
        b = ax.bar(xs + (i - (n - 1) / 2) * gap, m, w, color=col, label=name,
                   yerr=e, error_kw=dict(lw=0.7, capsize=1.5, ecolor=INK))
        for r, mv, ev in zip(b, m, e):
            ax.text(r.get_x() + r.get_width() / 2, mv + (ev or 0) + 0.02,
                    f"{mv:.2f}", ha="center", fontsize=fs, color=INK)
    ax.set_xticks(xs, groups, fontsize=6.6)
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0, 0.5, 1.0])
    style(ax)


# ================= Fig 1: method =================
fig, (a0, a1) = plt.subplots(2, 1, figsize=(W, 3.3), height_ratios=[2.1, 1.0],
                             gridspec_kw=dict(hspace=0.55))

a0.set_xlim(0, 10); a0.set_ylim(0, 10); a0.axis("off")
a0.set_title("A  Horizon-gated latent prediction", fontsize=8, loc="left")

def box(ax, x, y, w, h, text, fc, ec=MUTED, fs=6.6, tc=INK):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                   fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc)

def arrow(ax, x0, y0, x1, y1, **kw):
    ax.annotate("", (x1, y1), (x0, y0),
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9, **kw))

# input -> encoder -> z split
box(a0, 0.1, 6.6, 2.0, 1.5, "past patches\n$x_{\\leq t}$", SURF)
box(a0, 2.9, 6.6, 2.2, 1.5, "causal\nencoder", "#dbe9f9")
arrow(a0, 2.1, 7.35, 2.9, 7.35)
# z blocks
a0.add_patch(mp.Rectangle((5.9, 7.1), 1.15, 1.0, fc=BLUE, ec="white", lw=1.2))
a0.add_patch(mp.Rectangle((7.05, 7.1), 2.3, 1.0, fc=YELLOW, ec="white", lw=1.2))
a0.text(6.47, 7.6, "$z_{slow}$", ha="center", va="center", fontsize=6.6, color="white")
a0.text(8.2, 7.6, "$z_{fast}$", ha="center", va="center", fontsize=6.6, color=INK)
a0.text(7.6, 8.5, "$z_t \\in \\mathbb{R}^{64}$ (16 | 48)", ha="center", fontsize=6.0, color=MUTED)
arrow(a0, 5.1, 7.35, 5.9, 7.6)
# gate on fast path
box(a0, 7.55, 4.6, 1.6, 1.2, "$\\times\\, g(\\Delta)$\ngate", "#fdf1d8")
arrow(a0, 8.2, 7.1, 8.33, 5.85)
# predictor
box(a0, 3.4, 3.3, 2.6, 1.5, "predictor\n$P(\\cdot,\\Delta)$", "#dbe9f9")
arrow(a0, 6.1, 7.1, 4.6, 4.85)            # slow path (always open)
arrow(a0, 7.9, 4.6, 6.2, 4.35)            # gated fast path
a0.text(5.75, 5.55, "slow path:\nalways open", fontsize=5.4, color=BLUE, ha="left")
# target encoder (same color as online encoder: it IS the encoder, EMA copy)
box(a0, 0.1, 3.3, 1.7, 1.5, "target\nencoder", "#dbe9f9")
a0.text(0.95, 2.95, "(EMA copy)", fontsize=5.2, color=MUTED, ha="center")
a0.add_patch(mp.Rectangle((2.05, 3.55), 0.85, 1.0, fc=SURF, ec=MUTED, lw=0.8))
a0.text(2.47, 4.05, "$\\bar z_{t+\\Delta}$", ha="center", va="center",
        fontsize=6.2, color=INK)
arrow(a0, 1.8, 4.05, 2.05, 4.05)
a0.annotate("", (2.9, 4.05), (3.4, 4.05),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
a0.text(3.15, 4.55, "$L_2$ /\nInfoNCE", fontsize=5.4, color=MUTED, ha="center")
# dcor between blocks
a0.annotate("", (7.05, 7.0), (6.6, 7.0),
            arrowprops=dict(arrowstyle="<->", color=VIOLET, lw=0.9))
a0.text(6.8, 6.45, "dcor $\\lambda$", fontsize=5.6, color=VIOLET, ha="center")
# caption line
a0.text(0.1, 1.6, "Long horizons see only $z_{slow}$ $\\Rightarrow$ slow factors are\n"
                  "assigned there (Prop. 1); bottleneck + dcor exclude fast (Prop. 2).",
        fontsize=6.2, color=INK, va="top")

# gate schedule
d = np.linspace(0, 140, 400)
tau, w_ = 16, 4
a1.plot(d, 1 / (1 + np.exp(-(tau - d) / w_)), color=YELLOW, lw=1.8)
for off in [1, 4, 16, 64, 128]:
    a1.axvline(off, color=GRID, lw=0.6, zorder=0)
    a1.text(off, 1.06, str(off), ha="center", fontsize=5.6, color=MUTED)
a1.axvline(tau, color=INK, lw=0.8, ls="--")
a1.text(tau + 3, 0.75, "$\\tau = c\\,T_{ac}$\n(label-free)", fontsize=6.0, color=INK)
a1.set_xlabel("horizon $\\Delta$ (patches, trained set marked)", fontsize=6.6)
a1.set_ylabel("gate $g(\\Delta)$", fontsize=6.6)
a1.set_ylim(-0.04, 1.18); a1.set_yticks([0, 1])
a1.set_xlim(0, 140)
style(a1)
a1.set_title("B  Gate schedule: fast block fades beyond $\\tau$", fontsize=8, loc="left")
fig.savefig("fig_method.pdf", bbox_inches="tight")
fig.savefig("fig_method.png", bbox_inches="tight")

# ================= Fig 2: mechanism =================
V = {"Ours\n(JEPA)": "runs/nepa_g1_d1_s*_tau16_ds16_lam4.json",
     "Ours\n(CPC)": "runs/cpc_g1_d1_s*_tau16_ds16_lam4.json",
     "Gate\nonly": "runs/nepa_g1_d0_s*_tau16_ds16_lam4.json",
     "dcor\nonly": "runs/nepa_g0_d1_s*_tau16_ds16_lam4.json",
     "No\ngate": "runs/nepa_g0_d0_s*_tau16_ds16_lam4.json",
     "AR\n+gate": "runs/ar_g1_d0_s*_tau16_ds16_lam4.json"}
S = {k: stats(p) for k, p in V.items()}

fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(W, 6.4),
                                 gridspec_kw=dict(hspace=0.62))
keys = [("z_slow->regime_acc", "slow kept: regime acc", BLUE),
        ("z_slow->u_r2", "fast leak: $u$ $R^2$", YELLOW),
        ("z_slow->phase_r2", "fast leak: phase $R^2$", AQUA)]
series = [(name, col,
           [max(S[v][k][0], 0) for v in V],
           [S[v][k][1] for v in V]) for k, name, col in keys]
bars(a0, list(V), series, gap=0.28, w=0.25, fs=5.4)
a0.legend(fontsize=5.8, frameon=False, ncols=3, loc="lower center",
          bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=1.1)
a0.set_ylabel("$z_{slow}$ probe (absolute)", fontsize=6.8)
a0.set_title("A  Only gate+dcor separates; latent target required", fontsize=8,
             loc="left", pad=24)

# B: block x factor matrices
a1.axis("off")
a1.set_title("B  Block $\\times$ factor probe matrix", fontsize=8, loc="left")
for j, (v, t) in enumerate([("Ours\n(JEPA)", "ours (gate+dcor)"), ("No\ngate", "no gate")]):
    sub = a1.inset_axes([0.06 + j * 0.52, 0.0, 0.40, 0.78])
    M = np.array([[max(S[v][f"{b}->{k}"][0], 0) for k in
                   ("regime_acc", "u_r2", "phase_r2")] for b in ("z_slow", "z_fast")])
    sub.imshow(M, cmap="Blues", vmin=0, vmax=1)
    for r in range(2):
        for c in range(3):
            sub.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center", fontsize=6.4,
                     color="white" if M[r, c] > 0.6 else INK)
    sub.set_xticks(range(3), ["regime", "$u$", "phase"], fontsize=6.0)
    sub.set_yticks(range(2), ["$z_{slow}$", "$z_{fast}$"], fontsize=6.2)
    sub.set_title(t, fontsize=6.6)
    sub.tick_params(length=0)
    for s in sub.spines.values():
        s.set_visible(False)

# C: width sweep + honesty
dz = [64, 128, 256]
fixed = [stats(f"runs/mech_dz{z}_ds16_g1_d1_t0_s*.json") if z != 64
         else stats("runs/mech_dz64_ds16_g1_d1_t0_s*.json") for z in dz]
prop_map = {64: "ds16", 128: "ds32", 256: "ds64"}
prop = [stats(f"runs/mech_dz{z}_{prop_map[z]}_g1_d1_t0_s*.json") for z in dz]
a2.plot(dz, [f["z_slow->u"][0] for f in fixed], "o-", color=BLUE, lw=1.6, ms=4,
        label="fixed 16-dim slow block")
a2.plot(dz, [p["z_slow->u"][0] for p in prop], "s--", color=RED, lw=1.6, ms=4,
        label="block scaled with width")
for z, f, p in zip(dz, fixed, prop):
    a2.text(z, f["z_slow->u"][0] - 0.09, f"{f['z_slow->u'][0]:.2f}", ha="center",
            fontsize=5.8, color=BLUE)
    a2.text(z, p["z_slow->u"][0] + 0.05, f"{p['z_slow->u'][0]:.2f}", ha="center",
            fontsize=5.8, color=RED)
nl = stats("runs/nl_nepa_g1_d1_s*_tau16_ds16_lam4.json")
a2.axhline(nl["mlp_z_slow->u_r2"][0], color=MUTED, lw=0.9, ls=":")
a2.text(250, nl["mlp_z_slow->u_r2"][0] + 0.03,
        f"MLP probe on ours: {nl['mlp_z_slow->u_r2'][0]:.2f} (nonlinear leak)",
        fontsize=5.8, color=MUTED, ha="right")
a2.set_xscale("log", base=2); a2.set_xticks(dz, [str(z) for z in dz], fontsize=6.6)
a2.set_xlabel("embedding width $d_z$", fontsize=6.8)
a2.set_ylabel("fast leak: $u$ $R^2$ in $z_{slow}$", fontsize=6.8)
a2.set_ylim(0, 0.95)
a2.legend(fontsize=6.0, frameon=False, loc="center left")
style(a2)
a2.set_title("C  Exclusion rides on absolute block size (and is linear)",
             fontsize=8, loc="left")
fig.savefig("fig_mech.pdf", bbox_inches="tight")
fig.savefig("fig_mech.png", bbox_inches="tight")

# ================= Fig 3: real data =================
fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(W, 5.6),
                                 gridspec_kw=dict(hspace=0.72))
H = {"Gate": "runs_hapt/hapt_nepa_g1_d0_s*.json",
     "Gate\n+dcor": "runs_hapt/hapt_nepa_g1_d1_s*.json",
     "No gate": "runs_hapt/hapt_nepa_g0_d0_s*.json",
     "AR\n+gate": "runs_hapt/hapt_ar_g1_d0_s*.json"}
Hs = {k: stats(p) for k, p in H.items()}
bars(a0, list(H), [
    ("slow kept: activity F1", BLUE,
     [max(Hs[v]["z_slow->activity_f1"][0], 0) for v in H],
     [Hs[v]["z_slow->activity_f1"][1] for v in H]),
    ("fast leak: acc-mag $R^2$", YELLOW,
     [max(Hs[v]["z_slow->accmag_r2"][0], 0) for v in H],
     [Hs[v]["z_slow->accmag_r2"][1] for v in H])], gap=0.22, w=0.2, fs=5.4)
a0.legend(fontsize=5.8, frameon=False, ncols=2, loc="lower center",
          bbox_to_anchor=(0.5, 0.99), handlelength=1.1)
a0.set_title("A  HAPT (held-out subjects): dcor hurts", fontsize=8,
             loc="left", pad=16)

P = {"Gate": "runs_ptbxl/ptbxl_nepa_g1_d0_s*.json",
     "Gate\n+dcor": "runs_ptbxl/ptbxl_nepa_g1_d1_s*.json",
     "No gate": "runs_ptbxl/ptbxl_nepa_g0_d0_s*.json",
     "AR\n+gate": "runs_ptbxl/ptbxl_ar_g1_d0_s*.json"}
Ps = {k: stats(p) for k, p in P.items()}
bars(a1, list(P), [
    ("slow kept: norm F1", BLUE,
     [max(Ps[v]["z_slow->norm_f1"][0], 0) for v in P],
     [Ps[v]["z_slow->norm_f1"][1] for v in P]),
    ("fast leak: ECG $R^2$", YELLOW,
     [max(Ps[v]["z_slow->ecg_r2"][0], 0) for v in P],
     [Ps[v]["z_slow->ecg_r2"][1] for v in P])], gap=0.22, w=0.2, fs=5.4)
a1.legend(fontsize=5.8, frameon=False, ncols=2, loc="lower center",
          bbox_to_anchor=(0.5, 0.99), handlelength=1.1)
a1.set_title("B  PTB-XL (held-out patients): dcor helps", fontsize=8,
             loc="left", pad=16)

# C: XJTU negative — life (RUL) probe, held-out bearings, learned vs classical
Xs = stats("runs_am/am_nepa_g1_d1_s*.json")
hb = json.load(open("runs_am/hilbert_baseline.json"))
names = ["$z_{slow}$\n(ours)", "$z_{full}$", "Hilbert\nenvelope", "low-pass"]
vals = [Xs["z_slow->life_r2"][0], Xs["z_full->life_r2"][0],
        hb["hilbert->life_r2"], hb["lowpass->life_r2"]]
cols = [BLUE, MUTED, RED, RED]
xs = np.arange(len(names))
b = a2.bar(xs, vals, 0.6, color=cols)
for r, v in zip(b, vals):
    a2.text(r.get_x() + r.get_width() / 2,
            v + (0.05 if v >= 0 else -0.12), f"{v:.2f}",
            ha="center", fontsize=6.0, color=INK)
a2.axhline(0, color=INK, lw=0.7)
a2.set_xticks(xs, names, fontsize=6.4)
a2.set_ylabel("health/RUL probe $R^2$", fontsize=6.8)
a2.set_ylim(-1.3, 0.5)
style(a2)
a2.set_title("C  XJTU negative: no timescale gap; all methods weak,\n"
             "$z_{slow}$ carries what little exists", fontsize=7.4, loc="left")
fig.savefig("fig_real.pdf", bbox_inches="tight")
fig.savefig("fig_real.png", bbox_inches="tight")
print("saved fig_method, fig_mech, fig_real (column-width, vertical)")