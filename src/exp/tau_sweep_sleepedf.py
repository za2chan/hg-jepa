"""tau sensitivity on Sleep-EDF (user-approved 2026-08-06).

tau is reported as a HYPERPARAMETER with a sensitivity plateau, not as an
automatic estimate -- consistent with how HAPT is already reported (the
automatic-tau claim was withdrawn). The default tau=24 is not arbitrary: it is
D1's rule applied to the energy-derived series of the EOG fast proxy
(T_ac 4 patches, u-scale x2 for a squared series, c=3.0 -> 24). The literal
min-over-all-derived-series form of D1 picks raw EOG (T_ac 1 patch -> tau 3),
which puts tau below dmin so the gate never opens; that is reported, not hidden.

python3 src/exp/tau_sweep_sleepedf.py  ->  runs_v2/tau_sweep_sleepedf.json + .png
"""
import json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "hglp"))
from train_real import train_real, probe                      # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
NPZ = str(ROOT / "data" / "sleepedf_v2.npz")
OUT = ROOT / "runs_v2"
TAUS = [8, 16, 24, 32, 48]
SEEDS = [0, 1]
STEPS = 5000                       # 2500 (HAPT's value) leaves z_fast undertrained here
CFG = dict(n_ax=3, w=12, dmin=12, dmax=128, W_gate=4.0,
           loss_kind="nce", target_enc="ema", log_every=100000)


def gate_stats(tau, dmin=12, dmax=128, W=4.0, n=300000):
    d = np.exp(np.random.default_rng(0).uniform(np.log(dmin), np.log(dmax), n))
    g = 1 / (1 + np.exp(-(tau - d) / W))
    return dict(mean_g=float(g.mean()), frac_open=float((g > .9).mean()),
                frac_closed=float((g < .1).mean()))


def main():
    res = {"config": dict(CFG, steps=STEPS, seeds=SEEDS, npz="sleepedf_v2.npz"),
           "gate": {}, "cells": []}
    for tau in TAUS:
        res["gate"][str(tau)] = gate_stats(tau)
        for seed in SEEDS:
            r = train_real(NPZ, seed=seed, steps=STEPS, tau=float(tau), **CFG)
            row = {"tau": tau, "seed": seed}
            for b in ("z_slow", "z_fast", "z_full"):
                p = probe(r, b)
                row[b] = {k: float(p[k]) for k in ("slow_kept_f1", "leak_r2", "rankme")}
            res["cells"].append(row)
            print(f"tau {tau:3d} seed {seed} | slow kept {row['z_slow']['slow_kept_f1']:.3f} "
                  f"leak {row['z_slow']['leak_r2']:.3f} | fast leak {row['z_fast']['leak_r2']:.3f} "
                  f"| full kept {row['z_full']['slow_kept_f1']:.3f} leak {row['z_full']['leak_r2']:.3f}",
                  flush=True)

    agg = {}
    for tau in TAUS:
        c = [x for x in res["cells"] if x["tau"] == tau]
        agg[str(tau)] = {b: {k: [float(np.mean([x[b][k] for x in c])),
                               float(np.std([x[b][k] for x in c]))]
                            for k in ("slow_kept_f1", "leak_r2", "rankme")}
                         for b in ("z_slow", "z_fast", "z_full")}
    res["agg"] = agg
    OUT.mkdir(exist_ok=True)
    (OUT / "tau_sweep_sleepedf.json").write_text(json.dumps(res, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for b, col in [("z_slow", "#1a5fb4"), ("z_fast", "#e8a33d"), ("z_full", "#9aa0a6")]:
        for j, k in enumerate(["slow_kept_f1", "leak_r2"]):
            m = [agg[str(t)][b][k][0] for t in TAUS]
            s = [agg[str(t)][b][k][1] for t in TAUS]
            ax[j].errorbar(TAUS, m, yerr=s, fmt="o-", color=col, capsize=3, label=b)
    ax[0].set_ylabel("sleep-stage macro-F1 $\\uparrow$")
    ax[1].set_ylabel("fast-proxy leak $R^2$")
    for a, t in zip(ax[:2], ["slow factor kept", "fast factor leak"]):
        a.set_xlabel(r"$\tau$ (patches)"); a.set_xscale("log"); a.set_title(t, fontsize=10)
        a.axvline(24, color="#c01c28", ls="--", lw=1, alpha=.7)
        a.grid(alpha=.25, lw=.6); a.set_axisbelow(True); a.legend(fontsize=8)
    g = [res["gate"][str(t)] for t in TAUS]
    ax[2].plot(TAUS, [x["frac_open"] for x in g], "o-", color="#3a9e6e", label="gate open (g>0.9)")
    ax[2].plot(TAUS, [x["frac_closed"] for x in g], "s-", color="#c01c28", label="gate closed (g<0.1)")
    ax[2].axhline(0.405, color="#888", ls=":", lw=1, label="HAPT balance")
    ax[2].axvline(24, color="#c01c28", ls="--", lw=1, alpha=.7)
    ax[2].set_xscale("log"); ax[2].set_xlabel(r"$\tau$ (patches)")
    ax[2].set_ylabel("fraction of sampled horizons")
    ax[2].legend(fontsize=8); ax[2].set_title("gate duty cycle", fontsize=10)
    ax[2].grid(alpha=.25, lw=.6); ax[2].set_axisbelow(True)
    fig.suptitle(r"Sleep-EDF: $\tau$ is a hyperparameter with a sensitivity plateau "
                 r"(default $\tau=24$, dashed)", fontsize=11, y=1.03)
    fig.tight_layout(); fig.savefig(OUT / "tau_sweep_sleepedf.png", dpi=180, bbox_inches="tight")
    print("\nwrote runs_v2/tau_sweep_sleepedf.json + .png")


if __name__ == "__main__":
    main()
