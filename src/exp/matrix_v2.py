"""v2 stem matrix: {reg+ema, nce+ema, nce+online} x 3 seeds, on synthetic and
real data. Reports mean±std so the D4 re-decision rests on error bars, not one
seed. Usage:  python3 matrix_v2.py [synth|ptbxl|hapt]
Writes runs_v2/matrix_<dataset>.json
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hglp"))

import json
import os
import sys

import numpy as np

STEMS = [("reg", "ema"), ("nce", "ema"), ("nce", "online")]
SEEDS = [0, 1, 2]


def agg(vals):
    return float(np.mean(vals)), float(np.std(vals))


def run_synth():
    from train import train
    from stems_synth import probe
    res = {}
    for lk, te in STEMS:
        tag = f"{lk}+{te}"
        cells = []
        for s in SEEDS:
            r = train(loss_kind=lk, target_enc=te, seed=s, log_every=10**9)
            cells.append(probe(r["enc"]))
            print(f"  {tag} seed{s}: slow-kept {cells[-1]['z_slow']['regime_acc']:.3f} "
                  f"leak {cells[-1]['z_slow']['leak_u_r2']:+.3f}", flush=True)
        res[tag] = {b: {m: agg([c[b][m] for c in cells])
                        for m in ("regime_acc", "leak_u_r2", "rankme")}
                    for b in ("z_slow", "z_fast", "z_full")}
    return res, ("regime_acc", "leak_u_r2")


def run_real(npz, n_ax, kw, static=False):
    from train_real import train_real, probe
    res = {}
    for lk, te in STEMS:
        tag = f"{lk}+{te}"
        cells = []
        for s in SEEDS:
            r = train_real(npz, n_ax=n_ax, seed=s, loss_kind=lk, target_enc=te,
                           log_every=10**9, **kw)
            if static:
                from run_ptbxl import probe_static
                p = probe_static(r)
                p = {b: dict(regime_acc=v["slow_kept_f1"], leak_u_r2=v["leak_r2"],
                             rankme=v["rankme"]) for b, v in p.items()}
            else:
                p = {b: probe(r, b, max_samples=60000) for b in ("z_slow", "z_fast", "z_full")}
                p = {b: dict(regime_acc=v["slow_kept_f1"], leak_u_r2=v["leak_r2"],
                             rankme=v["rankme"]) for b, v in p.items()}
            cells.append(p)
            print(f"  {tag} seed{s}: slow-kept {p['z_slow']['regime_acc']:.3f} "
                  f"leak {p['z_slow']['leak_u_r2']:+.3f}", flush=True)
        res[tag] = {b: {m: agg([c[b][m] for c in cells])
                        for m in ("regime_acc", "leak_u_r2", "rankme")}
                    for b in ("z_slow", "z_fast", "z_full")}
    return res, ("regime_acc", "leak_u_r2")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "synth"
    os.makedirs("../../runs_v2", exist_ok=True)
    print(f"=== v2 stem matrix: {which} (3 stems x {len(SEEDS)} seeds) ===")
    if which == "synth":
        res, _ = run_synth()
    elif which == "ptbxl":
        res, _ = run_real("../../data/ptbxl_v2.npz", 1,
                          dict(tau=16.0, w=8, dmin=8, dmax=48, min_context=8), static=True)
    else:
        res, _ = run_real("../../data/hapt_v2.npz", 3,
                          dict(tau=40.0, w=12, dmin=12, dmax=128, min_context=16))
    json.dump(res, open(f"../../runs_v2/matrix_{which}.json", "w"), indent=2)

    print(f"\n=== {which}: mean ± std over {len(SEEDS)} seeds ===")
    print(f"{'stem':12s} {'slow-kept':>16} {'leak (z_slow)':>17} {'z_fast carries':>16}")
    for tag, r in res.items():
        sk, sks = r["z_slow"]["regime_acc"]; lk, lks = r["z_slow"]["leak_u_r2"]
        zf, zfs = r["z_fast"]["leak_u_r2"]
        print(f"{tag:12s} {sk:8.3f}±{sks:.3f} {lk:+9.3f}±{lks:.3f} {zf:9.3f}±{zfs:.3f}")
    print(f"saved runs_v2/matrix_{which}.json")
