"""Gate x xcov ablation (2x2) — the experiment the central claim needs.

Every v2 result so far had gate AND xcov both ON, and compared z_slow against
z_full OF THE SAME GATED MODEL. That can show "the two blocks differ", but it
CANNOT show "the gate creates the separation". This runs the 2x2:

    gate=0 xcov=0   no mechanism (ungated control — the real baseline for z_full)
    gate=1 xcov=0   gate only
    gate=0 xcov=1   xcov only
    gate=1 xcov=1   both (our method)

Reads: does slow-kept stay high while leak drops, and which term causes it?
Usage: python3 ablation.py [synth|hapt|ptbxl] [stem]     stem = reg+ema|nce+ema
Writes runs_v2/ablation_<dataset>_<stem>.json
"""
import json
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hglp"))

import numpy as np

CELLS = [(False, False), (True, False), (False, True), (True, True)]
SEEDS = [0, 1, 2]


def run_synth(loss_kind, target_enc):
    from train import train
    from stems_synth import probe
    out = {}
    for gate, xcov in CELLS:
        tag = f"g{int(gate)}_x{int(xcov)}"
        cells = []
        for s in SEEDS:
            r = train(loss_kind=loss_kind, target_enc=target_enc, seed=s,
                      gate=gate, xcov=xcov, log_every=10 ** 9)
            cells.append(probe(r["enc"]))
        out[tag] = {b: {m: (float(np.mean([c[b][m] for c in cells])),
                            float(np.std([c[b][m] for c in cells])))
                        for m in ("regime_acc", "leak_u_r2", "rankme")}
                    for b in ("z_slow", "z_fast", "z_full")}
        z = out[tag]["z_slow"]
        print(f"  {tag}: slow-kept {z['regime_acc'][0]:.3f}±{z['regime_acc'][1]:.3f} "
              f"| leak {z['leak_u_r2'][0]:+.3f}±{z['leak_u_r2'][1]:.3f}", flush=True)
    return out


def run_real(npz, n_ax, kw, loss_kind, target_enc):
    from train_real import train_real, probe
    out = {}
    for gate, xcov in CELLS:
        tag = f"g{int(gate)}_x{int(xcov)}"
        cells = []
        for s in SEEDS:
            r = train_real(npz, n_ax=n_ax, seed=s, gate=gate, xcov=xcov,
                           loss_kind=loss_kind, target_enc=target_enc,
                           log_every=10 ** 9, **kw)
            cells.append({b: probe(r, b) for b in ("z_slow", "z_fast", "z_full")})
        out[tag] = {b: {m: (float(np.mean([c[b][m] for c in cells])),
                            float(np.std([c[b][m] for c in cells])))
                        for m in ("slow_kept_f1", "leak_r2", "rankme")}
                    for b in ("z_slow", "z_fast", "z_full")}
        z = out[tag]["z_slow"]
        print(f"  {tag}: slow-kept {z['slow_kept_f1'][0]:.3f}±{z['slow_kept_f1'][1]:.3f} "
              f"| leak {z['leak_r2'][0]:+.3f}±{z['leak_r2'][1]:.3f}", flush=True)
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "synth"
    stem = sys.argv[2] if len(sys.argv) > 2 else "nce+ema"
    lk, te = stem.split("+")
    os.makedirs("../../runs_v2", exist_ok=True)
    print(f"=== gate x xcov ablation: {which}, stem {stem}, {len(SEEDS)} seeds ===")
    if which == "synth":
        res = run_synth(lk, te); sk, lkm = "regime_acc", "leak_u_r2"
    elif which == "ptbxl":
        res = run_real("../../data/ptbxl_v2.npz", 1,
                       dict(tau=16.0, w=8, dmin=8, dmax=48, min_context=8), lk, te)
        sk, lkm = "slow_kept_f1", "leak_r2"
    else:
        res = run_real("../../data/hapt_v2.npz", 3,
                       dict(tau=40.0, w=12, dmin=12, dmax=128, min_context=16), lk, te)
        sk, lkm = "slow_kept_f1", "leak_r2"
    json.dump(res, open(f"../../runs_v2/ablation_{which}_{stem}.json", "w"), indent=2)

    print(f"\n=== {which} / {stem} — does the GATE create the separation? ===")
    print(f"{'cell':10s} {'slow-kept':>16} {'leak (z_slow)':>17} {'z_fast carries':>16}")
    for tag, r in res.items():
        a, b = r["z_slow"][sk], r["z_slow"][lkm]
        c = r["z_fast"][lkm]
        print(f"{tag:10s} {a[0]:8.3f}±{a[1]:.3f} {b[0]:+9.3f}±{b[1]:.3f} {c[0]:9.3f}±{c[1]:.3f}")
    base, both = res["g0_x0"]["z_slow"], res["g1_x1"]["z_slow"]
    print(f"\n  leak: no-mechanism {base[lkm][0]:+.3f} -> both {both[lkm][0]:+.3f} "
          f"(Δ {both[lkm][0]-base[lkm][0]:+.3f})")
    print(f"  slow-kept: {base[sk][0]:.3f} -> {both[sk][0]:.3f} "
          f"(Δ {both[sk][0]-base[sk][0]:+.3f})")
