"""Protocol D sweeps — the hyper-parameters the paper picks but never justified.

Three axes, each answering a question a reviewer will ask first:

  lam    xcov weight. The COST CURVE. Every separation number we report sits at
         lam=4 and nothing says what a different price buys. Without this, Part 2
         is one point, not a trade-off.
  tau    the gate threshold. D1's "automatic label-free tau" was retracted
         (2026-08-04) in favour of "a hyper-parameter with a principled init plus
         a sensitivity plateau" — and the plateau was never measured. If there
         is no plateau, that is a limitation we must state rather than discover
         in review.
  d_slow block width. The sharpest alternative explanation left: separation may
         be a block-SIZE effect rather than a gating effect. The random-subspace
         null already controls for width at a fixed 16; this varies it.

HAPT IS included. An earlier version excluded it because its random-16 null is
0.04, but that only disables the EXCLUSION readout -- the cost side (z_full
accuracy) is perfectly measurable, and HAPT is where the gate costs the most
(-0.030 vs ~0 elsewhere), so it is the most informative dataset for a price tag.
tau matters there too: CLAUDE.md section 5 records that HAPT has 0 of 6 horizons
beyond auto-tau and "a monotone preference for smaller tau", which this sweep
answers directly. Read HAPT's exclusion column with the null beside it.

Usage: python3 sweeps.py [lam|tau|dslow] [synth|ptbxl] [n_seed]
Writes runs_v2/sweep_<axis>_<dataset>.json
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hglp"))

import numpy as np

from model import D_SLOW, D_Z
from probes import block_factor, sep_index
from twosided import synth_feats, real_feats, DATASETS

STEMS = ["l1+ema", "nce+ema"]
# tau is swept multiplicatively around each dataset's own default, per protocol D
GRIDS = {"lam": ("lam", [0.0, 1.0, 4.0, 16.0, 64.0]),
         "tau": ("tau", [0.5, 1.0, 2.0, 4.0]),
         "dslow": ("d_slow", [8, 16, 32])}
TAU_DEFAULT = {"synth": 16.0, "ptbxl": 16.0, "hapt": 40.0}


def one(dataset, stem, seed, axis, value):
    lk, tgt = stem.split("+")
    kw = dict(loss_kind=lk, target_enc=tgt, seed=seed, log_every=10 ** 9)
    key, _ = GRIDS[axis]
    kw[key] = TAU_DEFAULT[dataset] * value if axis == "tau" else value
    ds = value if axis == "dslow" else D_SLOW
    if dataset == "synth":
        from train import train
        r = train(**kw)
        f = synth_feats(r["enc"])
    else:
        npz, n_ax, base, static = DATASETS[dataset]
        r = train_real_(npz, n_ax, base, kw)
        f = real_feats(r, static)
    return block_factor(*f, d_slow=ds, seed=seed)


def train_real_(npz, n_ax, base, kw):
    from train_real import train_real
    return train_real(npz, n_ax=n_ax, **{**base, **kw})


def run(axis, dataset, n_seed):
    _, grid = GRIDS[axis]
    out = {}
    for stem in STEMS:
        for v in grid:
            cells = [one(dataset, stem, s, axis, v) for s in range(n_seed)]
            keys = cells[0].keys()
            out[f"{stem}/{v}"] = {k: dict(
                slow=float(np.mean([c[k][0] for c in cells])),
                slow_sd=float(np.std([c[k][0] for c in cells])),
                fast=float(np.mean([c[k][1] for c in cells])),
                fast_sd=float(np.std([c[k][1] for c in cells]))) for k in keys}
            b = out[f"{stem}/{v}"]
            ds = v if axis == "dslow" else D_SLOW
            print(f"  {stem} {axis}={v}: z_slow {b['z_slow']['slow']:.3f}/"
                  f"{b['z_slow']['fast']:+.3f} | z_fast {b['z_fast']['fast']:+.3f} "
                  f"| rand{ds} {b[f'rand{ds}']['fast']:+.3f}", flush=True)
    return out


def report(out, axis, dataset):
    """Cost curve: what each setting buys (exclusion) and what it costs (z_full)."""
    ceil = max(v["z_full"]["fast"] for v in out.values())
    print(f"\n=== {dataset} / {axis} — 얻는 것 vs 잃는 것 ===")
    print(f"{'스템':9}{axis:>7} {'z_slow 느림':>11} {'z_full 느림':>11} "
          f"{'배제 여유':>10} {'SEP':>7}")
    for tag, b in out.items():
        stem, v = tag.split("/")
        ds = int(float(v)) if axis == "dslow" else D_SLOW
        null = b[f"rand{ds}"]["fast"]
        m = {k: (x["slow"], x["fast"]) for k, x in b.items()}
        # sep_index reads rand<D_SLOW>; alias it when d_slow was swept
        m[f"rand{D_SLOW}"] = m[f"rand{ds}"]
        s = sep_index(m, ceil, null)
        print(f"{stem:9}{v:>7} {b['z_slow']['slow']:11.3f} {b['z_full']['slow']:11.3f} "
              f"{null - b['z_slow']['fast']:+10.3f} {s['sep']:7.3f}")


if __name__ == "__main__":
    axis = sys.argv[1] if len(sys.argv) > 1 else "lam"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "synth"
    n_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    assert axis in GRIDS and dataset in TAU_DEFAULT
    os.makedirs("../../runs_v2", exist_ok=True)
    print(f"=== sweep {axis} on {dataset}, {len(GRIDS[axis][1])} values x "
          f"{len(STEMS)} stems x {n_seed} seeds ===", flush=True)
    res = run(axis, dataset, n_seed)
    json.dump(dict(result=res, config=dict(axis=axis, dataset=dataset, n_seed=n_seed,
                                           grid=GRIDS[axis][1],
                                           tau_default=TAU_DEFAULT[dataset])),
              open(f"../../runs_v2/sweep_{axis}_{dataset}.json", "w"), indent=2)
    report(res, axis, dataset)
    print(f"\nsaved runs_v2/sweep_{axis}_{dataset}.json")
