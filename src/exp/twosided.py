"""BLOCK x FACTOR evaluation — replaces the one-sided "z_slow: slow high, leak low".

Why this exists. Every v2 number so far scored ONLY z_slow. That cannot tell real
separation apart from three impostors, all of which score a low leak:

  (1) vacuous     the fast factor was never encoded anywhere (nce+online:
                  HAPT z_fast carries it at 0.163)
  (2) accidental  the subspace happens to sit in low-variance directions where
                  the fast proxy does not live (PCA on HAPT)
  (3) unmeasurable  the fast proxy is so weakly represented that ANY subspace
                  scores low on it, gate or no gate

(1) and (2) are caught by scoring z_fast on BOTH factors. (3) is caught by the
random-subspace null: what an arbitrary block of the SAME width gives on the SAME
embedding. A gate that "excludes" no better than a random rotation excludes
nothing.

Runs the 2x2 gate x xcov cells for both stems plus the two ablation stems, on all
three datasets, and writes the full matrix + the SEP index.

Usage: python3 twosided.py [synth|hapt|ptbxl]
Writes runs_v2/twosided_<dataset>.json
"""
import json
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hglp"))

import numpy as np
import torch

from probes import block_factor, sep_index, encode_all, C_MIN
from model import D_SLOW, D_Z, P, L

SEEDS = [0, 1, 2]
# Synthetic difficulty. The default +-5% is nearly solved before training (a
# training-free FFT classifier reaches 0.784), so the whole 2x2 sits in a regime
# where slow-kept saturates at 0.99 for every cell. GAP=0.01 drops the classical
# baseline to 0.459 and is the setting the mechanism claim should also survive.
GAP = float(os.environ.get("HGLP_GAP", 0.05))
# (loss, target, gate, xcov, blocknorm). 2x2 for both stems + the two loss ablations.
CELLS = ([(lk, "ema", g, x, True) for lk in ("l1", "nce")
          for g, x in ((False, False), (True, False), (False, True), (True, True))]
         + [("reg", "ema", True, True, True), ("nce", "online", True, True, True)])

# B2's per-block LayerNorm privileges the coordinate split even with gate and xcov
# OFF, so "g0_x0" was never a mechanism-free control -- which is why the ungated
# block and the random-16 null do not agree. It also normalises away each block's
# MAGNITUDE, and HAPT's fast proxy IS a magnitude (||acc||). These cells isolate it:
# identical models, one LayerNorm over all of D_Z instead of two per block.
BN_CELLS = [(lk, "ema", g, x, bn) for lk in ("l1", "nce")
            for g, x in ((False, False), (True, True)) for bn in (True, False)]

# The two gate x xcov cells missing from the LN-free world, so the 2x2 can be read
# entirely without per-block LN. "no mechanism" must mean no gate, no xcov AND no
# per-block LN -- LN privileges the coordinate split on its own.
BN22_CELLS = [(lk, "ema", g, x, False) for lk in ("l1", "nce")
              for g, x in ((True, False), (False, True))]


def synth_feats(enc, seed=99, n_win=400, positions=tuple(range(64, 240, 12)), gap=0.05):
    """Held-out synthetic windows -> (full embeddings, regime label, fast factor u)
    split by a WINDOW-LEVEL time cut (a position-level cut leaks: window starts are
    random, so neighbouring windows overlap ~95%)."""
    from datagen import make_dataset
    from train import DEV
    d = make_dataset(300_000, seed=seed, gap=gap)
    x, s, u = d["x"], d["s"], d["u"]
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(x) - L * P - 1, n_win)
    xb = torch.from_numpy(np.stack([x[st:st + L * P].reshape(L, P)
                                    for st in starts])).to(DEV)
    with torch.no_grad():
        Z = torch.cat([enc(xb[i:i + 128]) for i in range(0, len(xb), 128)]).cpu().numpy()
    F_ = np.concatenate([Z[:, a] for a in positions])
    ys = np.concatenate([s[starts + a * P + (P - 1)] for a in positions])
    yu = np.concatenate([u[starts + a * P + (P - 1)] for a in positions])
    wid = np.tile(np.arange(n_win), len(positions))
    cut = np.sort(starts)[n_win // 2]
    tr = np.flatnonzero(starts[wid] + L * P <= cut)
    te = np.flatnonzero(starts[wid] > cut)
    assert not (set(wid[tr]) & set(wid[te])), "window leaked across split"
    return F_[tr], ys[tr], yu[tr], F_[te], ys[te], yu[te]


def real_feats(r, static):
    """Real data -> the same six arrays, from the scored positions only."""
    Z = encode_all(r["enc"], r["Wt"])
    lab, fast = r["lab"], r["fast"]
    if static:                                   # PTB-XL: one label per record
        g = lambda i: (Z[i, -1], lab[i, -1], fast[i, -1])
    else:
        ok = (lab >= 1) & (lab <= 6); ok[:, :C_MIN] = False
        g = lambda i: (Z[i][ok[i]], lab[i][ok[i]] - 1, fast[i][ok[i]])
    a, b = g(r["tr"]), g(r["te"])
    return a[0], a[1], a[2], b[0], b[1], b[2]


DATASETS = {
    "synth": None,
    "hapt": ("../../data/hapt_v2.npz", 3,
             dict(tau=40.0, w=12, dmin=12, dmax=128, min_context=16), False),
    "sleepedf": ("../../data/sleepedf_v2.npz", 3,
                 dict(tau=24.0, w=12, dmin=12, dmax=128, min_context=16, steps=5000), False),
    "ptbxl": ("../../data/ptbxl_v2.npz", 1,
              dict(tau=16.0, w=8, dmin=8, dmax=48, min_context=8), True),
}


def run(which, cells_spec=CELLS):
    out = {}
    for lk, te_, gate, xcov, bn in cells_spec:
        tag = (f"{lk}+{te_}/g{int(gate)}_x{int(xcov)}"
               + ("" if bn else "_noBN"))
        cells = []
        for s in SEEDS:
            if which == "synth":
                from train import train
                r = train(loss_kind=lk, target_enc=te_, seed=s, gate=gate, xcov=xcov,
                          blocknorm=bn, gap=GAP, log_every=10 ** 9)
                f = synth_feats(r["enc"], gap=GAP)
            else:
                from train_real import train_real
                npz, n_ax, kw, static = DATASETS[which]
                r = train_real(npz, n_ax=n_ax, seed=s, gate=gate, xcov=xcov,
                               blocknorm=bn, loss_kind=lk, target_enc=te_,
                               log_every=10 ** 9, **kw)
                f = real_feats(r, static)
            cells.append(block_factor(*f, seed=s))
        keys = cells[0].keys()
        out[tag] = {k: dict(slow=float(np.mean([c[k][0] for c in cells])),
                            slow_sd=float(np.std([c[k][0] for c in cells])),
                            fast=float(np.mean([c[k][1] for c in cells])),
                            fast_sd=float(np.std([c[k][1] for c in cells]))) for k in keys}
        b = out[tag]
        print(f"  {tag:16s} z_slow {b['z_slow']['slow']:.3f}/{b['z_slow']['fast']:+.3f} "
              f"| z_fast {b['z_fast']['slow']:.3f}/{b['z_fast']['fast']:+.3f} "
              f"| rand{D_SLOW} {b[f'rand{D_SLOW}']['slow']:.3f}/{b[f'rand{D_SLOW}']['fast']:+.3f}",
              flush=True)
    return out


def report(out):
    ceil = max(v["z_full"]["fast"] for v in out.values())    # comparison-set ceiling
    print(f"\n{'cell':16s} {'z_slow s/f':>15} {'z_fast s/f':>15} {'rand16 s/f':>15} "
          f"{'incl':>6} {'alloc':>6} {'excl':>6} {'SEP':>6}")
    rank = []
    for tag, b in out.items():
        m = {k: (v["slow"], v["fast"]) for k, v in b.items()}
        stem = tag.split("/")[0]
        base = out.get(f"{stem}/g0_x0")
        s = sep_index(m, ceil, base[f"rand{D_SLOW}"]["fast"] if base else None)
        rank.append((s["sep"], tag))
        print(f"{tag:16s} {m['z_slow'][0]:6.3f}/{m['z_slow'][1]:+.3f} "
              f"{m['z_fast'][0]:6.3f}/{m['z_fast'][1]:+.3f} "
              f"{m[f'rand{D_SLOW}'][0]:6.3f}/{m[f'rand{D_SLOW}'][1]:+.3f} "
              f"{s['inclusion']:6.3f} {s['allocation']:6.3f} {s['exclusion']:6.3f} "
              f"{s['sep']:6.3f}")
    print(f"\n  fast-factor ceiling (best z_full in this set): {ceil:.3f}")
    print("  best by SEP: " + ", ".join(f"{t} ({v:.3f})" for v, t in
                                        sorted(rank, reverse=True)[:3]))
    return ceil


def _selfcheck():
    """A perfectly separated embedding must beat its own random null; a rotated
    copy of the same information must not."""
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 3, n); u = rng.standard_normal(n)
    Z = np.zeros((n, D_Z))
    Z[:, :D_SLOW] = y[:, None] + 0.1 * rng.standard_normal((n, D_SLOW))
    Z[:, D_SLOW:] = u[:, None] + 0.1 * rng.standard_normal((n, D_Z - D_SLOW))
    tr, te = np.arange(n // 2), np.arange(n // 2, n)
    args = (Z[tr], y[tr], u[tr], Z[te], y[te], u[te])
    good = block_factor(*args, seed=0)
    Q = np.linalg.qr(rng.standard_normal((D_Z, D_Z)))[0]
    mixed = block_factor(Z[tr] @ Q, y[tr], u[tr], Z[te] @ Q, y[te], u[te], seed=0)
    assert good["z_slow"][1] < good[f"rand{D_SLOW}"][1] - 0.3, good
    assert good["z_fast"][1] > 0.9 and good["z_slow"][0] > 0.9, good
    assert mixed["z_slow"][1] > mixed[f"rand{D_SLOW}"][1] - 0.1, mixed   # null-like
    gs, ms = (sep_index(d, 1.0)["sep"] for d in (good, mixed))
    assert gs > 0.8 > ms, (gs, ms)
    print(f"selfcheck OK — separated SEP {gs:.3f} vs rotated-mix SEP {ms:.3f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "synth"
    if which == "selfcheck":
        _selfcheck(); sys.exit()
    os.makedirs("../../runs_v2", exist_ok=True)
    mode = sys.argv[2] if len(sys.argv) > 2 else ""
    suf = "" if GAP == 0.05 else f"_gap{GAP:g}"
    spec, tag = {"bn": (BN_CELLS, f"twosided_bn_{which}{suf}"),
                 "bn22": (BN22_CELLS, f"twosided_bn22_{which}{suf}")}.get(
                     mode, (CELLS, f"twosided_{which}{suf}"))
    print(f"=== block x factor + random null: {which}, {len(spec)} cells x {len(SEEDS)} seeds ===")
    res = run(which, spec)
    json.dump(res, open(f"../../runs_v2/{tag}.json", "w"), indent=2)
    report(res)
    print(f"\nsaved runs_v2/{tag}.json")
