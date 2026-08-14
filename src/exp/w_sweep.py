"""The target-window sweep the protocol planned and never ran.

w sets the target encoder's receptive field, so it decides how much of the
transient factor is averaged away inside the target itself. The three datasets
use 8 / 8 / 12, which is 1.53 / 0.91 / 0.42 times each one's T_ac -- not a
constant ratio, and never justified.

WHY THE PLANNED SWEEP CANNOT RUN. The protocol asked for {1, 2, 4} x T_ac. Both
trainers assert dmin >= w, and handoff 6.8 requires dmin < tau or z_mix receives
no gradient at all. Together these bound w above by tau, and 2x T_ac already
exceeds tau on PTB-XL and HAPT while 4x exceeds it everywhere. The sweep was
impossible as specified, which is presumably why it was never run. This file
sweeps the feasible range instead and includes w = tau as the endpoint, so the
breakdown is measured rather than asserted.

The confound is reported, not hidden: raising w raises dmin, which shrinks the
share of sampled horizons below tau. That share is recorded per cell, because a
change in SEP could come from the target window or from z_mix losing gradient,
and the two must be told apart.

Ordering is seed-major so an interrupted run still covers every condition at
fewer seeds rather than some conditions at three. Results are appended to the
JSON after every cell and completed cells are skipped on restart.

python3 src/exp/w_sweep.py -> runs_v2/w_sweep.json
"""
import json, pathlib, sys
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "hglp"))
sys.path.insert(0, str(HERE.parent))
from probes import block_factor, sep_index                            # noqa: E402
from mlp_probe_check import feats_synth, feats_real                   # noqa: E402

ROOT = HERE.parents[2]
OUT = ROOT / "runs_v2" / "w_sweep.json"
SEEDS = (0, 1, 2)
STEMS = ("nce", "l1")
# w values run from well below the current setting up to tau, the hard ceiling.
GRID = {
    "synth": dict(npz=None, n_ax=None, static=False, tau=16.0, dmax=128,
                  tac=8 / 1.53, ws=(4, 6, 8, 12, 16), kw=dict(gap=0.03)),
    "ptbxl": dict(npz=ROOT / "data/ptbxl_v2.npz", n_ax=1, static=True, tau=16.0,
                  dmax=48, tac=8 / 0.91, ws=(4, 6, 8, 12, 16),
                  kw=dict(tau=16.0, dmax=48, min_context=8)),
    "hapt": dict(npz=ROOT / "data/hapt_v2.npz", n_ax=3, static=False, tau=40.0,
                 dmax=128, tac=12 / 0.42, ws=(8, 12, 20, 28, 40),
                 kw=dict(tau=40.0, dmax=128, min_context=16)),
}


def frac_below_tau(dmin, dmax, tau, L=256, cmin=16, n=200_000, seed=0):
    """Share of sampled horizons that leave the gate open. Log-uniform in
    [dmin, min(dmax, L-1-t)], matching train.py / train_real.py."""
    rng = np.random.default_rng(seed)
    t = rng.integers(cmin, max(cmin + 1, L - dmin), n)
    hi = np.minimum(dmax, L - 1 - t).astype(float)
    ok = hi >= dmin
    d = np.exp(rng.uniform(np.log(dmin), np.log(hi[ok])))
    return float(np.mean(d < tau))


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(res):
    OUT.write_text(json.dumps(res, indent=1))


def cell(name, cfg, w, stem, seed):
    kw = dict(cfg["kw"]); kw.update(w=w, dmin=w)
    if name == "synth":
        from train import train
        r = train(loss_kind=stem, target_enc="ema", seed=seed, gate=True,
                  xcov=True, lam=4.0, log_every=10 ** 9, **kw)
        a, b = feats_synth(r["enc"], gap=kw["gap"])
    else:
        from train_real import train_real
        r = train_real(str(cfg["npz"]), n_ax=cfg["n_ax"], seed=seed, loss_kind=stem,
                       target_enc="ema", gate=True, xcov=True, lam=4.0,
                       log_every=10 ** 9, **kw)
        a, b = feats_real(r, cfg["static"])
    bf = block_factor(*a, *b, seed=0)
    s = sep_index(bf)
    return dict(w=w, w_over_tac=w / cfg["tac"], seed=seed, stem=stem,
                inclusion=s["inclusion"], allocation=s["allocation"],
                exclusion=s["exclusion"], sep=s["sep"],
                ceiling=float(bf["z_full"][1]),
                frac_below_tau=frac_below_tau(w, cfg["dmax"], cfg["tau"]))


def main():
    res = load()
    total = len(SEEDS) * sum(len(c["ws"]) for c in GRID.values()) * len(STEMS)
    done = 0
    for seed in SEEDS:                       # seed-major: full coverage first
        for name, cfg in GRID.items():
            for w in cfg["ws"]:
                for stem in STEMS:
                    key = f"{name}/w{w}/{stem}/s{seed}"
                    done += 1
                    if key in res:
                        continue
                    print(f"[{done}/{total}] {key} ...", flush=True)
                    try:
                        res[key] = cell(name, cfg, w, stem, seed)
                        r = res[key]
                        print(f"    SEP {r['sep']:.3f}  incl {r['inclusion']:.3f}  "
                              f"alloc {r['allocation']:.3f}  excl {r['exclusion']:.3f}  "
                              f"ceil {r['ceiling']:.3f}  "
                              f"Δ<τ {r['frac_below_tau']:.0%}", flush=True)
                    except Exception as e:
                        res[key] = dict(error=f"{type(e).__name__}: {e}")
                        print(f"    FAILED {type(e).__name__}: {e}", flush=True)
                    save(res)

    print("\n=== 요약 (시드 평균) ===")
    for name, cfg in GRID.items():
        print(f"\n{name}   (τ={cfg['tau']:.0f}, T_ac≈{cfg['tac']:.1f})")
        print(f"{'w':>4}{'w/T_ac':>8}{'Δ<τ':>7}" +
              "".join(f"{s+' SEP':>10}{s+' incl':>11}" for s in STEMS))
        for w in cfg["ws"]:
            cells = {s: [res[f"{name}/w{w}/{s}/s{d}"] for d in SEEDS
                         if f"{name}/w{w}/{s}/s{d}" in res
                         and "error" not in res[f"{name}/w{w}/{s}/s{d}"]]
                     for s in STEMS}
            if not any(cells.values()):
                continue
            any_c = next(c for c in cells.values() if c)[0]
            line = f"{w:>4}{any_c['w_over_tac']:>8.2f}{any_c['frac_below_tau']:>7.0%}"
            for s in STEMS:
                c = cells[s]
                line += (f"{np.mean([x['sep'] for x in c]):>10.3f}"
                         f"{np.mean([x['inclusion'] for x in c]):>11.3f}") if c else \
                        f"{'-':>10}{'-':>11}"
            print(line)
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
