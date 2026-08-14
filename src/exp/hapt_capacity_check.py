"""Is the encoder simply too small to take six channels?

Doubling the input channels adds 1,152 parameters -- the input projection goes
from Linear(12, 96) to Linear(24, 96). The trunk (4 layers, d_model 96, d_ff 256)
and the embedding (D_Z 64) are identical, 355k parameters either way. So when the
6-axis run held LESS of the transient proxy anywhere in its embedding (z_full R2
0.763 -> 0.623 for NCE), capacity dilution is the obvious suspect.

The test widens the TRUNK only (d_model 96 -> 192, d_ff 256 -> 512) and leaves
D_Z = 64 and d_slow = 16 alone, so the block geometry SEP depends on is unchanged
and the numbers stay comparable. Both channel sets are run at both capacities: if
width helps the 6-axis run specifically, the dilution story holds; if it helps
both equally, the model was just small and channels had nothing to do with it.

Caveat to carry into any conclusion: the learning rate, step count and every
other hyper-parameter are held at the values tuned for the narrow model, so the
wide runs are not tuned in their own right. This bounds the effect from below,
not from above.

python3 src/exp/hapt_capacity_check.py -> runs_v2/hapt_capacity_check.json
"""
import json, pathlib, sys
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "hglp"))
sys.path.insert(0, str(HERE.parent))
import model as M                                                     # noqa: E402
from probes import block_factor, sep_index                            # noqa: E402
from hapt_gyro_check import probe_set, per_class, SETS                # noqa: E402

OUT = HERE.parents[2] / "runs_v2"
CAPS = {"base": dict(D_MODEL=96, D_FF=256), "wide": dict(D_MODEL=192, D_FF=512)}


def run(npz, n_ax, stem, cap):
    old = {k: getattr(M, k) for k in CAPS["base"]}
    for k, v in CAPS[cap].items():
        setattr(M, k, v)
    try:
        import importlib, train_real as TR
        importlib.reload(TR)                       # rebind Encoder to patched globals
        r = TR.train_real(str(npz), n_ax=n_ax, seed=0, loss_kind=stem,
                          target_enc="ema", gate=True, xcov=True, lam=4.0,
                          log_every=10 ** 9)
        n_par = sum(p.numel() for p in r["enc"].parameters())
        a, b = probe_set(r, r["tr"]), probe_set(r, r["te"])
        pc = per_class(a[0][:, :M.D_SLOW], a[1], b[0][:, :M.D_SLOW], b[1])
        bf = block_factor(a[0], a[1], a[2], b[0], b[1], b[2], seed=0)
        pc["sep"] = sep_index(bf)
        pc["z_full_proxy_r2"] = float(bf["z_full"][1])
        pc["params"] = int(n_par)
        return pc
    finally:
        for k, v in old.items():
            setattr(M, k, v)


def main():
    res = {}
    for tag, (npz, n_ax) in SETS.items():
        for cap in CAPS:
            for stem in ("nce", "l1"):
                key = f"{tag}/{cap}/{stem}"
                print(f"training {key} ...", flush=True)
                p = run(npz, n_ax, stem, cap)
                res[key] = p
                print(f"  params {p['params']:,}  macro-F1 {p['macro']:.3f}  "
                      f"SEP {p['sep']['sep']:.3f}  "
                      f"z_full proxy R2 {p['z_full_proxy_r2']:.3f}", flush=True)

    print(f"\n{'':<14}" + "".join(f"{c:>22}" for c in ("base", "wide")))
    for tag in SETS:
        for stem in ("nce", "l1"):
            row = [res[f"{tag}/{c}/{stem}"] for c in CAPS]
            print(f"{tag+'/'+stem:<14}" +
                  "".join(f"  F1 {r['macro']:.3f} SEP {r['sep']['sep']:.3f} "
                          f"ceil {r['z_full_proxy_r2']:.3f}" for r in row))

    (OUT / "hapt_capacity_check.json").write_text(json.dumps(dict(
        config=dict(caps=CAPS, d_z=M.D_Z, d_slow=M.D_SLOW, seed=0, lam=4.0,
                    note="trunk width only; D_Z and d_slow fixed so SEP stays "
                         "comparable. Other hyper-parameters not retuned."),
        results=res), indent=1))
    print("\nwrote hapt_capacity_check.json")


if __name__ == "__main__":
    main()
