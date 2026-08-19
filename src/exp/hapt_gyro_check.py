"""Does the gyroscope explain HAPT's walking-class confusion?

`prep/hapt.py` keeps only the 3 accelerometer axes and drops the gyroscope, which
`RawData/` ships alongside it. Static postures separate on the gravity direction
and are fine; walking, walking upstairs and walking downstairs all produce
periodic acceleration of similar amplitude, and trunk angular velocity is much of
what tells them apart. So a low score on those three classes has an explanation
that has nothing to do with the gate, and it should be ruled in or out before the
paper attributes the confusion to the method.

Controlled: `data/hapt_v2_gyro.npz` has the same windows, the same labels and the
same transient proxy as `data/hapt_v2.npz`, with the first three axes identical
byte for byte. The only difference is three extra input channels.

Scoring follows the paper's probe protocol: labelled positions only (1-6), the
C_MIN context floor, and the subject-level group split train_real already makes,
so nothing here is scored on a subject the encoder trained on. Per-class F1 is
reported alongside macro-F1 because the macro average is exactly what hides a
failure confined to two classes. SEP is reported too: more input channels could
raise accuracy while making the separation worse, and that would matter more.

python3 src/exp/hapt_gyro_check.py -> runs_v2/hapt_gyro_check.json
"""
import json, pathlib, sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "hglp"))
from train_real import train_real                                     # noqa: E402
from train import DEV                                                 # noqa: E402
from model import D_SLOW                                              # noqa: E402
from probes import C_MIN, block_factor, sep_index                     # noqa: E402

ROOT = HERE.parents[2]
OUT = ROOT / "runs_v2"
SETS = {"acc3": (ROOT / "data/hapt_v2.npz", 3),
        "acc3+gyro3": (ROOT / "data/hapt_v2_gyro.npz", 6)}
ACTS = ["walking", "walking upstairs", "walking downstairs",
        "sitting", "standing", "lying"]
MAXN = 60000


def probe_set(r, idx):
    """Embeddings, activity label and transient proxy at the SCORED positions."""
    with torch.no_grad():
        Z = torch.cat([r["enc"](r["Wt"][idx][i:i + 64])
                       for i in range(0, len(idx), 64)]).cpu().numpy()
    lab, fast = r["lab"][idx], r["fast"][idx]
    ok = (lab >= 1) & (lab <= 6)
    ok[:, :C_MIN] = False                              # protocol C1
    return Z[ok], lab[ok] - 1, fast[ok]


def per_class(Ftr, ytr, Fte, yte, seed=0):
    rng = np.random.default_rng(seed)
    pick = lambda F, y: (lambda s: (F[s], y[s]))(
        rng.choice(len(F), min(MAXN, len(F)), replace=False))
    Ftr, ytr = pick(Ftr, ytr); Fte, yte = pick(Fte, yte)
    p = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
    pred = p.fit(Ftr, ytr).predict(Fte)
    lb = np.arange(6)
    f = f1_score(yte, pred, average=None, labels=lb, zero_division=0)
    return dict(macro=float(f1_score(yte, pred, average="macro", labels=lb,
                                     zero_division=0)),
                per_class={ACTS[i]: float(v) for i, v in enumerate(f)},
                n_test=int(len(yte)))


def main():
    res = {}
    for tag, (npz, n_ax) in SETS.items():
        for stem in ("nce", "l1"):
            key = f"{tag}/{stem}"
            print(f"training {key} ...", flush=True)
            r = train_real(str(npz), n_ax=n_ax, seed=0, loss_kind=stem,
                           target_enc="ema", gate=True, xcov=True, lam=4.0,
                           log_every=10 ** 9)
            a, b = probe_set(r, r["tr"]), probe_set(r, r["te"])
            pc = per_class(a[0][:, :D_SLOW], a[1], b[0][:, :D_SLOW], b[1])
            bf = block_factor(a[0], a[1], a[2], b[0], b[1], b[2], seed=0)
            # Keep the WHOLE matrix, not just the index. z_full is the ceiling:
            # how much of the proxy the embedding holds anywhere. Without it an
            # allocation drop cannot be told apart from the model simply encoding
            # less of that particular proxy.
            pc["block_factor"] = {k: dict(persistent=v[0], transient=v[1])
                                  for k, v in bf.items()}
            pc["sep"] = sep_index(bf)
            res[key] = pc
            print(f"  macro-F1 {pc['macro']:.3f}   SEP {pc['sep']['sep']:.3f}   "
                  f"z_full proxy R2 {bf['z_full'][1]:.3f}   "
                  + "  ".join(f"{k.split()[-1]} {v:.2f}"
                              for k, v in pc["per_class"].items()), flush=True)

    print(f"\n{'class':<20}" + "".join(f"{k:>16}" for k in res))
    for i, a_ in enumerate(ACTS):
        print(f"{a_:<20}" + "".join(f"{res[k]['per_class'][a_]:>16.3f}" for k in res))
    print(f"{'macro-F1':<20}" + "".join(f"{res[k]['macro']:>16.3f}" for k in res))
    print(f"{'SEP':<20}" + "".join(f"{res[k]['sep']['sep']:>16.3f}" for k in res))

    (OUT / "hapt_gyro_check.json").write_text(json.dumps(dict(
        config=dict(datasets={k: str(v[0].name) for k, v in SETS.items()},
                    stems=["nce", "l1"], gate=True, xcov=True, lam=4.0, tau=40.0,
                    seed=0, note="same windows/labels/proxy; only input channels "
                                 "differ. Scored on the held-out subject split."),
        results=res), indent=1))
    print("\nwrote hapt_gyro_check.json")


if __name__ == "__main__":
    main()
