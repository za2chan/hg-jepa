"""Which transient proxy on Sleep-EDF actually has measurement power?

The rotation run showed the exclusion term is uninformative there: a random
16-dim subspace already recovers the EOG-energy proxy at R^2 0.347 while the
full 64 dims reach only 0.468, so every method scores exclusion ~0.65 and the
term cannot separate anything. That is the failure mode CLAUDE.md's C4 warns
about ("on datasets where 16 dims naturally carry little of the fast factor,
exclusion is high for everything").

The proxy is EVALUATION-ONLY -- train_real never puts it in the loss -- so one
trained encoder can score every candidate. This pilot trains a single ungated
encoder and, for each candidate proxy, reports

    full   = R^2 from all 64 dims        (is the proxy represented at all?)
    rand16 = R^2 from a random 16 dims   (how much comes for free?)
    room   = full - rand16               (what exclusion can possibly measure)

`room` is the quantity that matters: it is the dynamic range the exclusion term
has to work with. HAPT has 0.559 and PTB-XL 0.590; Sleep-EDF's current proxy has
0.121, which is why nothing separated. Pick the candidate with the largest room
before committing to the full 12-hour re-run.

python3 src/exp/proxy_pilot_sleepedf.py [n_seed]
    -> runs_v2/proxy_pilot_sleepedf.json
"""
import json
import pathlib
import sys

import numpy as np
from scipy import signal as sps
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "hglp"))
OUT = ROOT / "runs_v2"
NPZ = ROOT / "data" / "sleepedf_v2.npz"

FS, PATCH, N_AX, L = 100.0, 50, 3, 256
KW = dict(tau=24.0, w=12, dmin=12, dmax=128, min_context=16, steps=5000)
# (name, channel index, band) -- band None means broadband energy.
# Channels: 0 EEG Fpz-Cz, 1 EEG Pz-Oz, 2 EOG horizontal.
CANDIDATES = [
    ("EOG energy (current)", 2, (0.5, 20.0)),
    ("delta env EEG",        0, (0.5, 4.0)),
    ("alpha env EEG",        0, (8.0, 12.0)),
    ("beta env EEG",         0, (16.0, 30.0)),
    ("EEG broadband energy", 0, None),
    ("EOG env 0.3-4Hz",      2, (0.3, 4.0)),
]


def unpatch(W):
    n, Ln, _ = W.shape
    x = W.reshape(n, Ln, PATCH, N_AX)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2).reshape(n, N_AX, Ln * PATCH))


def make_proxy(x, ch, band):
    """Per-patch proxy, built the same way prep/sleepedf.py builds its own:
    band envelope squared, then log1p of the median-normalised value."""
    v = x[:, ch]
    if band is not None:
        b, a = sps.butter(4, [band[0] / (FS / 2), band[1] / (FS / 2)], btype="band")
        v = np.abs(sps.hilbert(sps.filtfilt(b, a, v, axis=-1), axis=-1))
    e = v ** 2
    e = np.log1p(e / (np.median(e) + 1e-12))
    ends = np.arange(L) * PATCH + (PATCH - 1)          # end-of-patch, as in prep
    return e[:, ends].astype(np.float32)


def r2(Ftr, ztr, Fte, zte):
    m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Ftr, ztr)
    return float(m.score(Fte, zte))


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    from train_real import train_real
    from probes import encode_all, C_MIN, rand_subspace
    from model import D_SLOW, D_Z

    d = np.load(NPZ, allow_pickle=True)
    x = unpatch(d["W"])
    proxies = {nm: make_proxy(x, ch, bd) for nm, ch, bd in CANDIDATES}
    del x
    print(f"windows {len(d['W'])}  candidates {len(proxies)}  seeds {n_seed}")

    acc = {nm: {"full": [], "rand16": []} for nm in proxies}
    for s in range(n_seed):
        r = train_real(str(NPZ), n_ax=N_AX, seed=s, loss_kind="nce", target_enc="ema",
                       gate=False, xcov=False, blocknorm=False,
                       log_every=10 ** 9, **KW)
        Z = encode_all(r["enc"], r["Wt"])
        lab, tr, te = r["lab"], r["tr"], r["te"]
        ok = (lab >= 1) & (lab <= 6)
        ok[:, :C_MIN] = False
        Q = rand_subspace(D_Z, D_SLOW, s)
        for nm, z in proxies.items():
            ztr, zte = z[tr][ok[tr]], z[te][ok[te]]
            Ftr, Fte = Z[tr][ok[tr]], Z[te][ok[te]]
            acc[nm]["full"].append(r2(Ftr, ztr, Fte, zte))
            acc[nm]["rand16"].append(r2(Ftr @ Q, ztr, Fte @ Q, zte))
        print(f"  seed {s} done", flush=True)

    print(f"\n{'proxy':24s}{'full':>8s}{'rand16':>9s}{'room':>8s}")
    res = {}
    for nm in proxies:
        fu, rd = np.mean(acc[nm]["full"]), np.mean(acc[nm]["rand16"])
        res[nm] = dict(full=float(fu), rand16=float(rd), room=float(fu - rd),
                       full_per_seed=acc[nm]["full"], rand16_per_seed=acc[nm]["rand16"])
        print(f"{nm:24s}{fu:8.3f}{rd:9.3f}{fu - rd:8.3f}")
    best = max(res, key=lambda k: res[k]["room"])
    print(f"\nlargest room: {best}  ({res[best]['room']:.3f})")
    print("reference rooms -- HAPT 0.559, PTB-XL 0.590, synthetic 0.022")

    (OUT / "proxy_pilot_sleepedf.json").write_text(json.dumps(
        dict(config=dict(kw=KW, n_seed=n_seed, d_slow=D_SLOW, d_z=D_Z,
                         candidates=[(n, c, b) for n, c, b in CANDIDATES]),
             result=res, best=best), indent=1))
    print("  -> runs_v2/proxy_pilot_sleepedf.json")


if __name__ == "__main__":
    main()
