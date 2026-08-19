"""Is the SFA baseline handicapped by a rank defect we introduced?

The encoder ends in a LayerNorm, which fixes each embedding's mean and norm and
so confines the D=64 output to a 62-dimensional manifold. The covariance is
numerically rank-62 (smallest eigenvalues ~6e-15 against a largest of ~8).

SFA minimises the variance of the temporal derivative AFTER whitening. A
direction with no variance has no derivative variance either, so the two dead
directions are ranked SLOWEST and consume two of the d_slow=16 slots that
rotation.py hands to SFA. The baseline has been competing with 14 usable
directions against our 16.

This script re-scores SFA with those directions dropped before whitening, on the
same features and the same protocol rotation.py uses, so the two numbers differ
only in that one change. It writes nothing rotation.py reads; existing result
files are untouched.

Reported per variant, exactly as the paper defines them (probes.sep_index):
  inclusion  = the subspace's persistent-factor macro-F1
  allocation = the complement's transient-factor R^2
  exclusion  = 1 - the subspace's transient-factor R^2
  SEP        = the product, each term clipped to [0, 1]

python3 src/exp/sfa_rank_check.py -> runs_v2/sfa_rank_check.json
"""
import json, pathlib, sys
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "hglp"))
sys.path.insert(0, str(HERE.parent))
from model import Encoder, P, L, D_SLOW, D_Z                          # noqa: E402
from train import DEV                                                 # noqa: E402
from datagen import make_dataset                                      # noqa: E402
from rotation import sfa_basis, two_sided                             # noqa: E402

OUT = HERE.parents[2] / "runs_v2"
GAP = 0.03
CACHE = {0: "model_embedfig_ungated.pt",
         **{s: f"model_sfaseed_ungated_s{s}.pt" for s in range(1, 5)}}
GATED = {0: "model_embedfig_nce_lam4.pt",
         **{s: f"model_sfaseed_gated_s{s}.pt" for s in range(1, 5)}}


def sfa_basis_ranked(X, pairs, tol=1e-9):
    """SFA after dropping the numerically dead directions. Identical to
    rotation.sfa_basis except that whitening is restricted to the directions the
    data actually spans, so a zero-variance direction can no longer win the
    slowness ranking by having nothing to vary."""
    Xc = X - X.mean(0)
    ev, EV = np.linalg.eigh(np.cov(Xc, rowvar=False))
    keep = ev > tol * ev.max()
    Wht = EV[:, keep] / np.sqrt(ev[keep])
    Y = Xc @ Wht
    dY = Y[pairs[:, 1]] - Y[pairs[:, 0]]
    _, DV = np.linalg.eigh(np.cov(dY, rowvar=False))          # ascending = slowest
    return Wht @ DV, int(keep.sum())


def sep(res):
    """(slow-half, fast-half) scores -> the three terms and their product."""
    (incl, leak), (_, alloc) = res
    c = lambda v: float(min(max(v, 0.0), 1.0))
    i, a, e = c(incl), c(alloc), c(1 - leak)
    return dict(inclusion=i, allocation=a, exclusion=e, sep=i * a * e)


def load(name):
    enc = Encoder().to(DEV)
    enc.load_state_dict(torch.load(OUT / name, map_location=DEV))
    enc.eval()
    return enc


def features(enc, xb, pos):
    with torch.no_grad():
        Z = torch.cat([enc(xb[i:i + 128]) for i in range(0, len(xb), 128)]).cpu().numpy()
    return np.concatenate([Z[:, a] for a in pos])


def main():
    # Held-out set built exactly as rotation.run_synth builds it.
    d = make_dataset(300_000, seed=99, gap=GAP)
    x, sreg, u = d["x"], d["s"], d["u"]
    rng = np.random.default_rng(99)
    starts = rng.integers(0, len(x) - L * P - 1, 400)
    xb = torch.from_numpy(np.stack([x[st:st + L * P].reshape(L, P)
                                    for st in starts])).to(DEV)
    pos = list(range(64, 240, 12))
    cut = np.sort(starts)[len(starts) // 2]
    wid = np.tile(np.arange(len(starts)), len(pos))
    tr = np.flatnonzero(starts[wid] + L * P <= cut)
    te = np.flatnonzero(starts[wid] > cut)
    ys = np.concatenate([sreg[starts + a * P + (P - 1)] for a in pos])
    yu = np.concatenate([u[starts + a * P + (P - 1)] for a in pos])

    nw, npos = len(starts), len(pos)
    rank = -np.ones(nw * npos, int); rank[tr] = np.arange(len(tr))
    pm = np.arange(npos)[:, None] * nw + np.arange(nw)[None, :]
    pr = np.stack([rank[pm[p]] for p in range(npos)])
    ok = (pr[:-1] >= 0) & (pr[1:] >= 0)
    pairs = np.stack([pr[:-1][ok], pr[1:][ok]], 1)

    rows = []
    for s in sorted(CACHE):
        Fu = features(load(CACHE[s]), xb, pos)
        Fg = features(load(GATED[s]), xb, pos)
        arg = lambda F: (F[tr], ys[tr], yu[tr], F[te], ys[te], yu[te])

        S0 = sfa_basis(Fu[tr], pairs)                       # as shipped
        S1, kept = sfa_basis_ranked(Fu[tr], pairs)          # dead directions dropped
        I = np.eye(D_Z)
        r = dict(
            seed=s, effective_rank=kept,
            sfa_asis=sep(two_sided(*arg(Fu), (S0[:, :D_SLOW], S0[:, D_SLOW:]), True)),
            sfa_ranked=sep(two_sided(*arg(Fu), (S1[:, :D_SLOW], S1[:, D_SLOW:]), True)),
            gate=sep(two_sided(*arg(Fg), (I[:, :D_SLOW], I[:, D_SLOW:]), True)))
        rows.append(r)
        print(f"seed {s} (rank {kept}/{D_Z})  "
              f"SFA as-is {r['sfa_asis']['sep']:.3f}  "
              f"SFA rank-fixed {r['sfa_ranked']['sep']:.3f}  "
              f"gate {r['gate']['sep']:.3f}", flush=True)

    def agg(k):
        v = np.array([r[k]["sep"] for r in rows])
        return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), n=len(v))

    summary = {k: agg(k) for k in ("sfa_asis", "sfa_ranked", "gate")}
    delta = summary["sfa_ranked"]["mean"] - summary["sfa_asis"]["mean"]
    print(f"\nSEP mean over {len(rows)} seeds")
    for k, v in summary.items():
        print(f"  {k:<12} {v['mean']:.3f} +- {v['sd']:.3f}")
    print(f"  dropping the dead directions changes SFA by {delta:+.3f}")

    (OUT / "sfa_rank_check.json").write_text(json.dumps(dict(
        config=dict(gap=GAP, stem="nce", target="ema", d_slow=D_SLOW, d_z=D_Z,
                    lam_gated=4.0, n_seeds=len(rows),
                    note="ungated control matches rotation.py (gate=False, "
                         "xcov=False, blocknorm=False)"),
        per_seed=rows, summary=summary, delta_sep_from_rank_fix=delta), indent=1))
    print("wrote sfa_rank_check.json")


if __name__ == "__main__":
    main()
