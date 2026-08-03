"""Post-hoc unmixing baseline (reviewer #1, the decisive control).

JEPA's L2 latent-regression is ~invariant to rotations of the embedding, so an
UNGATED model has no reason to align factors to coordinate blocks -- "no-gate's
slow block is a near-copy" is a symmetry artifact, not evidence. The real test:
if we run LABEL-FREE linear unmixing (PCA / ICA / SFA) on the ungated embedding,
does a low-dim subspace recover the slow factor while excluding the fast ones,
as well as our gate does? If yes, our contribution reduces to a coordinate
choice achievable by post-processing. If no, the gate does something unmixing
cannot.

SFA (Slow Feature Analysis, Wiskott & Sejnowski 2002) is the strongest baseline:
it orders directions by temporal slowness, so "slowest-k" is a label-free pick.

Usage: python3 unmixing.py [seed=0]
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LogisticRegression, Ridge

from train import (Encoder, Predictor, make_dataset, DEV, P, L, D_Z, OFFSETS, W,
                   EMA, STEPS, BATCH, LR, batches)

D_SLOW = 16


def train_ungated(seed, vfloor=False):  # D7: off by default since 2026-08-02
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    train = make_dataset(1_000_000, seed=0)
    enc = Encoder().to(DEV); pred = Predictor(D_Z).to(DEV)
    tgt = Encoder().to(DEV); tgt.load_state_dict(enc.state_dict())
    for p in tgt.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=LR)
    n_anchor = 8
    for step in range(STEPS):
        xb, _ = batches(train["x"], rng, BATCH)
        z = enc(xb)
        anchors = torch.from_numpy(rng.integers(64, L - max(OFFSETS), (BATCH, n_anchor))).to(DEV)
        didx = torch.from_numpy(rng.integers(0, len(OFFSETS), (BATCH, n_anchor))).to(DEV)
        dvals = torch.tensor(OFFSETS, device=DEV)[didx]
        bi = torch.arange(BATCH, device=DEV)[:, None].expand_as(anchors)
        za = z[bi.flatten(), anchors.flatten()]
        zhat = pred(za, didx.flatten())           # NO gate
        with torch.no_grad():
            ztgt = tgt(xb)[bi.flatten(), (anchors + dvals).flatten()]
        loss = ((zhat - ztgt) ** 2).mean()
        if vfloor:
            loss = loss + F.relu(1.0 - z.reshape(-1, D_Z).std(0)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pe, pt in zip(enc.parameters(), tgt.parameters()):
                pt.mul_(EMA).add_(pe, alpha=1 - EMA)
    return enc


def ordered_embeddings(enc, seed=99):
    """Temporally-ordered eval embeddings + ground-truth factors."""
    evald = make_dataset(200_000, seed=seed)
    x = evald["x"]
    starts = np.arange(0, len(x) - L * P, L * P)          # disjoint, time-ordered
    enc.eval(); Z = []
    with torch.no_grad():
        for i in range(0, len(starts), 64):
            bs = starts[i:i + 64]
            xb = torch.from_numpy(np.stack([x[s:s + L * P].reshape(L, P) for s in bs])).to(DEV)
            Z.append(enc(xb)[:, -1].cpu().numpy())
    Z = np.concatenate(Z)
    end = starts + L * P - 1
    y = dict(regime=evald["s"][end], u=evald["u"][end],
             phase=np.stack([evald["sin_phi"][end], evald["cos_phi"][end]], 1))
    return Z, y


def sfa_axes(Z):
    """SFA: directions ordered slow->fast. Returns projection matrix (D, D)."""
    Zc = Z - Z.mean(0)
    cov = np.cov(Zc, rowvar=False) + 1e-4 * np.eye(Z.shape[1])
    d, E = np.linalg.eigh(cov)
    Wh = E @ np.diag(d ** -0.5) @ E.T             # whitening
    Zw = Zc @ Wh
    dZ = np.diff(Zw, axis=0)                       # temporal derivative
    _, Es = np.linalg.eigh(np.cov(dZ, rowvar=False))   # ascending: slowest first
    return Wh @ Es                                 # slow features = Z @ (Wh Es)


def probe(B, y, ntr):
    r = LogisticRegression(max_iter=2000).fit(B[:ntr], y["regime"][:ntr]).score(B[ntr:], y["regime"][ntr:])
    u = Ridge().fit(B[:ntr], y["u"][:ntr]).score(B[ntr:], y["u"][ntr:])
    ph = Ridge().fit(B[:ntr], y["phase"][:ntr]).score(B[ntr:], y["phase"][ntr:])
    return dict(regime=float(r), u=float(u), phase=float(ph))


def main(seed=0):
    enc = train_ungated(seed)
    Z, y = ordered_embeddings(enc)
    ntr = len(Z) // 2                              # contiguous, leak-free
    res = {"seed": seed}

    # baselines: take the slow k-subspace by each label-free method, probe it
    for k in (2, 16):
        # SFA: k slowest directions
        Zsfa = Z @ sfa_axes(Z)
        res[f"SFA_slow{k}"] = probe(Zsfa[:, :k], y, ntr)
        # PCA: top-k variance (not slowness-aware -> should NOT isolate slow)
        Zpca = PCA(n_components=k, random_state=0).fit_transform(Z)
        res[f"PCA_top{k}"] = probe(Zpca, y, ntr)
        # ICA: k independent components; pick the k most slowly varying of them
        ica = FastICA(n_components=min(k * 2, Z.shape[1]), random_state=0, max_iter=1000)
        Zica = ica.fit_transform(Z)
        slowness = np.mean(np.diff(Zica, axis=0) ** 2, axis=0)  # small = slow
        sel = np.argsort(slowness)[:k]
        res[f"ICA_slow{k}"] = probe(Zica[:, sel], y, ntr)

    # reference: raw first-16 block of the SAME ungated embedding (arbitrary split)
    res["ungated_block16"] = probe(Z[:, :D_SLOW], y, ntr)
    print(json.dumps(res, indent=2), flush=True)
    json.dump(res, open(f"runs/unmixing_s{seed}.json", "w"), indent=2)


if __name__ == "__main__":
    args = dict(a.split("=") for a in sys.argv[1:])
    main(int(args.get("seed", 0)))
