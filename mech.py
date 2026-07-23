"""Mechanism isolation (reviewer #5): is the gate doing the work, or is it
capacity pressure / dcor?

(a) target masking (tmask): at long Delta the target's fast dims are
    unpredictable, yet the loss still asks the predictor to fit them. tmask
    down-weights the fast-dim target loss by the same gate g(Delta), so the
    gate is applied symmetrically (input AND target). If gate-alone then
    separates, the earlier plateau was because the gate was only half-applied.
(b) width sweep (dz): widen the embedding (64->128->256) with the slow block
    scaled or fixed. If separation survives, the gate -- not a narrow
    bottleneck -- drives it (matters for wide foundation-model backbones).

Usage: python3 mech.py [dz=64] [dslow=16] [gate=1] [dcor=0] [tmask=0] [seed=0]
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge

from datagen import make_dataset

P, L = 8, 256
D_MODEL = 96
OFFSETS = [1, 4, 16, 64, 128]
TAU, W = 16.0, 4.0
EMA = 0.996
STEPS, BATCH, LR = 3000, 64, 3e-4
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    def __init__(self, d_z):
        super().__init__()
        self.embed = nn.Linear(P, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, L, D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(D_MODEL, 4, 256, batch_first=True,
                                           norm_first=True, dropout=0.0)
        self.tf = nn.TransformerEncoder(layer, 4)
        self.out = nn.Linear(D_MODEL, d_z)
        self.d_z = d_z
        self.register_buffer("mask", torch.triu(torch.full((L, L), float("-inf")), 1))

    def forward(self, x):
        h = self.tf(self.embed(x) + self.pos, mask=self.mask)
        return F.layer_norm(self.out(h), (self.d_z,))


class Predictor(nn.Module):
    def __init__(self, d_z):
        super().__init__()
        self.demb = nn.Embedding(len(OFFSETS), 16)
        self.net = nn.Sequential(nn.Linear(d_z + 16, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU(), nn.Linear(256, d_z))

    def forward(self, z, didx):
        return self.net(torch.cat([z, self.demb(didx)], -1))


def batches(x, rng, n):
    starts = rng.integers(0, len(x) - L * P - 1, n)
    idx = starts[:, None] + np.arange(L * P)[None]
    return torch.from_numpy(x[idx].reshape(n, L, P)).to(DEV)


def main(args):
    d_z = int(args.get("dz", 64))
    d_slow = int(args.get("dslow", 16))
    gated = args.get("gate", "1") == "1"
    dcor = args.get("dcor", "0") == "1"
    tmask = args.get("tmask", "0") == "1"
    seed = int(args.get("seed", 0)); lam = float(args.get("lam", 4))
    tag = f"mech_dz{d_z}_ds{d_slow}_g{int(gated)}_d{int(dcor)}_t{int(tmask)}_s{seed}"

    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    train = make_dataset(1_000_000, seed=0); evald = make_dataset(200_000, seed=99)
    enc = Encoder(d_z).to(DEV); pred = Predictor(d_z).to(DEV)
    tgt = Encoder(d_z).to(DEV); tgt.load_state_dict(enc.state_dict())
    for p in tgt.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=LR)
    gvals = torch.sigmoid((TAU - torch.tensor(OFFSETS, dtype=torch.float32)) / W).to(DEV)

    n_anchor = 8
    for step in range(STEPS):
        xb = batches(train["x"], rng, BATCH)
        z = enc(xb)
        anchors = torch.from_numpy(rng.integers(64, L - max(OFFSETS), (BATCH, n_anchor))).to(DEV)
        didx = torch.from_numpy(rng.integers(0, len(OFFSETS), (BATCH, n_anchor))).to(DEV)
        dvals = torch.tensor(OFFSETS, device=DEV)[didx]
        bi = torch.arange(BATCH, device=DEV)[:, None].expand_as(anchors)
        za = z[bi.flatten(), anchors.flatten()]
        g = gvals[didx.flatten()].unsqueeze(-1)            # (M,1)
        za_in = torch.cat([za[:, :d_slow], za[:, d_slow:] * (g if gated else 1.0)], -1)
        zhat = pred(za_in, didx.flatten())
        with torch.no_grad():
            ztgt = tgt(xb)[bi.flatten(), (anchors + dvals).flatten()]
        if tmask:                                          # gate the fast-dim target loss
            err = (zhat - ztgt) ** 2
            wfast = g.expand(-1, d_z - d_slow)             # (M, d_fast): down-weight at long Delta
            loss = err[:, :d_slow].mean() + (wfast * err[:, d_slow:]).mean()
        else:
            loss = ((zhat - ztgt) ** 2).mean()
        loss = loss + F.relu(1.0 - z.reshape(-1, d_z).std(0)).mean()
        if dcor:
            zc = za - za.mean(0)
            C = (zc[:, :d_slow].T @ zc[:, d_slow:]) / (len(za) - 1)
            loss = loss + lam * (C ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pe, pt in zip(enc.parameters(), tgt.parameters()):
                pt.mul_(EMA).add_(pe, alpha=1 - EMA)

    # ---- leak-free probe (disjoint windows + contiguous split) ----
    enc.eval(); x = evald["x"]
    starts = np.arange(0, len(x) - L * P, L * P)
    Z = []
    with torch.no_grad():
        for i in range(0, len(starts), 64):
            bs = starts[i:i + 64]
            xb = torch.from_numpy(np.stack([x[s:s + L * P].reshape(L, P) for s in bs])).to(DEV)
            Z.append(enc(xb)[:, -1].cpu().numpy())
    Z = np.concatenate(Z); end = starts + L * P - 1
    ys = evald["s"][end]; yu = evald["u"][end]
    yphi = np.stack([evald["sin_phi"][end], evald["cos_phi"][end]], 1)
    ntr = len(Z) // 2
    res = {"tag": tag, "d_z": d_z, "d_slow": d_slow}
    for name, sl in [("z_slow", slice(0, d_slow)), ("z_fast", slice(d_slow, d_z)),
                     ("z_full", slice(0, d_z))]:
        B = Z[:, sl]
        res[f"{name}->regime"] = float(LogisticRegression(max_iter=2000).fit(B[:ntr], ys[:ntr]).score(B[ntr:], ys[ntr:]))
        res[f"{name}->u"] = float(Ridge().fit(B[:ntr], yu[:ntr]).score(B[ntr:], yu[ntr:]))
        res[f"{name}->phase"] = float(Ridge().fit(B[:ntr], yphi[:ntr]).score(B[ntr:], yphi[ntr:]))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    json.dump(res, open(f"runs/{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    os.makedirs("runs", exist_ok=True)
    main(dict(a.split("=") for a in sys.argv[1:]))
