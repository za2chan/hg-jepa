"""Synthetic pilot: horizon-gated latent prediction (HG-JEPA) vs raw-signal
prediction (AR / next-token family), on two-timescale synthetic data.

Usage: python3 train.py mode=nepa gate=1 dcor=0 seed=0 [tau=16] [dslow=16] [lam=4]
  mode=nepa : predict future EMA-encoder embeddings (JEPA family)
  mode=ar   : predict future raw patches (next-token / reconstruction family)
  mode=cpc  : latent-contrastive (InfoNCE vs EMA embeddings, in-batch negatives)
  gate=1    : predictor's access to z_fast decays for horizons beyond tau
  dcor=1    : cross-covariance penalty between z_slow / z_fast blocks
Probe eval uses disjoint windows + a contiguous time split (leak-free).
Writes runs/<tag>.json with block-factor probe matrix.
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge

from datagen import make_dataset

P = 8                  # patch length (steps)
L = 256                # sequence length (patches) -> 2048-step window
D_MODEL, D_Z = 96, 64
OFFSETS = [1, 4, 16, 64, 128]  # horizons (patches) = 8..1024 steps
W = 4.0                        # gate softness (patches)
EMA = 0.996
STEPS, BATCH, LR = 3000, 64, 3e-4
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(P, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, L, D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(D_MODEL, 4, 256, batch_first=True,
                                           norm_first=True, dropout=0.0)
        self.tf = nn.TransformerEncoder(layer, 4)
        self.out = nn.Linear(D_MODEL, D_Z)
        mask = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):                      # x: (B, L, P)
        h = self.embed(x) + self.pos
        h = self.tf(h, mask=self.mask)
        return F.layer_norm(self.out(h), (D_Z,))


class Predictor(nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.demb = nn.Embedding(len(OFFSETS), 16)
        self.net = nn.Sequential(nn.Linear(D_Z + 16, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU(),
                                 nn.Linear(256, d_out))

    def forward(self, z, didx):
        return self.net(torch.cat([z, self.demb(didx)], -1))


def batches(x, rng, n):
    starts = rng.integers(0, len(x) - L * P - 1, n)
    idx = starts[:, None] + np.arange(L * P)[None]
    return torch.from_numpy(x[idx].reshape(n, L, P)).to(DEV), starts


def main(args):
    mode = args.get("mode", "nepa")
    gated = args.get("gate", "1") == "1"
    dcor = args.get("dcor", "0") == "1"
    seed = int(args.get("seed", 0))
    tau = float(args.get("tau", 16))
    d_slow = int(args.get("dslow", 16))
    lam = float(args.get("lam", 4))
    tag = f"{mode}_g{int(gated)}_d{int(dcor)}_s{seed}_tau{int(tau)}_ds{d_slow}_lam{int(lam)}"

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # data fixed across seeds by design; seeds vary only init + sampling order
    train = make_dataset(1_000_000, seed=0)
    evald = make_dataset(200_000, seed=99)

    enc = Encoder().to(DEV)
    pred = Predictor(P if mode == "ar" else D_Z).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=LR)
    if mode != "ar":
        tgt = Encoder().to(DEV)
        tgt.load_state_dict(enc.state_dict())
        for p in tgt.parameters():
            p.requires_grad_(False)

    gvals = torch.sigmoid((tau - torch.tensor(OFFSETS, dtype=torch.float32)) / W).to(DEV)
    n_anchor = 8
    for step in range(STEPS):
        xb, _ = batches(train["x"], rng, BATCH)
        z = enc(xb)
        anchors = torch.from_numpy(
            rng.integers(64, L - max(OFFSETS), (BATCH, n_anchor))).to(DEV)
        didx = torch.from_numpy(
            rng.integers(0, len(OFFSETS), (BATCH, n_anchor))).to(DEV)
        dvals = torch.tensor(OFFSETS, device=DEV)[didx]
        bi = torch.arange(BATCH, device=DEV)[:, None].expand_as(anchors)
        za = z[bi.flatten(), anchors.flatten()]
        g = gvals[didx.flatten()].unsqueeze(-1) if gated else 1.0
        za_in = torch.cat([za[:, :d_slow], za[:, d_slow:] * g], -1)
        zhat = pred(za_in, didx.flatten())
        if mode == "ar":                        # predict raw future patch
            ztgt = xb[bi.flatten(), (anchors + dvals).flatten()]
        else:
            with torch.no_grad():
                ztgt = tgt(xb)[bi.flatten(), (anchors + dvals).flatten()]
        if mode == "cpc":                       # InfoNCE, in-batch negatives
            logits = F.normalize(zhat, dim=-1) @ F.normalize(ztgt, dim=-1).T / 0.1
            loss = F.cross_entropy(logits, torch.arange(len(zhat), device=DEV))
        else:
            loss = ((zhat - ztgt) ** 2).mean()
        loss = loss + F.relu(1.0 - z.reshape(-1, D_Z).std(0)).mean()  # anti-collapse
        if dcor:
            zc = za - za.mean(0)
            C = (zc[:, :d_slow].T @ zc[:, d_slow:]) / (len(za) - 1)
            loss = loss + lam * (C ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if mode != "ar":
            with torch.no_grad():
                for pe, pt in zip(enc.parameters(), tgt.parameters()):
                    pt.mul_(EMA).add_(pe, alpha=1 - EMA)
        if step % 1000 == 0:
            zs = z.reshape(-1, D_Z).std(0)
            print(f"[{tag}] step {step} loss {loss.item():.4f} "
                  f"std {zs[:d_slow].mean():.3f}/{zs[d_slow:].mean():.3f}", flush=True)

    # ---- probe evaluation ----
    # Leak-free: tile the eval series into DISJOINT windows and split by a
    # contiguous time cut, so no probe-train window overlaps a probe-test one.
    enc.eval()
    x = evald["x"]
    starts = np.arange(0, len(x) - L * P, L * P)          # non-overlapping
    embs, ys, yu, yphi = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(starts), 64):
            bs = starts[i:i + 64]
            xb = torch.from_numpy(np.stack([x[s:s + L * P].reshape(L, P) for s in bs])).to(DEV)
            embs.append(enc(xb)[:, -1].cpu().numpy())
            end = bs + L * P - 1
            ys.append(evald["s"][end]); yu.append(evald["u"][end])
            yphi.append(np.stack([evald["sin_phi"][end], evald["cos_phi"][end]], 1))
    Z = np.concatenate(embs); ys = np.concatenate(ys)
    yu = np.concatenate(yu); yphi = np.concatenate(yphi)
    ntr = len(Z) // 2                                     # contiguous cut (early=train)
    res = {"tag": tag, "n_probe": int(len(Z))}
    for name, sl in [("z_slow", slice(0, d_slow)), ("z_fast", slice(d_slow, D_Z)),
                     ("z_full", slice(0, D_Z))]:
        B = Z[:, sl]
        res[f"{name}->regime_acc"] = LogisticRegression(max_iter=2000).fit(
            B[:ntr], ys[:ntr]).score(B[ntr:], ys[ntr:])
        res[f"{name}->u_r2"] = Ridge().fit(B[:ntr], yu[:ntr]).score(B[ntr:], yu[ntr:])
        res[f"{name}->phase_r2"] = Ridge().fit(B[:ntr], yphi[:ntr]).score(B[ntr:], yphi[ntr:])
    print(json.dumps(res, indent=2), flush=True)
    json.dump(res, open(f"runs/{tag}.json", "w"), indent=2)
    np.savez(f"runs/emb_{tag}.npz", Z=Z, regime=ys, u=yu, phase=yphi)  # for DCI/MIG


if __name__ == "__main__":
    import os; os.makedirs("runs", exist_ok=True)
    main(dict(a.split("=") for a in sys.argv[1:]))
