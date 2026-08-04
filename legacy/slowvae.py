"""SlowVAE competitor (Klindt et al. 2021): VAE with Laplace transition prior
on consecutive latents -- reconstruction + slowness, WITH identifiability
theory. Same encoder backbone / context / probe protocol as train.py so the
comparison is architecture-matched. Its "slow" code = the 16 latents with the
smallest normalized temporal variation (the model has no designated block).

Usage: python3 slowvae.py seed=0 [beta=1] [gamma=1]
Writes runs/slowvae_s<seed>.json + runs/emb_slowvae_s<seed>.npz
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge

from datagen import make_dataset
from train import (Encoder, batches, P, L, D_MODEL, D_Z, STEPS, BATCH, LR, DEV)

D_SLOW = 16


class VEnc(Encoder):
    def __init__(self):
        super().__init__()
        self.out_lv = nn.Linear(D_MODEL, D_Z)

    def forward(self, x):
        h = self.embed(x) + self.pos
        h = self.tf(h, mask=self.mask)
        return self.out(h), self.out_lv(h).clamp(-8, 4)


def main(args):
    seed = int(args.get("seed", 0))
    beta = float(args.get("beta", 1))
    gamma = float(args.get("gamma", 1))          # Laplace rate on transitions
    tag = f"slowvae_s{seed}"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_d = make_dataset(1_000_000, seed=0)
    evald = make_dataset(200_000, seed=99)

    enc = VEnc().to(DEV)
    dec = nn.Sequential(nn.Linear(D_Z, 256), nn.GELU(), nn.Linear(256, P)).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=LR)

    for step in range(STEPS):
        xb, _ = batches(train_d["x"], rng, BATCH)
        mu, lv = enc(xb)
        std = (0.5 * lv).exp()
        z = mu + std * torch.randn_like(std)
        recon = ((dec(z) - xb) ** 2).mean()
        logq = (-0.5 * ((z - mu) / std) ** 2 - lv / 2).sum(-1)
        lp0 = (-0.5 * z[:, 0] ** 2).sum(-1)                       # N(0,1) at t=0
        lptr = (-gamma * (z[:, 1:] - z[:, :-1]).abs()).sum(-1)    # Laplace transitions
        kl = (logq.sum(1) - lp0 - lptr.sum(1)).mean() / (L * D_Z)
        loss = recon + beta * kl
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"[{tag}] step {step} recon {recon.item():.4f} kl {kl.item():.4f}",
                  flush=True)

    # ---- probe evaluation (identical leak-free protocol to train.py) ----
    enc.eval()
    x = evald["x"]
    starts = np.arange(0, len(x) - L * P, L * P)
    embs, ys, yu, yphi = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(starts), 64):
            bs = starts[i:i + 64]
            xb = torch.from_numpy(np.stack([x[s:s + L * P].reshape(L, P) for s in bs])).to(DEV)
            mu, _ = enc(xb)
            embs.append(mu.cpu().numpy())                          # (B, L, D_Z)
            end = bs + L * P - 1
            ys.append(evald["s"][end]); yu.append(evald["u"][end])
            yphi.append(np.stack([evald["sin_phi"][end], evald["cos_phi"][end]], 1))
    ZL = np.concatenate(embs)                    # keep sequence to rank slowness
    Z = ZL[:, -1]
    slowness = (np.diff(ZL, axis=1) ** 2).mean((0, 1)) / (ZL.var((0, 1)) + 1e-8)
    slow_idx = np.argsort(slowness)[:D_SLOW]
    fast_idx = np.argsort(slowness)[D_SLOW:]
    ys = np.concatenate(ys); yu = np.concatenate(yu); yphi = np.concatenate(yphi)
    ntr = len(Z) // 2
    res = {"tag": tag, "n_probe": int(len(Z))}
    for name, idx in [("z_slow", slow_idx), ("z_fast", fast_idx),
                      ("z_full", np.arange(D_Z))]:
        B = Z[:, idx]
        res[f"{name}->regime_acc"] = LogisticRegression(max_iter=2000).fit(
            B[:ntr], ys[:ntr]).score(B[ntr:], ys[ntr:])
        res[f"{name}->u_r2"] = Ridge().fit(B[:ntr], yu[:ntr]).score(B[ntr:], yu[ntr:])
        res[f"{name}->phase_r2"] = Ridge().fit(B[:ntr], yphi[:ntr]).score(B[ntr:], yphi[ntr:])
    print(json.dumps(res, indent=2), flush=True)
    json.dump(res, open(f"runs/{tag}.json", "w"), indent=2)
    np.savez(f"runs/emb_{tag}.npz", Z=Z, regime=ys, u=yu, phase=yphi,
             slow_idx=slow_idx)


if __name__ == "__main__":
    main(dict(a.split("=") for a in sys.argv[1:]))
