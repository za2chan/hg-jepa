"""HC-NEPA on XJTU raw AM windows: does z_slow capture the envelope (slow)
and z_fast the carrier (fast)? Also a low-pass baseline that, per our thesis,
cannot recover the fault-bearing envelope from the low band.

Usage: python3 raw_am_train.py mode=nepa gate=1 dcor=1 seed=0
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge

PATCH, L = 4, 512
D_MODEL, D_Z, D_SLOW = 128, 64, 16
OFFSETS = [1, 2, 4, 16, 64]
TAU, W = 6.0, 2.0
EMA = 0.996
STEPS, BATCH, LR = 3000, 64, 3e-4
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load():
    d = np.load("data/xjtu_raw.npz")
    bs = sorted({k.split("__")[0] for k in d.files})
    return {b: {k: d[f"{b}__{k}"] for k in ("W", "env", "car", "life")} for b in bs}


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(PATCH, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, L, D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(D_MODEL, 4, 256, batch_first=True,
                                           norm_first=True, dropout=0.0)
        self.tf = nn.TransformerEncoder(layer, 4)
        self.out = nn.Linear(D_MODEL, D_Z)
        self.register_buffer("mask", torch.triu(torch.full((L, L), float("-inf")), 1))

    def forward(self, x):
        h = self.embed(x) + self.pos
        h = self.tf(h, mask=self.mask)
        return F.layer_norm(self.out(h), (D_Z,))


class Predictor(nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.demb = nn.Embedding(len(OFFSETS), 16)
        self.net = nn.Sequential(nn.Linear(D_Z + 16, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU(), nn.Linear(256, d_out))

    def forward(self, z, didx):
        return self.net(torch.cat([z, self.demb(didx)], -1))


def main(args):
    mode = args.get("mode", "nepa")
    gated = args.get("gate", "1") == "1"
    dcor = args.get("dcor", "1") == "1"
    seed = int(args.get("seed", 0))
    lam = float(args.get("lam", 4))
    tag = f"am_{mode}_g{int(gated)}_d{int(dcor)}_s{seed}"

    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    data = load()
    keys = sorted(data)
    Wall = np.concatenate([data[b]["W"] for b in keys])          # (N, L, PATCH)
    env = np.concatenate([data[b]["env"] for b in keys])
    car = np.concatenate([data[b]["car"] for b in keys])
    life = np.concatenate([data[b]["life"] for b in keys])
    test_bearings = set(keys[::3])                # group split for the life probe
    is_te = np.concatenate([np.full(len(data[b]["W"]), b in test_bearings) for b in keys])
    Wt = torch.from_numpy(Wall).to(DEV)

    enc = Encoder().to(DEV)
    pred = Predictor(D_Z if mode == "nepa" else PATCH).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=LR)
    if mode == "nepa":
        tgt = Encoder().to(DEV); tgt.load_state_dict(enc.state_dict())
        for p in tgt.parameters():
            p.requires_grad_(False)
    gvals = torch.sigmoid((TAU - torch.tensor(OFFSETS, dtype=torch.float32)) / W).to(DEV)

    n_anchor = 8
    for step in range(STEPS):
        bi_win = torch.from_numpy(rng.integers(0, len(Wt), BATCH)).to(DEV)
        xb = Wt[bi_win]
        z = enc(xb)
        anchors = torch.from_numpy(rng.integers(32, L - max(OFFSETS), (BATCH, n_anchor))).to(DEV)
        didx = torch.from_numpy(rng.integers(0, len(OFFSETS), (BATCH, n_anchor))).to(DEV)
        dvals = torch.tensor(OFFSETS, device=DEV)[didx]
        bi = torch.arange(BATCH, device=DEV)[:, None].expand_as(anchors)
        za = z[bi.flatten(), anchors.flatten()]
        g = gvals[didx.flatten()].unsqueeze(-1) if gated else 1.0
        za_in = torch.cat([za[:, :D_SLOW], za[:, D_SLOW:] * g], -1)
        zhat = pred(za_in, didx.flatten())
        if mode == "nepa":
            with torch.no_grad():
                ztgt = tgt(xb)[bi.flatten(), (anchors + dvals).flatten()]
        else:
            ztgt = xb[bi.flatten(), (anchors + dvals).flatten()]
        loss = ((zhat - ztgt) ** 2).mean()
        std = z.reshape(-1, D_Z).std(0)
        loss = loss + 1.0 * F.relu(1.0 - std).mean()
        if dcor:
            zc = za - za.mean(0)
            C = (zc[:, :D_SLOW].T @ zc[:, D_SLOW:]) / (len(za) - 1)
            loss = loss + lam * (C ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if mode == "nepa":
            with torch.no_grad():
                for pe, pt in zip(enc.parameters(), tgt.parameters()):
                    pt.mul_(EMA).add_(pe, alpha=1 - EMA)
        if step % 1000 == 0:
            print(f"[{tag}] step {step} loss {loss.item():.4f}", flush=True)

    # ---- probe: envelope (slow) vs carrier (fast) at final patch, in-dist 50/50 ----
    enc.eval()
    with torch.no_grad():
        Z = torch.cat([enc(Wt[i:i + 256])[:, -1] for i in range(0, len(Wt), 256)]).cpu().numpy()
    y_env = env[:, -1]; y_car = car[:, -1]
    perm = rng.permutation(len(Z)); tr, te = perm[:len(Z) // 2], perm[len(Z) // 2:]

    def r2(B, y):
        return float(Ridge().fit(B[tr], y[tr]).score(B[te], y[te]))
    res = {"tag": tag}
    for name, sl in [("z_slow", slice(0, D_SLOW)), ("z_fast", slice(D_SLOW, D_Z)),
                     ("z_full", slice(0, D_Z))]:
        res[f"{name}->envelope_r2"] = r2(Z[:, sl], y_env)
        res[f"{name}->carrier_r2"] = r2(Z[:, sl], y_car)
        # health/RUL proxy, held-out bearings (non-circular target, cf. #6)
        res[f"{name}->life_r2"] = float(Ridge().fit(
            Z[~is_te][:, sl], life[~is_te]).score(Z[is_te][:, sl], life[is_te]))

    # low-pass baseline: mean of low-frequency band of the window cannot recover
    # the envelope, which lives in high-frequency sidebands (AM)
    if mode == "nepa" and gated:
        lp = np.stack([np.fft.irfft(np.fft.rfft(w.reshape(-1))[:L // 8],
                                     n=L * PATCH).reshape(L, PATCH).mean(1) for w in Wall])
        res["lowpass->envelope_r2"] = float(
            Ridge().fit(lp[tr], y_env[tr]).score(lp[te], y_env[te]))
    print(json.dumps(res, indent=2), flush=True)
    json.dump(res, open(f"runs_am/{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    os.makedirs("runs_am", exist_ok=True)
    main(dict(a.split("=") for a in sys.argv[1:]))
