"""HC-NEPA on XJTU-SY bearing snapshot sequences.

Window = L consecutive snapshots of one bearing; patch = decimated waveform (256).
Slow proxy = life fraction (RUL); fast proxy = per-snapshot kurtosis.
SSL pretrains on all bearings; probe eval uses an in-distribution 50/50 split
(matching the synthetic protocol) so blocks are compared on the same footing.

Usage: python3 real_train.py mode=nepa gate=1 dcor=1 seed=0
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score

PATCH = 256
L = 64
D_MODEL, D_Z, D_SLOW = 128, 64, 16
OFFSETS = [1, 2, 4, 8, 16]
TAU, W = 6.0, 2.0
EMA = 0.996
STEPS, BATCH, LR = 3000, 64, 3e-4
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load():
    d = np.load("data/xjtu.npz")
    bearings = sorted({k.split("__")[0] for k in d.files})
    out = {}
    for b in bearings:
        n = len(d[f"{b}__life"])
        if n < L + max(OFFSETS) + 1:
            continue
        out[b] = {k: d[f"{b}__{k}"] for k in ("patch", "life", "kurt")}
    return out


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


def sample(data, keys, rng, n):
    xs, starts, bkeys = [], [], []
    for _ in range(n):
        b = keys[rng.integers(len(keys))]
        p = data[b]["patch"]
        s = rng.integers(0, len(p) - L - max(OFFSETS))
        xs.append(p[s:s + L]); starts.append(s); bkeys.append(b)
    return torch.from_numpy(np.stack(xs)).to(DEV), starts, bkeys


def main(args):
    mode = args.get("mode", "nepa")
    gated = args.get("gate", "1") == "1"
    dcor = args.get("dcor", "1") == "1"
    seed = int(args.get("seed", 0))
    lam = float(args.get("lam", 4))
    tag = f"real_{mode}_g{int(gated)}_d{int(dcor)}_s{seed}"

    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    data = load()
    train_keys = list(data)                       # SSL pretrain on all bearings

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
        xb, _, _ = sample(data, train_keys, rng, BATCH)
        z = enc(xb)
        anchors = torch.from_numpy(rng.integers(16, L - max(OFFSETS), (BATCH, n_anchor))).to(DEV)
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
        # VICReg-style variance floor prevents EMA collapse (per-dim std -> 1)
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

    # ---- eval: probe on held-out bearings; dense windows ----
    enc.eval()
    Z, life, kurt = [], [], []
    with torch.no_grad():
        for b in train_keys:
            p = data[b]["patch"]
            for s in range(0, len(p) - L, 4):
                x = torch.from_numpy(p[s:s + L][None]).to(DEV)
                Z.append(enc(x)[0, -1].cpu().numpy())
                life.append(data[b]["life"][s + L - 1]); kurt.append(data[b]["kurt"][s + L - 1])
    Z = np.array(Z); life = np.array(life); kurt = np.array(kurt)
    # in-distribution 50/50 split (matches synthetic protocol); factor probes per block
    perm = rng.permutation(len(Z)); tr, te = perm[:len(Z) // 2], perm[len(Z) // 2:]
    Ztr, Zte = Z[tr], Z[te]; ltr, lte = life[tr], life[te]; ktr, kte = kurt[tr], kurt[te]
    stage_tr = np.digitize(ltr, [0.5, 0.8]); stage_te = np.digitize(lte, [0.5, 0.8])

    res = {"tag": tag, "n_train": len(Ztr), "n_test": len(Zte)}
    for name, sl in [("z_slow", slice(0, D_SLOW)), ("z_fast", slice(D_SLOW, D_Z)),
                     ("z_full", slice(0, D_Z))]:
        Btr, Bte = Ztr[:, sl], Zte[:, sl]
        rul = Ridge(alpha=1.0).fit(Btr, ltr).predict(Bte)
        res[f"{name}->rul_spearman"] = float(spearmanr(rul, lte)[0])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Btr, stage_tr)
        res[f"{name}->stage_f1"] = float(f1_score(stage_te, clf.predict(Bte), average="macro"))
        res[f"{name}->kurt_r2"] = float(Ridge().fit(Btr, ktr).score(Bte, kte))
    print(json.dumps(res, indent=2), flush=True)
    json.dump(res, open(f"runs_real/{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    import os; os.makedirs("runs_real", exist_ok=True)
    main(dict(a.split("=") for a in sys.argv[1:]))
