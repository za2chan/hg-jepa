"""HC-NEPA on PTB-XL ECG windows: does z_slow capture the diagnosis label
(slow, constant over the record) and z_fast the within-beat waveform (fast)?

Usage: python3 ptbxl_train.py mode=nepa gate=1 dcor=1 seed=0
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge

PATCH, L = 10, 100
D_MODEL, D_Z, D_SLOW = 128, 64, 16
OFFSETS = [1, 2, 4, 8, 16, 32]
TAU, W = 5.0, 2.0
EMA = 0.996
STEPS, BATCH, LR = 3000, 64, 3e-4
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load():
    d = np.load("data/ptbxl.npz")
    return d["W"], d["norm"], d["ecgend"]


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
    tag = f"ptbxl_{mode}_g{int(gated)}_d{int(dcor)}_s{seed}"

    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Wall, norm, ecgend = load()
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

    # ---- probe: diagnosis NORM (slow) vs instantaneous ECG value (fast) ----
    # random 50/50 split is fine here: one window per record, no overlap
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    enc.eval()
    with torch.no_grad():
        Z = torch.cat([enc(Wt[i:i + 256])[:, -1] for i in range(0, len(Wt), 256)]).cpu().numpy()
    perm = rng.permutation(len(Z)); tr, te = perm[:len(Z) // 2], perm[len(Z) // 2:]
    res = {"tag": tag}
    for name, sl in [("z_slow", slice(0, D_SLOW)), ("z_fast", slice(D_SLOW, D_Z)),
                     ("z_full", slice(0, D_Z))]:
        B = Z[:, sl]
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(B[tr], norm[tr])
        res[f"{name}->norm_f1"] = float(f1_score(norm[te], clf.predict(B[te]), average="macro"))
        res[f"{name}->ecg_r2"] = float(Ridge().fit(B[tr], ecgend[tr]).score(B[te], ecgend[te]))
    print(json.dumps(res, indent=2), flush=True)
    json.dump(res, open(f"runs_ptbxl/{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    os.makedirs("runs_ptbxl", exist_ok=True)
    main(dict(a.split("=") for a in sys.argv[1:]))
