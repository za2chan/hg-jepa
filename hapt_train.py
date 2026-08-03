"""HG-JEPA on HAPT inertial windows: does z_slow capture the activity (slow)
and z_fast the instantaneous acceleration (fast)?
Probe uses a SUBJECT-group split (held-out subjects) -- leak-free and doubles
as the domain-robustness test (does z_slow transfer to unseen subjects better
than z_full?).

Usage: python3 hapt_train.py mode=nepa gate=1 dcor=1 seed=0
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge

PATCH, L = 12, 128   # 12 = 4 samples x 3 axes
D_MODEL, D_Z, D_SLOW = 128, 64, 16
OFFSETS = [1, 2, 4, 8, 16, 32]
TAU, W = 5.0, 2.0
EMA = 0.996
STEPS, BATCH, LR = 3000, 64, 3e-4
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load():
    d = np.load("data/hapt.npz")
    return d["W"], d["act"], d["accmag"], d["subj"]


def rankme(Z, eps=1e-7):
    """Effective rank (RankMe, Garrido et al. 2023): collapse monitor."""
    s = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
    p = s / (s.sum() + eps) + eps
    return float(np.exp(-(p * np.log(p)).sum()))


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
    vfloor = args.get("vfloor", "0") == "1"  # D7: off by default since 2026-08-02

    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Wall, act, accmag, subj = load()
    # subject-group split: hold out ~1/3 of subjects entirely (SSL + probe)
    users = np.unique(subj)
    test_u = set(rng.permutation(users)[:max(1, len(users) // 3)].tolist())
    is_test = np.array([u in test_u for u in subj])
    Wt = torch.from_numpy(Wall).to(DEV)
    tr_idx = np.flatnonzero(~is_test)                     # pretrain + probe-fit pool

    # D1 (amended): tau=auto from TRAINING subjects only; taumult= sweeps it
    tau, tau_lbl = TAU, f"tau{TAU:g}"
    taumult = float(args.get("taumult", 1))
    if args.get("tau", "") == "auto":
        from tac import tau_from_tac
        sig = np.linalg.norm(
            Wall[tr_idx].reshape(len(tr_idx), L, 4, 3), axis=-1).reshape(len(tr_idx), -1)
        tac_info = tau_from_tac(sig, patch=4, eps=float(args.get("eps", 0.05)))
        tau = tac_info["tau"] * taumult
        tau_lbl = f"tauauto{taumult:g}x"
    elif "tau" in args:
        tau = float(args["tau"]); tau_lbl = f"tau{tau:g}"
    tag = f"hapt_{mode}_g{int(gated)}_d{int(dcor)}_s{seed}" + \
          ("" if tau_lbl == f"tau{TAU:g}" and "tau" not in args else f"_{tau_lbl}") + \
          ("" if vfloor else "_vf0")

    enc = Encoder().to(DEV)
    pred = Predictor(D_Z if mode == "nepa" else PATCH).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=LR)
    if mode == "nepa":
        tgt = Encoder().to(DEV); tgt.load_state_dict(enc.state_dict())
        for p in tgt.parameters():
            p.requires_grad_(False)
    gvals = torch.sigmoid((tau - torch.tensor(OFFSETS, dtype=torch.float32)) / W).to(DEV)

    n_anchor = 8
    for step in range(STEPS):
        bi_win = torch.from_numpy(rng.choice(tr_idx, BATCH)).to(DEV)  # train subjects only
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
        if vfloor:
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

    # ---- probe: activity (slow) vs instantaneous acc mag (fast) ----
    # SUBJECT-group split: fit on train subjects, test on held-out subjects.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    enc.eval()
    with torch.no_grad():
        Z = torch.cat([enc(Wt[i:i + 256])[:, -1] for i in range(0, len(Wt), 256)]).cpu().numpy()
    tr, te = tr_idx, np.flatnonzero(is_test)
    res = {"tag": tag, "n_test_subj": len(test_u), "rankme": float(rankme(Z)),
           "tau": float(tau), "tau_source": "auto" if args.get("tau") == "auto" else "manual",
           "taumult": taumult}
    for name, sl in [("z_slow", slice(0, D_SLOW)), ("z_fast", slice(D_SLOW, D_Z)),
                     ("z_full", slice(0, D_Z))]:
        B = Z[:, sl]
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(B[tr], act[tr])
        res[f"{name}->activity_f1"] = float(f1_score(act[te], clf.predict(B[te]), average="macro"))
        res[f"{name}->accmag_r2"] = float(Ridge().fit(B[tr], accmag[tr]).score(B[te], accmag[te]))
    print(json.dumps(res, indent=2), flush=True)
    json.dump(res, open(f"runs_hapt/{tag}.json", "w"), indent=2)
    if args.get("save", "0") == "1":                       # for anchor/subset probes
        torch.save({"enc": enc.state_dict(), "test_u": list(test_u)},
                   f"runs_hapt/model_{tag}.pt")


if __name__ == "__main__":
    os.makedirs("runs_hapt", exist_ok=True)
    main(dict(a.split("=") for a in sys.argv[1:]))
